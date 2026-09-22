"""Nested seed-held-out audit-to-action transfer, with audit-only layer selection.

This is a post hoc revision analysis, not a preregistered protocol. The outer
action test data are never used to select layers or refit/recalibrate a probe.
Only NumPy and scikit-learn are needed; no model forward or GPU is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    "runs/qwen-7b-v2", "runs/mistral-7b-v2", "runs/hermes-8b-v2",
    "runs/llama-8b-neutral-v2",
]
DIRECTIONS = [("imp", "int"), ("int", "imp")]


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def cache_hashes(run_dir: Path) -> dict:
    files = {"forward": run_dir / "forward.npz", "audit_hidden": run_dir / "audit_hidden.npy"}
    if (run_dir / "forward_meta.json").exists():
        files["forward_meta"] = run_dir / "forward_meta.json"
    return {name: sha256(path) for name, path in files.items()}


def safe_auc(y: np.ndarray, score: np.ndarray, weight=None) -> float | None:
    if weight is not None:
        present = weight > 0
        if len(np.unique(y[present])) != 2:
            return None
    elif len(np.unique(y)) != 2:
        return None
    return float(roc_auc_score(y, score, sample_weight=weight))


def seed_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups).copy()
    if len(unique) < n_splits:
        raise ValueError(f"{len(unique)} seeds cannot support {n_splits} folds")
    np.random.default_rng(seed).shuffle(unique)
    folds = []
    for test_seeds in np.array_split(unique, n_splits):
        test = np.flatnonzero(np.isin(groups, test_seeds))
        train = np.flatnonzero(~np.isin(groups, test_seeds))
        assert not set(groups[train]) & set(groups[test])
        folds.append((train, test))
    return folds


def fit_probe(X: np.ndarray, y: np.ndarray, seed: int, C: float, max_iter: int):
    if len(np.unique(y)) != 2:
        raise ValueError("Single-class training fold")
    scaler = StandardScaler().fit(X)
    classifier = LogisticRegression(
        C=C, solver="liblinear", dual=True, max_iter=max_iter,
        random_state=seed, tol=1e-4,
    )
    # A failed fit should stop the run rather than silently yield a paper number.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.transform(X), y)
    return scaler, classifier


def frozen_score(scaler, classifier, X: np.ndarray) -> np.ndarray:
    return classifier.decision_function(scaler.transform(X))


def fold_details(train, test, ids, groups) -> dict:
    return {
        "train_seed_ids": sorted(set(groups[train].tolist())),
        "test_seed_ids": sorted(set(groups[test].tolist())),
        "train_item_ids": ids[train].tolist(),
        "test_item_ids": ids[test].tolist(),
    }


def conditional_seed_bootstrap(y, scores, groups, fold_ids, n_boot, seed) -> dict:
    """Cluster-resample seeds inside each outer fold, then average fold AUROCs.

    Each seed contributes all its examples together. Stratifying by outer fold
    preserves the held-out fitted model attached to every prediction. This is
    conditional evaluation uncertainty: probes/layers/splits are NOT refitted.
    """
    rng = np.random.default_rng(seed)
    folds = np.unique(fold_ids)
    draws = []
    for _ in range(n_boot):
        aucs = []
        for fold in folds:
            mask = fold_ids == fold
            unique, inverse = np.unique(groups[mask], return_inverse=True)
            multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            auc = safe_auc(y[mask], scores[mask], multiplicity[inverse])
            if auc is None:
                break
            aucs.append(auc)
        if len(aucs) == len(folds):
            draws.append(float(np.mean(aucs)))
    if n_boot and len(draws) < 0.95 * n_boot:
        raise ValueError("Too many single-class bootstrap draws; revise resampling protocol")
    interval = np.quantile(draws, [0.025, 0.975]).tolist() if draws else None
    return {
        "confidence_level": 0.95,
        "method": "percentile semantic-seed cluster bootstrap stratified by outer fold",
        "statistic": "unweighted mean of outer-fold AUROCs",
        "n_requested": n_boot, "n_valid": len(draws), "rng_seed": seed,
        "interval": interval,
        "interpretation": "Conditional on fitted out-of-fold predictions, selected layers, and "
                          "fixed splits; excludes retraining, layer-selection, and split uncertainty.",
        "draws": draws,
    }


def run_model(run_dir: Path, output: Path, items_by_id: dict, args) -> dict:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {"forward": run_dir / "forward.npz", "audit_hidden": run_dir / "audit_hidden.npy"}
    with np.load(input_paths["forward"], allow_pickle=True) as fwd:
        ids = np.asarray(fwd["item_ids"], dtype=str)
        action = fwd["H"]
    audit = np.load(input_paths["audit_hidden"], mmap_mode="r", allow_pickle=False)
    if action.shape != audit.shape or action.ndim != 3 or len(ids) != len(action):
        raise ValueError(f"Invalid or inconsistent cache shapes: {action.shape}, {audit.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item IDs in forward cache")
    missing = set(ids) - set(items_by_id)
    if missing:
        raise ValueError(f"Missing dataset item IDs: {sorted(missing)}")
    items = [items_by_id[item] for item in ids]
    labels = np.asarray([item["intent"] for item in items], dtype=int)
    forms = np.asarray([item["form"] for item in items])
    groups = np.asarray([item["pair_id"] for item in items])
    if set(labels) != {0, 1} or set(forms) != {"imp", "int"}:
        raise ValueError("Expected binary intent and imp/int forms")

    # Smoke tests deliberately run an incomplete dataset and mark the results.
    if args.max_seeds is not None:
        selected = np.unique(groups)[:args.max_seeds]
        keep = np.flatnonzero(np.isin(groups, selected))
        ids, labels, forms, groups = ids[keep], labels[keep], forms[keep], groups[keep]
        action, audit = action[keep], audit[keep]
    if not np.isfinite(action).all() or not np.isfinite(audit).all():
        raise ValueError("Non-finite cached hidden state")
    if args.layers:
        layers = sorted({int(value) for value in args.layers.split(",")})
    elif args.layer_grid == "nine":
        layers = np.unique(np.linspace(0, action.shape[1] - 1, 9).round().astype(int)).tolist()
    else:
        layers = list(range(action.shape[1]))
    if not layers or min(layers) < 0 or max(layers) >= action.shape[1]:
        raise ValueError(f"Layer list {layers} does not match {action.shape[1]} cached layers")

    outer = seed_folds(groups, args.outer_folds, args.seed)
    split_records = []
    score_records = []
    direction_summaries = {}
    for direction_index, (source_form, target_form) in enumerate(DIRECTIONS):
        direction = f"{source_form}2{target_form}"
        outer_action_scores = np.full(len(ids), np.nan)
        outer_audit_scores = np.full(len(ids), np.nan)
        outer_fold_ids = np.full(len(ids), -1)
        fold_results = []
        for outer_index, (outer_train, outer_test) in enumerate(outer):
            inner_seed = args.seed + 1000 + outer_index
            inner_local = seed_folds(groups[outer_train], args.inner_folds, inner_seed)
            inner = [(outer_train[train], outer_train[test]) for train, test in inner_local]
            inner_scores = np.full((len(ids), len(layers)), np.nan)
            inner_fold_ids = np.full(len(ids), -1)
            layer_aucs = np.zeros((len(inner), len(layers)))
            inner_splits = []
            for inner_index, (inner_train, inner_test) in enumerate(inner):
                train = inner_train[forms[inner_train] == source_form]
                test = inner_test[forms[inner_test] == target_form]
                if len(np.unique(labels[test])) != 2:
                    raise ValueError("Single-class inner validation fold; cannot select by AUROC")
                inner_fold_ids[test] = inner_index
                inner_splits.append({"fold": inner_index, **fold_details(train, test, ids, groups)})
                for layer_index, layer in enumerate(layers):
                    scaler, classifier = fit_probe(
                        audit[train, layer].astype(np.float32), labels[train],
                        args.seed, args.C, args.max_iter,
                    )
                    scores = frozen_score(scaler, classifier, audit[test, layer].astype(np.float32))
                    inner_scores[test, layer_index] = scores
                    layer_aucs[inner_index, layer_index] = safe_auc(labels[test], scores)
            means = layer_aucs.mean(axis=0)
            # An exact/tiny numerical tie is broken toward the shallowest layer.
            chosen_index = int(np.flatnonzero(means >= means.max() - 1e-12)[0])
            chosen_layer = layers[chosen_index]
            train = outer_train[forms[outer_train] == source_form]
            test = outer_test[forms[outer_test] == target_form]
            if len(np.unique(labels[test])) != 2:
                raise ValueError("Single-class outer test fold")
            assert not set(groups[train]) & set(groups[test])
            scaler, classifier = fit_probe(
                audit[train, chosen_layer].astype(np.float32), labels[train],
                args.seed, args.C, args.max_iter,
            )
            scores = frozen_score(scaler, classifier, action[test, chosen_layer].astype(np.float32))
            audit_scores = frozen_score(scaler, classifier, audit[test, chosen_layer].astype(np.float32))
            outer_action_scores[test], outer_audit_scores[test] = scores, audit_scores
            outer_fold_ids[test] = outer_index
            prefix = f"{direction}_outer{outer_index}"
            inner_valid = inner_fold_ids >= 0
            np.savez_compressed(
                output / f"{prefix}_inner_scores.npz", item_ids=ids[inner_valid],
                labels=labels[inner_valid], seed_ids=groups[inner_valid],
                inner_fold_ids=inner_fold_ids[inner_valid], layers=np.asarray(layers),
                scores=inner_scores[inner_valid], fold_layer_auroc=layer_aucs,
            )
            np.savez_compressed(
                output / f"{prefix}_fitted_probe.npz", layer=np.asarray(chosen_layer),
                scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                scaler_var=scaler.var_, coef=classifier.coef_, intercept=classifier.intercept_,
                classes=classifier.classes_, n_iter=classifier.n_iter_,
            )
            result = {
                "fold": outer_index, "selected_layer": chosen_layer,
                "n_train_items": len(train), "n_test_items": len(test),
                "n_train_seeds": len(np.unique(groups[train])),
                "n_test_seeds": len(np.unique(groups[test])),
                "n_test_positive": int(labels[test].sum()),
                "n_test_negative": int(len(test) - labels[test].sum()),
                "inner_mean_auroc_by_layer": {str(layer): float(value) for layer, value in zip(layers, means)},
                "inner_fold_auroc_by_layer": layer_aucs.tolist(),
                "selected_inner_mean_auroc": float(means[chosen_index]),
                "action_test_auroc": safe_auc(labels[test], scores),
                "audit_test_auroc": safe_auc(labels[test], audit_scores),
            }
            fold_results.append(result)
            split_records.append({
                "direction": direction, "outer_fold": outer_index,
                **fold_details(train, test, ids, groups), "inner_rng_seed": inner_seed,
                "inner_folds": inner_splits,
            })
            for index, score, audit_score in zip(test, scores, audit_scores):
                score_records.append({
                    "item_id": str(ids[index]), "pair_id": str(groups[index]),
                    "form": str(forms[index]), "intent": int(labels[index]),
                    "direction": direction, "outer_fold": outer_index,
                    "selected_layer": chosen_layer, "action_score": float(score),
                    "audit_score": float(audit_score),
                })
            print(f"{run_dir.name} {direction} outer={outer_index} layer={chosen_layer} "
                  f"inner={means[chosen_index]:.4f} action={result['action_test_auroc']:.4f}", flush=True)
            # Preserve completed folds if a later fit fails or a job is interrupted.
            dump_json(output / "splits.json", split_records)
            dump_json(output / "fold_results.partial.json", {
                **direction_summaries, direction: {"folds": fold_results},
            })

        mask = forms == target_form
        assert np.isfinite(outer_action_scores[mask]).all()
        assert (outer_fold_ids[mask] >= 0).all()
        bootstrap = conditional_seed_bootstrap(
            labels[mask], outer_action_scores[mask], groups[mask], outer_fold_ids[mask],
            args.bootstrap, args.seed + 2000 + direction_index,
        )
        np.save(output / f"{direction}_conditional_bootstrap.npy", np.asarray(bootstrap.pop("draws")))
        action_aucs = [fold["action_test_auroc"] for fold in fold_results]
        audit_aucs = [fold["audit_test_auroc"] for fold in fold_results]
        direction_summaries[direction] = {
            "source_context": "audit", "source_form": source_form,
            "target_context": "action", "target_form": target_form,
            "folds": fold_results,
            "primary_mean_fold_auroc": float(np.mean(action_aucs)),
            "fold_auroc_sd_descriptive": float(np.std(action_aucs, ddof=1)),
            "fold_auroc_range": [float(min(action_aucs)), float(max(action_aucs))],
            "pooled_oof_auroc_diagnostic_only": safe_auc(labels[mask], outer_action_scores[mask]),
            "pooled_warning": "Scores from different fitted outer-fold probes need not be calibrated "
                              "on a common scale. Pooled AUROC is not the primary estimate.",
            "audit_target_form_mean_fold_auroc": float(np.mean(audit_aucs)),
            "conditional_seed_bootstrap": bootstrap,
        }
    with (output / "item_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(score_records, key=lambda x: (x["direction"], x["outer_fold"], x["item_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if len(score_records) != len(ids):
        raise AssertionError("Expected every cached item to appear once as an outer action test item")
    summary = {
        "run_dir": str(run_dir), "n_items": len(ids), "n_seeds": len(np.unique(groups)),
        "cache_shape_after_subset": list(action.shape), "candidate_layers": layers,
        "input_sha256": cache_hashes(run_dir),
        "forward_metadata": json.loads((run_dir / "forward_meta.json").read_text(encoding="utf-8"))
                            if (run_dir / "forward_meta.json").exists() else None,
        "audit_order_provenance": "audit_hidden.npy has no embedded item IDs. Alignment follows "
                                  "the forward.npz item_ids order used by scripts/09_audit_transfer.py; "
                                  "shape is checked but historical extraction order cannot be independently "
                                  "verified from the .npy cache alone.",
        "results": direction_summaries, "elapsed_seconds": time.monotonic() - started,
    }
    dump_json(output / "summary.json", summary)
    (output / "fold_results.partial.json").unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    parser.add_argument("--out-dir", default="runs/revision-20260910/nested_transfer")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--layer-grid", choices=["all", "nine"], default="all")
    parser.add_argument("--layers", help="Explicit comma-separated layer indices (for smoke tests/sensitivity)")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--max-seeds", type=int, help="Subset for a smoke test; result marked incomplete")
    parser.add_argument("--resume", action="store_true", help="Reuse complete model outputs with matching protocol/input hashes")
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.bootstrap < 0 or args.C <= 0:
        parser.error("fold counts must be >=2, bootstrap >=0, and C >0")
    run_dirs = [resolve(path) for path in args.run_dirs]
    if len({path.name for path in run_dirs}) != len(run_dirs):
        parser.error("run directory basenames must be unique")
    items_path = resolve(args.items)
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items_by_id = {item["item_id"]: item for item in items}
    if len(items_by_id) != len(items):
        raise ValueError("Duplicate dataset item IDs")
    output = resolve(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "analysis": "nested audit-to-action cross-form transfer",
        "status": "post hoc revision analysis; not preregistered",
        "smoke_test_incomplete": args.max_seeds is not None,
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "layer_grid": args.layer_grid, "explicit_layers": args.layers,
        "rng_seed": args.seed, "C": args.C, "max_iter": args.max_iter,
        "n_bootstrap": args.bootstrap, "max_seeds": args.max_seeds,
        "split": "shuffled sorted unique semantic seeds, then np.array_split; shared across directions/models",
        "selection": "mean inner-fold AUROC for audit(source form)→audit(target form); "
                     "ties within 1e-12 choose shallowest layer; no outer action data used",
        "fit": "StandardScaler fitted only on audit source-form training items; L2 LogisticRegression "
               "C=1 by default, liblinear dual, tol=1e-4; scaler and probe frozen at evaluation",
        "primary_statistic": "unweighted mean of outer-fold action AUROCs, directions reported separately",
        "inference_limit": "bootstrap conditional on fitted predictions; no uncertainty from retraining, "
                           "layer selection, split choice, or earlier examination of these data",
        "items_sha256": sha256(items_path), "script_sha256": sha256(Path(__file__)),
        "current_model_config_sha256": sha256(ROOT / "configs/models.yaml"),
        "cache_extraction_script_sha256": sha256(ROOT / "scripts/09_audit_transfer.py"),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("Output directory has a different protocol; choose a new --out-dir")
    dump_json(protocol_path, protocol)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "scipy"]},
        "items_path": str(items_path), "run_dirs": [str(path) for path in run_dirs],
    }
    dump_json(output / "invocation.json", metadata)
    results = {}
    for run_dir in run_dirs:
        model_output = output / run_dir.name
        complete = model_output / "summary.json"
        if args.resume and complete.exists():
            cached = json.loads(complete.read_text(encoding="utf-8"))
            current_hashes = cache_hashes(run_dir)
            if cached["input_sha256"] != current_hashes:
                raise ValueError(f"Input hashes changed for {run_dir.name}; choose a new output directory")
            results[run_dir.name] = cached
            print(f"Reusing completed {run_dir.name}", flush=True)
        else:
            results[run_dir.name] = run_model(run_dir, model_output, items_by_id, args)
        dump_json(output / "summary.json", {"protocol": protocol, "models": results})
    print(f"Saved {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
