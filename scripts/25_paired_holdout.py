"""Construction holdout with comparable, within-fold AUROC.

The split rule is fixed before fitting: enumerate all unordered pairs of the
construction groups used by script 13 and retain every pair for which BOTH the
training complement and test union contain both labels. This includes pairs
with a mixed-label construction; no pair is selected based on model performance.
All representations share these splits. Test folds overlap, so the reported
macro and weighted means are descriptive; folds are not independent replicates.

Example (from repository root)::

    python scripts/25_paired_holdout.py --model-dir /models/all-MiniLM-L6-v2 \
        --c-values 0.1 1 10

The original single-construction LOPO pooled AUROC is also recomputed as a
diagnostic, alongside the score consisting only of each fold's training prior.
Nothing from the test fold is used to fit a scaler, vectorizer, or classifier.
MiniLM is frozen and locally loaded at the original fixed revision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.schema import load_items  # noqa: E402

GROUP_SCRIPT = Path(__file__).with_name("13_group_holdout.py")
_spec = importlib.util.spec_from_file_location("original_holdout", GROUP_SCRIPT)
_holdout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_holdout)

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_RUNS = [
    "runs/qwen-7b-v2", "runs/mistral-7b-v2",
    "runs/hermes-8b-v2", "runs/llama-8b-neutral-v2",
]
SCHEMA_VERSION = "paired-construction-holdout-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def counts(y):
    return {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}


def make_splits(y, groups, item_ids):
    unique = sorted(set(groups.tolist()))
    schemes = {"paired_construction": [], "original_lopo_diagnostic": []}
    excluded = []

    def record(held_out, fold_id):
        test = np.flatnonzero(np.isin(groups, held_out))
        train = np.flatnonzero(~np.isin(groups, held_out))
        return {
            "fold_id": fold_id, "held_out_groups": list(held_out),
            "train_indices": train.tolist(), "test_indices": test.tolist(),
            "train_item_ids": [item_ids[i] for i in train],
            "test_item_ids": [item_ids[i] for i in test],
            "train_class_counts": counts(y[train]),
            "test_class_counts": counts(y[test]),
            "train_positive_prior": float(y[train].mean()),
            "n_train": len(train), "n_test": len(test),
            "test_positive_negative_pairs": int(np.sum(y[test] == 0) *
                                                np.sum(y[test] == 1)),
        }

    for first, second in itertools.combinations(unique, 2):
        fold = record([first, second], f"pair__{first}__{second}")
        if min(fold["train_class_counts"].values()) < 1:
            excluded.append({"held_out_groups": [first, second],
                             "reason": "single_class_train"})
        elif min(fold["test_class_counts"].values()) < 1:
            excluded.append({"held_out_groups": [first, second],
                             "reason": "single_class_test"})
        else:
            schemes["paired_construction"].append(fold)
    for group in unique:
        fold = record([group], f"lopo__{group}")
        if min(fold["train_class_counts"].values()) < 1:
            raise ValueError(f"LOPO diagnostic has single-class train: {group}")
        schemes["original_lopo_diagnostic"].append(fold)
    if not schemes["paired_construction"]:
        raise ValueError("No eligible two-construction test folds")
    return schemes, excluded


def aggregate_fold_metrics(rows):
    valid = [row for row in rows if row["auroc"] is not None]
    if not valid:
        return {"n_folds": len(rows), "n_dual_class_test_folds": 0}
    aucs = [row["auroc"] for row in valid]
    return {
        "n_folds": len(rows), "n_dual_class_test_folds": len(valid),
        "macro_auroc": float(np.mean(aucs)),
        "test_sample_weighted_auroc": float(np.average(
            aucs, weights=[row["n_test"] for row in valid])),
        "positive_negative_pair_weighted_auroc": float(np.average(
            aucs, weights=[row["test_positive_negative_pairs"] for row in valid])),
        "min_fold_auroc": float(min(aucs)), "max_fold_auroc": float(max(aucs)),
        "total_test_appearances": sum(row["n_test"] for row in rows),
        "interpretation": "Descriptive means; overlapping folds are not independent.",
    }


def fit_predict(kind, X, text, y, train, test, c_value, seed):
    if kind == "tfidf":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        X_train = vectorizer.fit_transform(text[train])
        X_test = vectorizer.transform(text[test])
        # Match script 13's text baseline solver and score convention.
        classifier = LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000,
                                        random_state=seed)
    else:
        scaler = StandardScaler().fit(X[train])
        X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])
        classifier = LogisticRegression(C=c_value, solver="liblinear", dual=True,
                                        max_iter=5000, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(X_train, y[train])
    score = (classifier.predict_proba(X_test)[:, 1] if kind == "tfidf"
             else classifier.decision_function(X_test))
    if not np.isfinite(score).all():
        raise ValueError("Non-finite classifier scores")
    return score, {
        "convergence_warnings": [str(warning.message) for warning in caught
                                 if issubclass(warning.category, ConvergenceWarning)],
        "n_iter": classifier.n_iter_.tolist(),
    }


def evaluate(name, kind, X, text, y, item_ids, schemes, c_values, seed, score_stream):
    results = {}
    for c_value in ([None] if kind == "train_prior" else c_values):
        c_key = "no_fit" if c_value is None else f"C={c_value:g}"
        per_scheme = {}
        for scheme, folds in schemes.items():
            rows = []
            pooled = np.full(len(y), np.nan)
            for fold in folds:
                train, test = np.asarray(fold["train_indices"]), np.asarray(fold["test_indices"])
                if kind == "train_prior":
                    scores = np.full(len(test), fold["train_positive_prior"])
                    fitting = {"convergence_warnings": [], "n_iter": []}
                else:
                    scores, fitting = fit_predict(kind, X, text, y, train, test,
                                                  c_value, seed)
                auc = (float(roc_auc_score(y[test], scores))
                       if len(np.unique(y[test])) == 2 else None)
                row = {key: fold[key] for key in (
                    "fold_id", "held_out_groups", "n_train", "n_test",
                    "train_class_counts", "test_class_counts", "train_positive_prior",
                    "test_positive_negative_pairs")}
                row.update({"auroc": auc, **fitting})
                rows.append(row)
                if scheme == "original_lopo_diagnostic":
                    if np.isfinite(pooled[test]).any():
                        raise ValueError("LOPO diagnostic assigns an item more than once")
                    pooled[test] = scores
                for item_index, score in zip(test, scores):
                    score_stream.write(json.dumps({
                        "representation": name, "C": c_value, "scheme": scheme,
                        "fold_id": fold["fold_id"], "item_index": int(item_index),
                        "item_id": item_ids[item_index], "label": int(y[item_index]),
                        "score": float(score),
                    }, ensure_ascii=False, allow_nan=False) + "\n")
            summary = aggregate_fold_metrics(rows)
            if scheme == "original_lopo_diagnostic":
                if not np.isfinite(pooled).all():
                    raise ValueError("Incomplete LOPO scores")
                summary["pooled_oof_auroc_diagnostic_only"] = float(roc_auc_score(y, pooled))
                summary["pooled_warning"] = (
                    "Scores are from separate fitted models; most test folds are single-class. "
                    "Pooled AUROC is not a within-fold construction-generalization estimate.")
            per_scheme[scheme] = {"summary": summary, "folds": rows}
            print(f"{name} {c_key} {scheme}: {json.dumps(summary)}", flush=True)
        results[c_key] = per_scheme
    return results


def get_embeddings(model_dir, cache_dir, items, data_sha, args):
    cache_dir.mkdir(parents=True, exist_ok=True)
    revision_marker = model_dir / "revision.txt"
    if revision_marker.exists():
        actual_revision = revision_marker.read_text().strip()
    elif model_dir.name == MINILM_REVISION:
        actual_revision = model_dir.name
    else:
        raise ValueError("MiniLM model directory must have revision.txt or be the fixed-revision "
                         "Hugging Face snapshot directory")
    if actual_revision != MINILM_REVISION:
        raise ValueError(f"Unexpected MiniLM revision: {actual_revision}")
    ids = [item.item_id for item in items]
    identity = {
        "model": MINILM_MODEL, "revision": MINILM_REVISION,
        "data_sha256": data_sha,
        "ordered_item_ids_sha256": json_hash(ids),
        "ordered_text_sha256": json_hash([item.user_turn for item in items]),
        "pooling": "attention-mask mean then L2 normalization", "max_length": 256,
        "dtype": "float32", "device": args.device,
    }
    cache_npz, cache_json = cache_dir / "minilm_embeddings.npz", cache_dir / "metadata.json"
    if cache_npz.exists() and cache_json.exists():
        metadata = json.loads(cache_json.read_text())
        if (metadata.get("identity") == identity and
                metadata.get("npz_sha256") == sha256_file(cache_npz)):
            with np.load(cache_npz, allow_pickle=False) as saved:
                if saved["item_ids"].tolist() != ids:
                    raise ValueError("MiniLM embedding cache item ordering is wrong")
                X = saved["embeddings"]
            if X.shape != (len(items), 384) or not np.isfinite(X).all():
                raise ValueError("Invalid cached MiniLM embeddings")
            return X, metadata
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().to(args.device)
    if model.config.hidden_size != 384 or model.config.num_hidden_layers != 6:
        raise ValueError("MiniLM architecture mismatch")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            encoded = tokenizer([item.user_turn for item in items[start:start + args.batch_size]],
                                padding=True, truncation=True, max_length=256,
                                return_tensors="pt").to(args.device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    X = np.concatenate(batches).astype(np.float32)
    if X.shape != (len(items), 384) or not np.isfinite(X).all():
        raise ValueError("Invalid MiniLM embeddings")
    np.savez_compressed(cache_npz, embeddings=X, item_ids=np.asarray(ids))
    metadata = {"identity": identity, "npz_sha256": sha256_file(cache_npz),
                "shape": list(X.shape), "model_files_sha256": {}}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(model_dir).parts):
            metadata["model_files_sha256"][str(path.relative_to(model_dir))] = sha256_file(path)
    write_json(cache_json, metadata)
    return X, metadata


def write_schema(path):
    write_json(path, {
        "schema_version": SCHEMA_VERSION,
        "files": {
            "splits.json": "One canonical item order, labels, groups, deterministic train/test indices and IDs; SHA256 of schemes.",
            "scores.jsonl": {"one_record": {
                "representation": "string (train_prior, tfidf, minilm, or probe:<run name>)",
                "C": "positive number; null for training-prior control", "scheme": "string",
                "fold_id": "string", "item_index": "integer index into canonical items",
                "item_id": "string", "label": "0 or 1", "score": "finite float",
            }},
            "metrics.json": "results[representation][C key][scheme] contains summary and folds; AUROC is null for single-class folds.",
            "manifest.json": "Input and output SHA256, full parameters, package versions, timing and completion status.",
            "embedding_cache/minilm_embeddings.npz": "embeddings: float32[n,384]; item_ids: unicode[n].",
            "embedding_cache/metadata.json": "Fixed model revision, input identity, encoder configuration and model-file checksums.",
        },
        "primary_metric": "Per-fold AUROC for all eligible paired-construction folds, then macro mean.",
        "weighted_metrics": {
            "test_sample_weighted_auroc": "sum(n_test * fold AUROC) / sum(n_test)",
            "positive_negative_pair_weighted_auroc": "sum(n_pos*n_neg * fold AUROC) / sum(n_pos*n_neg)",
        },
        "limitations": [
            "Overlapping held-out pairs and repeated semantic seeds mean folds are dependent; no IID-fold CI is reported.",
            "The construction split holds out lexical construction groups, not semantic seeds.",
            "No pooled score AUROC is used as a primary paired-fold result.",
            "C sensitivity is reported in full; no test-selected C or layer is reported as a confirmatory optimum.",
        ],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", type=Path, default=Path("data/pairs/pilot_v2.jsonl"))
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_RUNS])
    parser.add_argument("--model-dir", type=Path, help="Local MiniLM directory at the fixed original revision")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/revision-20260910/paired_holdout"))
    parser.add_argument("--embedding-cache", type=Path, help="Defaults to OUT_DIR/embedding_cache")
    parser.add_argument("--c-values", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits-only", action="store_true", help="Validate deterministic splits without any model fits")
    args = parser.parse_args()
    if any(not np.isfinite(c) or c <= 0 for c in args.c_values):
        parser.error("C values must be finite and positive")
    if args.threads < 1 or args.batch_size < 1:
        parser.error("threads and batch-size must be positive")
    if not args.splits_only and args.model_dir is None:
        parser.error("--model-dir is required unless --splits-only is used")
    args.c_values = sorted(set(args.c_values))
    if 1.0 not in args.c_values:
        parser.error("--c-values must include C=1, the prespecified primary classifier")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.threads)
    np.random.seed(args.seed)

    items = load_items(args.items)
    item_ids = [item.item_id for item in items]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Duplicate data item IDs")
    y = np.asarray([item.intent for item in items], dtype=int)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Expected binary labels")
    text = np.asarray([item.user_turn for item in items])
    groups = np.asarray([_holdout.verb_group(item.user_turn) for item in items])
    schemes, excluded = make_splits(y, groups, item_ids)
    split_sha = json_hash(schemes)
    splits = {
        "schema_version": SCHEMA_VERSION, "split_sha256": split_sha,
        "rule": "All unordered pairs of script-13 construction groups, retaining every pair with both labels in train and test.",
        "n_candidate_pairs": len(list(itertools.combinations(sorted(set(groups.tolist())), 2))),
        "excluded_pairs": excluded, "item_ids": item_ids, "labels": y.tolist(),
        "groups": groups.tolist(), "class_counts": counts(y),
        "group_class_counts": {g: counts(y[groups == g]) for g in sorted(set(groups.tolist()))},
        "schemes": schemes,
    }
    write_json(args.out_dir / "splits.json", splits)
    write_schema(args.out_dir / "schema.json")
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "splits_only" if args.splits_only else "running",
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "parameters": {key: ([str(x) for x in value] if key == "run_dirs" else
                             str(value) if isinstance(value, Path) else value)
                       for key, value in vars(args).items()},
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {}, "data_sha256": sha256_file(args.items),
        "script_sha256": sha256_file(__file__), "grouping_script_sha256": sha256_file(GROUP_SCRIPT),
        "split_sha256": split_sha, "inputs": {}, "outputs": {},
    }
    for package in ["numpy", "scikit-learn", "scipy", "torch", "transformers", "threadpoolctl"]:
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["versions"][package] = None
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps({"groups": splits["group_class_counts"],
                      "paired_folds": len(schemes["paired_construction"]),
                      "excluded_pairs": len(excluded), "split_sha256": split_sha}), flush=True)
    if args.splits_only:
        return 0

    names = [path.name for path in args.run_dirs]
    if len(set(names)) != len(names):
        raise ValueError("Run directory names must be unique")
    # Require every requested model to cover exactly the same data; align by ID.
    for run_dir in args.run_dirs:
        forward = run_dir / "forward.npz"
        with np.load(forward, allow_pickle=True) as saved:
            forward_ids = [str(x) for x in saved["item_ids"].tolist()]
        if len(forward_ids) != len(item_ids) or set(forward_ids) != set(item_ids):
            raise ValueError(f"{run_dir}: item coverage differs from canonical data")
        manifest["inputs"][str(forward)] = sha256_file(forward)
    embedding_cache = args.embedding_cache or args.out_dir / "embedding_cache"
    embeddings, embedding_metadata = get_embeddings(args.model_dir, embedding_cache, items,
                                                    manifest["data_sha256"], args)
    manifest["minilm"] = embedding_metadata
    write_json(args.out_dir / "manifest.json", manifest)
    metrics = {
        "schema_version": SCHEMA_VERSION, "split_sha256": split_sha,
        "primary_C": 1.0, "fixed_probe_relative_layer": 0.55,
        "fixed_probe_layer_rule": "int(round(0.55 * (H.shape[1] - 1)))",
        "classifiers": {
            "probe_and_minilm": "train-only StandardScaler + LogisticRegression(liblinear, dual=True, max_iter=5000)",
            "tfidf": "train-only char_wb TFIDF(2,4), min_df=1 + LogisticRegression(lbfgs, max_iter=2000)",
        },
        "score_types": {"tfidf": "positive-label probability", "probe_and_minilm": "decision function",
                        "train_prior": "training positive-label proportion"},
        "models": {}, "results": {}, "original_published_diagnostics": {},
    }
    for filename, key in [("runs/group_holdout.json", "script13"),
                          ("runs/semantic_baseline.json", "script22")]:
        path = ROOT / filename
        if path.exists():
            old = json.loads(path.read_text())
            metrics["original_published_diagnostics"][key] = {
                "path": filename, "sha256": sha256_file(path),
                "pooled_verb_auroc": ({name: {k: record[k] for k in ["probe_verb", "tfidf_verb"]}
                                       for name, record in old.items()} if key == "script13" else
                                      old["results"]["verb"]["auroc"]),
            }
    score_path = args.out_dir / "scores.jsonl"
    with score_path.open("w", encoding="utf-8") as score_stream:
        for name, kind, X in [("train_prior", "train_prior", None),
                              ("tfidf", "tfidf", None), ("minilm", "minilm", embeddings)]:
            metrics["results"][name] = evaluate(name, kind, X, text, y, item_ids, schemes,
                                                 args.c_values, args.seed, score_stream)
            score_stream.flush()
            write_json(args.out_dir / "metrics.json", metrics)
        for run_dir in args.run_dirs:
            with np.load(run_dir / "forward.npz", allow_pickle=True) as saved:
                index = {str(item_id): i for i, item_id in enumerate(saved["item_ids"].tolist())}
                H = saved["H"]
                layer = int(round(0.55 * (H.shape[1] - 1)))
                X = H[[index[item_id] for item_id in item_ids], layer, :].astype(np.float32)
                h_shape = list(H.shape)
                del H
            if not np.isfinite(X).all():
                raise ValueError(f"Non-finite hidden states in {run_dir}")
            name = f"probe:{run_dir.name}"
            metrics["models"][name] = {"layer": layer, "H_shape": h_shape,
                                      "forward_sha256": manifest["inputs"][str(run_dir / "forward.npz")]}
            metrics["results"][name] = evaluate(name, "probe", X, text, y, item_ids, schemes,
                                                 args.c_values, args.seed, score_stream)
            score_stream.flush()
            write_json(args.out_dir / "metrics.json", metrics)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    for name in ["schema.json", "splits.json", "scores.jsonl", "metrics.json"]:
        manifest["outputs"][name] = sha256_file(args.out_dir / name)
    write_json(args.out_dir / "manifest.json", manifest)
    print(f"Complete: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
