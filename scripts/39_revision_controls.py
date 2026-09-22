"""Controls requested by the 2026-09-23 checklist review (CPU only, existing caches).

1. Protocol A text baselines: character TF-IDF and frozen MiniLM trained on the source-form
   user turns of the outer-training seeds and tested on the target-form user turns of the
   outer-test seeds, with the exact outer folds of scripts/24_nested_transfer.py.
2. Protocol A shuffled-label control: for every model, direction, and outer fold, the audit
   probe is refit at the archived selected layer with training labels permuted (20 permutations),
   then frozen and scored on the target-form action states, as in the real protocol.
3. Audit judgment J as a zero-shot classifier on the 23 construction folds of protocol B
   (no fitting; per-fold AUROC of the median z-scored log-odds against the label).
4. Steering shift size: class gap g and |alpha| g relative to the mean L2 norm of the
   last-position action state at the hooked layer.

A self-check first reproduces every archived outer-fold action AUROC from the stored fitted
probes on the reconstructed folds; the script stops if any fold disagrees.
"""
from __future__ import annotations

import importlib.util
import json
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

spec = importlib.util.spec_from_file_location("nested24", ROOT / "scripts/24_nested_transfer.py")
n24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(n24)

MODELS = ["qwen-7b-v2", "mistral-7b-v2", "hermes-8b-v2", "llama-8b-neutral-v2"]
NESTED = ROOT / "runs/revision-20260910/nested_transfer"
PAIRED = ROOT / "runs/revision-20260910/paired_holdout"
OUT = ROOT / "runs/revision-20260923/controls.json"
N_PERM = 20


def auc(y, s):
    return float(roc_auc_score(y, s))


def main() -> int:
    summary = json.loads((NESTED / "summary.json").read_text())
    seed = summary["protocol"]["rng_seed"]
    items = {it.item_id: it for it in load_items(ROOT / "data/pairs/pilot_v2.jsonl")}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "n_permutations": N_PERM,
           "self_check": {}, "text_protocol_a": {}, "shuffled_label_protocol_a": {}, "j_zero_shot_protocol_b": {},
           "steering_shift_size": {}}

    # MiniLM user-turn embeddings (archived, with item IDs).
    emb = np.load(PAIRED / "embedding_cache/minilm_embeddings.npz", allow_pickle=True)
    emb_by_id = {str(i): v for i, v in zip(emb["item_ids"], emb["embeddings"])}

    for model in MODELS:
        with np.load(ROOT / f"runs/{model}/forward.npz", allow_pickle=True) as fwd:
            ids = np.asarray(fwd["item_ids"], dtype=str); action = fwd["H"]
        audit = np.load(ROOT / f"runs/{model}/audit_hidden.npy", mmap_mode="r")
        y = np.array([items[i].intent for i in ids]); forms = np.array([items[i].form for i in ids])
        groups = np.array([items[i].pair_id for i in ids]); text = np.array([items[i].user_turn for i in ids])
        E = np.stack([emb_by_id[i] for i in ids])
        outer = n24.seed_folds(groups, 5, seed)
        res = summary["models"][model]["results"]
        out["self_check"][model] = {}; out["shuffled_label_protocol_a"][model] = {}
        for direction in ["imp2int", "int2imp"]:
            src, tgt = direction[:3], direction[4:]
            # --- self-check against archived fitted probes
            mism = 0
            for k, (tr, te) in enumerate(outer):
                z = np.load(NESTED / model / f"{direction}_outer{k}_fitted_probe.npz", allow_pickle=True)
                layer = int(z["layer"]); test = te[forms[te] == tgt]
                s = ((action[test, layer].astype(np.float64) - z["scaler_mean"]) / z["scaler_scale"]) @ z["coef"][0] + z["intercept"][0]
                if abs(auc(y[test], s) - res[direction]["folds"][k]["action_test_auroc"]) > 1e-9:
                    mism += 1
            out["self_check"][model][direction] = {"folds_reproduced": 5 - mism}
            if mism:
                raise SystemExit(f"fold reconstruction mismatch: {model} {direction}")
            # --- text baselines (identical across models; computed once per direction below)
            if model == MODELS[0]:
                tf, ml = [], []
                for tr, te in outer:
                    train = tr[forms[tr] == src]; test = te[forms[te] == tgt]
                    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
                    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
                    clf.fit(vec.fit_transform(text[train]), y[train])
                    tf.append(auc(y[test], clf.predict_proba(vec.transform(text[test]))[:, 1]))
                    sc = StandardScaler().fit(E[train])
                    clf2 = LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000, random_state=0)
                    clf2.fit(sc.transform(E[train]), y[train])
                    ml.append(auc(y[test], clf2.decision_function(sc.transform(E[test]))))
                out["text_protocol_a"][direction] = {"tfidf_mean_fold_auroc": float(np.mean(tf)), "tfidf_folds": tf,
                                                    "minilm_mean_fold_auroc": float(np.mean(ml)), "minilm_folds": ml}
            # --- shuffled-label probe at archived selected layers
            rng = np.random.default_rng(20260923)
            means = []
            for p in range(N_PERM):
                fold_aucs = []
                for k, (tr, te) in enumerate(outer):
                    layer = res[direction]["folds"][k]["selected_layer"]
                    train = tr[forms[tr] == src]; test = te[forms[te] == tgt]
                    yp = rng.permutation(y[train])
                    sc = StandardScaler().fit(np.asarray(audit[train, layer], dtype=np.float32))
                    clf = LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000, random_state=0, tol=1e-4)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        clf.fit(sc.transform(np.asarray(audit[train, layer], dtype=np.float32)), yp)
                    fold_aucs.append(auc(y[test], clf.decision_function(sc.transform(action[test, layer].astype(np.float32)))))
                means.append(float(np.mean(fold_aucs)))
            out["shuffled_label_protocol_a"][model][direction] = {
                "mean": float(np.mean(means)), "p95": float(np.percentile(means, 95)), "max": float(np.max(means)),
                "real_mean_fold_auroc": res[direction]["primary_mean_fold_auroc"], "permutation_means": means}
        # --- J zero-shot on construction folds
        cog = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / f"runs/{model}/cognition.jsonl").read_text().splitlines() if l.strip()}
        P = np.clip(np.array([cog[i]["c_auth_per_variant"] for i in ids], dtype=float), 1e-6, 1 - 1e-6)
        R = np.log(P / (1 - P)); Z = (R - R.mean(0)) / R.std(0); J = np.median(Z, axis=1)
        splits = json.loads((PAIRED / "splits.json").read_text())
        pos = {i: k for k, i in enumerate(ids)}
        fold_aucs = []
        for f in splits["schemes"]["paired_construction"]:
            t = np.array([pos[i] for i in f["test_item_ids"]]); fold_aucs.append(auc(y[t], J[t]))
        out["j_zero_shot_protocol_b"][model] = {"macro_auroc": float(np.mean(fold_aucs)), "min_fold_auroc": float(np.min(fold_aucs)),
                                               "n_folds": len(fold_aucs), "folds": fold_aucs}
        # --- steering shift size
        dpath = ROOT / f"runs/intervention-20260911/{model}/directions.json"
        if dpath.exists():
            d = json.loads(dpath.read_text()); ci = d["selected_cache_index"]; g = d["gap_g"]
            norms = np.linalg.norm(action[:, ci].astype(np.float64), axis=1)
            out["steering_shift_size"][model] = {"cache_index": ci, "gap_g": g, "mean_action_state_norm": float(norms.mean()),
                                                "g_over_norm": g / float(norms.mean()), "two_g_over_norm": 2 * g / float(norms.mean())}
        print(model, "done", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"text_protocol_a": {k: {m: round(v[m], 4) for m in ("tfidf_mean_fold_auroc", "minilm_mean_fold_auroc")} for k, v in out["text_protocol_a"].items()},
                      "shuffled": {m: {d: (round(v["mean"], 3), round(v["p95"], 3)) for d, v in dd.items()} for m, dd in out["shuffled_label_protocol_a"].items()},
                      "J_B": {m: round(v["macro_auroc"], 4) for m, v in out["j_zero_shot_protocol_b"].items()},
                      "shift": {m: {k: round(v[k], 4) for k in ("gap_g", "mean_action_state_norm", "two_g_over_norm")} for m, v in out["steering_shift_size"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
