"""Compute steering directions from audit hidden states only (protocol in compute/intervention-20260911/PROTOCOL.md).

Layer: highest seed-grouped 5-fold audit->audit AUROC of the difference-of-means projection (ties -> shallower).
Directions at that layer: d_auth (class mean difference, unit), d_form (interrogative minus non-interrogative,
orthogonalised to d_auth, unit), d_rand0/d_rand1 (seeded Gaussian, unit). Gap g = mean projection gap on d_auth.
No action outcomes are read.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.schema import load_items  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True, help="audit_hidden_ids.npz with item_ids")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--items", type=Path, default=ROOT / "data/pairs/pilot_v2.jsonl")
    args = ap.parse_args()
    items = {it.item_id: it for it in load_items(args.items)}
    z = np.load(args.cache, allow_pickle=True)
    ids = [str(x) for x in z["item_ids"]]
    H = z["H"].astype(np.float32)  # (n, L, d)
    y = np.asarray([items[i].intent for i in ids]); form = np.asarray([items[i].form == "int" for i in ids]).astype(int)
    groups = np.asarray([items[i].pair_id for i in ids])
    n, L, d = H.shape
    aucs = []
    gkf = GroupKFold(n_splits=5)
    for l in range(L):
        pred = np.zeros(n)
        for tr, te in gkf.split(H[:, l], y, groups):
            dm = H[tr, l][y[tr] == 1].mean(0) - H[tr, l][y[tr] == 0].mean(0)
            pred[te] = H[te, l] @ dm
        aucs.append(float(roc_auc_score(y, pred)))
    aucs_sel = aucs[1:]  # exclude embedding layer
    layer = 1 + int(np.argmax(np.round(aucs_sel, 12)))  # argmax returns first (shallowest) among ties
    X = H[:, layer]
    d_auth = X[y == 1].mean(0) - X[y == 0].mean(0); d_auth = d_auth / np.linalg.norm(d_auth)
    proj = X @ d_auth
    gap = float(proj[y == 1].mean() - proj[y == 0].mean())
    d_form = X[form == 1].mean(0) - X[form == 0].mean(0); d_form -= (d_form @ d_auth) * d_auth; d_form /= np.linalg.norm(d_form)
    rng0, rng1 = np.random.default_rng(20260911), np.random.default_rng(20260912)
    d_r0 = rng0.standard_normal(d).astype(np.float32); d_r0 /= np.linalg.norm(d_r0)
    d_r1 = rng1.standard_normal(d).astype(np.float32); d_r1 /= np.linalg.norm(d_r1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "directions.npz", d_auth=d_auth, d_form=d_form, d_rand0=d_r0, d_rand1=d_r1,
             layer=np.int64(layer), gap=np.float32(gap))
    meta = {"cache": str(args.cache), "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
            "n_items": n, "n_cached_states": L, "hidden_dim": d, "selected_cache_index": layer,
            "hook_decoder_layer_index": layer - 1, "layer_rule": "argmax of seed-grouped 5-fold audit->audit AUROC of difference-of-means projection over cache indices 1..L-1; first (shallowest) on ties",
            "audit_auc_by_cache_index": aucs, "gap_g": gap,
            "projection_std_within_class": {"authorized": float(proj[y == 1].std()), "unauthorized": float(proj[y == 0].std())},
            "cos_auth_form_before_orthogonalisation": float(((X[form == 1].mean(0) - X[form == 0].mean(0)) / np.linalg.norm(X[form == 1].mean(0) - X[form == 0].mean(0))) @ d_auth),
            "cos_auth_rand": [float(d_r0 @ d_auth), float(d_r1 @ d_auth)],
            "form_auc_on_auth_direction": float(roc_auc_score(form, proj)),
            "alphas": [-2, -1, 1, 2], "shift_norm_per_alpha_unit": gap,
            "created_utc": datetime.now(timezone.utc).isoformat()}
    (args.out_dir / "directions.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ["selected_cache_index", "hook_decoder_layer_index", "gap_g", "projection_std_within_class", "cos_auth_form_before_orthogonalisation", "form_auc_on_auth_direction"]}, indent=1))
    print("audit AUROC by cache index:", [round(a, 3) for a in aucs])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
