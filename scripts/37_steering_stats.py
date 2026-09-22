"""Statistics for the activation-steering rollouts (protocol: compute/intervention-20260911/PROTOCOL.md).

Per condition: FAR, MAR (legacy parser, fixed 240/160 denominators), truncation and degenerate counts.
Paired seed-cluster bootstrap (2,000 draws, seed 0) of each steered condition minus the same-run baseline,
and of |delta_auth| - |delta_control| at equal |alpha| for each control direction. Evaluates the three
pre-registered criteria (monotone direction, exceeds controls with CI excluding zero, degenerate <= 5% at |alpha|=1).
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.schema import load_items  # noqa: E402


def load(steer: Path, items):
    prov = json.loads((steer / "provenance.json").read_text())
    conds = {}
    for c in prov["conditions"]:
        tag = f"{c['direction']}_a{c['alpha']:+.0f}" if c["direction"] != "none" else "baseline_a0"
        rows = {json.loads(l)["item_id"]: json.loads(l) for l in (steer / f"rollout_{tag}.jsonl").read_text().splitlines() if l.strip()}
        assert set(rows) == {it.item_id for it in items}, tag
        conds[tag] = {**c, "tag": tag, "rows": rows}
    return prov, conds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steer-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = load_items(ROOT / "data/pairs/pilot_v2.jsonl")
    prov, conds = load(args.steer_dir, items)
    ids = [it.item_id for it in items]; I = np.array([it.intent for it in items]); seeds = np.array([it.pair_id for it in items])
    useeds = sorted(set(seeds)); seed_idx = {s: np.flatnonzero(seeds == s) for s in useeds}
    Y = {t: np.array([c["rows"][i]["executed"] for i in ids], dtype=float) for t, c in conds.items()}
    T = {t: np.array([c["rows"][i]["truncated"] for i in ids]) for t, c in conds.items()}
    D = {t: np.array([c["rows"][i]["degenerate"] for i in ids]) for t, c in conds.items()}

    def rates(y, w=None):
        w = np.ones(len(y)) if w is None else w
        neg, pos = (I == 0), (I == 1)
        return (np.sum(w * y * neg) / np.sum(w * neg), np.sum(w * (1 - y) * pos) / np.sum(w * pos))
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(useeds), size=(args.n_boot, len(useeds)))
    weights = np.zeros((args.n_boot, len(ids)))
    for b in range(args.n_boot):
        mult = np.bincount(draws[b], minlength=len(useeds))
        for k, s in enumerate(useeds):
            weights[b, seed_idx[s]] = mult[k]
    base = "baseline_a0"
    def boot_rates(t):
        return np.array([rates(Y[t], weights[b]) for b in range(args.n_boot)])  # (B, 2)
    BR = {t: boot_rates(t) for t in conds}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "steer_dir": str(args.steer_dir.resolve().relative_to(ROOT)),
           "model": prov["model_key"], "hook_decoder_layer_index": prov["hook_decoder_layer_index"], "gap_g": prov["gap_g"],
           "max_new_tokens": prov["max_new_tokens"], "n_boot": args.n_boot, "conditions": {}, "contrasts_vs_baseline": {},
           "auth_minus_control_abs": {}, "criteria": {}}
    for t, c in conds.items():
        far, mar = rates(Y[t])
        out["conditions"][t] = {"direction": c["direction"], "alpha": c["alpha"], "FAR": far, "MAR": mar,
                                "FAR_ci95": list(np.percentile(BR[t][:, 0], [2.5, 97.5])), "MAR_ci95": list(np.percentile(BR[t][:, 1], [2.5, 97.5])),
                                "n_truncated": int(T[t].sum()), "n_degenerate": int(D[t].sum()), "degenerate_fraction": float(D[t].mean())}
        if t != base:
            d = BR[t] - BR[base]
            out["contrasts_vs_baseline"][t] = {"dFAR_pp": 100 * (far - rates(Y[base])[0]), "dFAR_ci95_pp": list(100 * np.percentile(d[:, 0], [2.5, 97.5])),
                                              "dMAR_pp": 100 * (mar - rates(Y[base])[1]), "dMAR_ci95_pp": list(100 * np.percentile(d[:, 1], [2.5, 97.5]))}
    alphas = sorted({c["alpha"] for c in conds.values() if c["direction"] != "none"})
    for a in alphas:
        ta = f"d_auth_a{a:+.0f}"
        for ctrl in ["d_form", "d_rand0", "d_rand1"]:
            tc = f"{ctrl}_a{a:+.0f}"
            if ta not in BR or tc not in BR:
                continue
            da = np.abs(BR[ta] - BR[base]); dc = np.abs(BR[tc] - BR[base]); diff = da - dc
            pa = np.abs(np.array(rates(Y[ta])) - np.array(rates(Y[base]))); pc = np.abs(np.array(rates(Y[tc])) - np.array(rates(Y[base])))
            out["auth_minus_control_abs"][f"alpha{a:+.0f}_vs_{ctrl}"] = {
                "abs_dFAR_auth_minus_ctrl_pp": 100 * (pa[0] - pc[0]), "ci95_pp": list(100 * np.percentile(diff[:, 0], [2.5, 97.5])),
                "abs_dMAR_auth_minus_ctrl_pp": 100 * (pa[1] - pc[1]), "MAR_ci95_pp": list(100 * np.percentile(diff[:, 1], [2.5, 97.5]))}
    # criteria
    fars = {a: out["conditions"][f"d_auth_a{a:+.0f}"]["FAR"] for a in alphas if f"d_auth_a{a:+.0f}" in out["conditions"]}
    mars = {a: out["conditions"][f"d_auth_a{a:+.0f}"]["MAR"] for a in alphas if f"d_auth_a{a:+.0f}" in out["conditions"]}
    seq = sorted(fars); far_base, mar_base = rates(Y[base])
    full = [fars[a] if a in fars else far_base for a in seq]; fullm = [mars[a] if a in mars else mar_base for a in seq]
    neg_a = [a for a in seq if a < 0]; pos_a = [a for a in seq if a > 0]
    mono_far = all(fars[a] < far_base for a in neg_a) and all(fars[a] > far_base for a in pos_a) and all(np.diff(full) >= 0)
    mono_mar = all(mars[a] > mar_base for a in neg_a) and all(mars[a] < mar_base for a in pos_a) and all(np.diff(fullm) <= 0)
    exceeds = {k: bool(v["ci95_pp"][0] > 0) for k, v in out["auth_minus_control_abs"].items()}
    degen_ok = all(out["conditions"][f"d_auth_a{a:+.0f}"]["degenerate_fraction"] <= 0.05 for a in [-1.0, 1.0] if f"d_auth_a{a:+.0f}" in out["conditions"])
    out["criteria"] = {"a_monotone_FAR": bool(mono_far), "a_monotone_MAR": bool(mono_mar),
                       "b_auth_exceeds_controls_FAR_ci_excludes_zero": exceeds, "b_all": all(exceeds.values()) if exceeds else False,
                       "c_degenerate_at_alpha1_le_5pct": bool(degen_ok),
                       "causal_use_supported": bool(mono_far and mono_mar and exceeds and all(exceeds.values()) and degen_ok)}
    outp = args.out or (args.steer_dir / "stats.json")
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["criteria"], indent=1))
    for t, c in out["conditions"].items():
        print(f"{t:18s} FAR {100*c['FAR']:6.2f}  MAR {100*c['MAR']:6.2f}  trunc {c['n_truncated']:3d}  degen {c['n_degenerate']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
