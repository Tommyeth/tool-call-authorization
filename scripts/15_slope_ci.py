"""导出各模板条件的 J 斜率与 bootstrap CI，供 Fig.2 右图使用。

12_factorial_stats 只存了 p 值，没存区间。图上必须有区间，
否则"斜率被砍掉 2/3"这句话看不出不确定性。
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl
from sklearn.linear_model import LogisticRegression

def fit(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs").fit(X, y)
    return np.concatenate([[m.intercept_[0]], m.coef_[0]])

def main():
    items = {i.item_id: i for i in load_items("data/pairs/pilot_v2.jsonl")}
    cog = {r["item_id"]: r for r in read_jsonl("runs/llama-8b-v2/cognition.jsonl")}
    ids = [i for i in items if i in cog]
    P = np.clip(np.array([cog[i]["c_auth_per_variant"] for i in ids]), 1e-6, 1-1e-6)
    lo = np.log(P/(1-P)); z = (lo-lo.mean(0))/np.maximum(lo.std(0), 1e-8)
    Jc = dict(zip(ids, np.median(z, axis=1)))

    out = {}
    rng = np.random.default_rng(0)
    for f in sorted(Path("runs/factorial-llama").glob("rollout_*.jsonl")):
        _, role, force = f.stem.split("_")
        rows = []
        for r in read_jsonl(f):
            it = items[r["item_id"]]
            if r["called_tool"] not in (None, it.action):
                continue
            rows.append((it.pair_id, Jc[r["item_id"]], it.intent,
                         it.level, it.form, int(r["executed"])))
        seeds = sorted({r[0] for r in rows})
        lv = sorted({r[3] for r in rows}); fm = sorted({r[4] for r in rows})
        def design(rs):
            return np.column_stack([
                [r[1] for r in rs], [r[2] for r in rs],
                *[[1.0 if r[3]==L else 0.0 for r in rs] for L in lv[1:]],
                *[[1.0 if r[4]==F else 0.0 for r in rs] for F in fm[1:]]])
        X = design(rows); y = np.array([r[5] for r in rows])
        pt = fit(X, y)[1]                      # J 的系数
        idx_by = {s: [k for k, r in enumerate(rows) if r[0] == s] for s in seeds}
        B = []
        for _ in range(1000):
            pick = rng.choice(len(seeds), len(seeds), replace=True)
            idx = np.concatenate([idx_by[seeds[p]] for p in pick])
            if len(set(y[idx].tolist())) < 2: continue
            try: B.append(fit(X[idx], y[idx])[1])
            except Exception: pass
        B = np.array(B)
        out[f"{role}-{force}"] = {"slope": float(pt),
                                  "lo": float(np.percentile(B, 2.5)),
                                  "hi": float(np.percentile(B, 97.5)),
                                  "n_boot": len(B)}
        print(f"  {role:6s} {force:11s} beta_J={pt:6.3f}  "
              f"[{out[f'{role}-{force}']['lo']:6.3f}, {out[f'{role}-{force}']['hi']:6.3f}]")
    Path("runs/factorial-llama/slope_ci.json").write_text(json.dumps(out, indent=2))
    print("\n-> runs/factorial-llama/slope_ci.json")

if __name__ == "__main__":
    main()
