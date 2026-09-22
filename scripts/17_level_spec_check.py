"""C3 的设定检验：L4 ≈ L0 是否也是共享系数造成的假象。

若 level 效应随模板条件剧变，池化模型给出的 L4 系数只是六个条件的平均，
"L4 与 L0 无差异"可能在任何单个条件下都不成立。
逐条件单独估计 L4 vs L0 的对数几率比，看方向是否一致。
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl
from sklearn.linear_model import LogisticRegression

def fit(X, y):
    m = LogisticRegression(C=1e6, max_iter=8000, solver="lbfgs").fit(X, y)
    return np.concatenate([[m.intercept_[0]], m.coef_[0]])

items = {i.item_id: i for i in load_items("data/pairs/pilot_v2.jsonl")}
cog = {r["item_id"]: r for r in read_jsonl("runs/llama-8b-v2/cognition.jsonl")}
ids = [i for i in items if i in cog]
Pv = np.clip(np.array([cog[i]["c_auth_per_variant"] for i in ids]), 1e-6, 1-1e-6)
lo_ = np.log(Pv/(1-Pv)); z = (lo_-lo_.mean(0))/np.maximum(lo_.std(0), 1e-8)
Jc = dict(zip(ids, np.median(z, axis=1)))

rng = np.random.default_rng(0)
print(f"{'condition':22s} {'L4 vs L0 (OR)':>16s} {'95% CI':>20s}   {'原始执行率 L0/L4':>18s}")
print("-" * 82)
for f in sorted(Path("runs/factorial-llama").glob("rollout_*.jsonl")):
    _, role, force = f.stem.split("_")
    rs = []
    for r in read_jsonl(f):
        it = items[r["item_id"]]
        if r["called_tool"] not in (None, it.action): continue
        rs.append((it.pair_id, Jc[r["item_id"]], it.level, it.form, int(r["executed"])))
    # 只比 L0 与 L4（同为 intent 明确的两端：一个是直接命令，一个是元语言提及）
    sub = [r for r in rs if r[2] in ("L0", "L4")]
    y = np.array([r[4] for r in sub])
    fms = sorted({r[3] for r in sub})[1:]
    def design(rows):
        return np.column_stack([[r[1] for r in rows],
                                [1.0*(r[2] == "L4") for r in rows],
                                *[[1.0*(r[3] == F) for r in rows] for F in fms]])
    X = design(sub); pt = fit(X, y)[2]
    seeds = sorted({r[0] for r in sub})
    idx_by = {s: [k for k, r in enumerate(sub) if r[0] == s] for s in seeds}
    B = []
    for _ in range(800):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        idx = np.concatenate([idx_by[seeds[p]] for p in pick])
        if len(set(y[idx].tolist())) < 2: continue
        try: B.append(fit(X[idx], y[idx])[2])
        except Exception: pass
    B = np.array(B); l, h = np.percentile(B, [2.5, 97.5])
    e0 = np.mean([r[4] for r in rs if r[2] == "L0"]); e4 = np.mean([r[4] for r in rs if r[2] == "L4"])
    print(f"{role+'-'+force:22s} {np.exp(pt):16.2f} [{np.exp(l):8.2f},{np.exp(h):8.2f}]"
          f"   {e0:8.2f} / {e4:.2f}")
print("\n判读：OR 接近 1 = L4 与直接命令 L0 的执行几率无差异。")
print("      若六个条件方向一致，C3 不依赖池化设定。")
