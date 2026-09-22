"""检验 J x condition 交互对模型设定的敏感性。

12_factorial_stats 用的是"共享 level/form 效应 + J x cond 交互"。
但因子表显示 mandatory 大幅改变 level 效应本身（L2/L3 从 ~0.05 抬到 ~0.5），
共享设定下这部分失配会被 J x cond 吸收，可能虚增斜率差异。

三种设定对比：
  A 共享 level/form + J x cond          （原设定）
  B 允许 cond x level 交互 + J x cond   （放松共享）
  C 每个条件完全独立拟合                 （无共享假设，最保守）
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl
from sklearn.linear_model import LogisticRegression

def fit(X, y, C=1e6):
    m = LogisticRegression(C=C, max_iter=8000, solver="lbfgs").fit(X, y)
    return m, np.concatenate([[m.intercept_[0]], m.coef_[0]])

def boot_ci(build, rows, seeds, y, n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx_by = {s: [k for k, r in enumerate(rows) if r["seed"] == s] for s in seeds}
    B = []
    for _ in range(n):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        idx = np.concatenate([idx_by[seeds[p]] for p in pick])
        if len(set(y[idx].tolist())) < 2: continue
        try: B.append(fit(build(idx), y[idx])[1])
        except Exception: pass
    return np.array(B)

items = {i.item_id: i for i in load_items("data/pairs/pilot_v2.jsonl")}
cog = {r["item_id"]: r for r in read_jsonl("runs/llama-8b-v2/cognition.jsonl")}
ids = [i for i in items if i in cog]
Pv = np.clip(np.array([cog[i]["c_auth_per_variant"] for i in ids]), 1e-6, 1-1e-6)
lo_ = np.log(Pv/(1-Pv)); z = (lo_-lo_.mean(0))/np.maximum(lo_.std(0), 1e-8)
Jc = dict(zip(ids, np.median(z, axis=1)))

rows = []
for f in sorted(Path("runs/factorial-llama").glob("rollout_*.jsonl")):
    _, role, force = f.stem.split("_")
    for r in read_jsonl(f):
        it = items[r["item_id"]]
        if r["called_tool"] not in (None, it.action): continue
        rows.append({"cond": f"{role}-{force}", "seed": it.pair_id, "J": Jc[r["item_id"]],
                     "I": it.intent, "level": it.level, "form": it.form,
                     "P": int(r["executed"])})
y = np.array([r["P"] for r in rows]); seeds = sorted({r["seed"] for r in rows})
conds = sorted({r["cond"] for r in rows}); base = conds[0]
lvs = sorted({r["level"] for r in rows})[1:]; fms = sorted({r["form"] for r in rows})[1:]

def cols_A(idx):
    rs = [rows[i] for i in idx]
    return np.column_stack([
        [r["J"] for r in rs], [r["I"] for r in rs],
        *[[1.0*(r["level"]==L) for r in rs] for L in lvs],
        *[[1.0*(r["form"]==F) for r in rs] for F in fms],
        *[[1.0*(r["cond"]==c) for r in rs] for c in conds[1:]],
        *[[r["J"]*(r["cond"]==c) for r in rs] for c in conds[1:]]])

def cols_B(idx):
    rs = [rows[i] for i in idx]
    return np.column_stack([
        [r["J"] for r in rs], [r["I"] for r in rs],
        *[[1.0*(r["level"]==L) for r in rs] for L in lvs],
        *[[1.0*(r["form"]==F) for r in rs] for F in fms],
        *[[1.0*(r["cond"]==c) for r in rs] for c in conds[1:]],
        *[[1.0*(r["cond"]==c and r["level"]==L) for r in rs]      # cond x level
          for c in conds[1:] for L in lvs],
        *[[r["J"]*(r["cond"]==c) for r in rs] for c in conds[1:]]])

allidx = np.arange(len(rows))
n_int = len(conds) - 1
for name, build in (("A 共享 level/form", cols_A), ("B 允许 cond×level", cols_B)):
    _, pt = fit(build(allidx), y)
    B = boot_ci(build, rows, seeds, y)
    jslope = pt[1]
    print(f"\n=== 设定 {name} ===")
    print(f"  基线 {base:20s} beta_J = {jslope:6.3f}")
    for k, c in enumerate(conds[1:]):
        d = pt[-n_int + k]; bd = B[:, -n_int + k]
        p = 2*min((bd <= 0).mean(), (bd >= 0).mean())
        print(f"  {c:20s} delta = {d:+6.3f} [{np.percentile(bd,2.5):+6.3f}, "
              f"{np.percentile(bd,97.5):+6.3f}] p={p:.3f}   => beta_J = {jslope+d:6.3f}")

print("\n=== 设定 C 每条件独立拟合（见 runs/factorial-llama/slope_ci.json）===")
sc = json.load(open("runs/factorial-llama/slope_ci.json"))
for c in conds:
    r = sc[c]; print(f"  {c:20s} beta_J = {r['slope']:6.3f} [{r['lo']:6.3f}, {r['hi']:6.3f}]")
