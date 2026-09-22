"""因子实验的正式统计：role × force 主效应/交互，以及 J×condition 斜率检验。

这是"模板是否改变授权敏感度"的**干净**检验：同一权重、同一 400 条数据、
同一份 J，只有 tool prompt 的 role 与 deontic force 变化。
跨模型的 J×config 交互（scripts/11）混淆了模型身份，这里没有这个问题。

关键对比：
  * force 显著改变截距（baseline call propensity）——预期成立
  * J 的斜率是否随 condition 变化？若不变，说明模板调的是"多爱行动"，
    而授权信息参与决策的程度不受影响；若变，说明模板还改变了
    授权信号进入 action policy 的通道强度。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_cj = import_module("11_continuous_J") if False else None  # 避免模块名以数字开头的导入问题

from sklearn.linear_model import LogisticRegression  # noqa: E402


def fit_logit(X, y, C=1e6):
    m = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    m.fit(X, y)
    return np.concatenate([[m.intercept_[0]], m.coef_[0]])


def design(rows, cols):
    X, names = [], []
    for c in cols:
        vals = [r[c] for r in rows]
        if c in ("J", "I") or (isinstance(vals[0], float) and len(set(vals)) > 2):
            X.append(np.array(vals, float)); names.append(c)
        else:
            for cat in sorted(set(vals))[1:]:
                X.append(np.array([1.0 if v == cat else 0.0 for v in vals]))
                names.append(f"{c}={cat}")
    return np.column_stack(X), names


def cluster_bootstrap(rows, cols, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    X, names = design(rows, cols)
    y = np.array([r["P"] for r in rows])
    point = fit_logit(X, y)
    seeds = sorted({r["seed"] for r in rows})
    idx_by = {s: [i for i, r in enumerate(rows) if r["seed"] == s] for s in seeds}
    B = []
    for _ in range(n_boot):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        idx = np.concatenate([idx_by[seeds[p]] for p in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        try:
            B.append(fit_logit(X[idx], y[idx]))
        except Exception:
            continue
    B = np.array(B)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    p = 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0))
    return ["(intercept)"] + names, point, lo, hi, p, len(B)


def show(title, names, pt, lo, hi, p, only=None):
    print(f"\n=== {title} ===")
    print(f"{'term':30s} {'coef':>8s} {'OR':>8s} {'95% CI (OR)':>22s} {'p':>8s}")
    for t, c, l, h, pv in zip(names, pt, lo, hi, p):
        if only and not any(t.startswith(o) for o in only):
            continue
        star = " *" if pv < 0.05 else ""
        print(f"{t:30s} {c:8.3f} {np.exp(c):8.2f} "
              f"[{np.exp(l):8.2f}, {np.exp(h):8.2f}] {pv:8.4f}{star}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-dir", default="runs/factorial-llama")
    ap.add_argument("--cognition", default="runs/llama-8b-v2/cognition.jsonl")
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    items = {i.item_id: i for i in load_items(args.items)}
    cog = {r["item_id"]: r for r in read_jsonl(args.cognition)}

    # 连续 J：三个 variant 的 log-odds 各自 z-score 后取中位数
    ids = [i for i in items if i in cog]
    Praw = np.clip(np.array([cog[i]["c_auth_per_variant"] for i in ids]), 1e-6, 1 - 1e-6)
    lo_ = np.log(Praw / (1 - Praw))
    z = (lo_ - lo_.mean(0)) / np.maximum(lo_.std(0), 1e-8)
    Jc = dict(zip(ids, np.median(z, axis=1)))

    rows = []
    for f in sorted(Path(args.fact_dir).glob("rollout_*.jsonl")):
        _, role, force = f.stem.split("_")
        for r in read_jsonl(f):
            it = items[r["item_id"]]
            if r["called_tool"] not in (None, it.action):
                continue
            rows.append({
                "role": role, "force": force, "cond": f"{role}-{force}",
                "seed": it.pair_id, "level": it.level, "form": it.form,
                "I": it.intent, "J": float(Jc[r["item_id"]]), "P": int(r["executed"]),
            })
    print(f"n={len(rows)} 行, {len({r['seed'] for r in rows})} seed, "
          f"{len({r['cond'] for r in rows})} 个模板条件")

    names, pt, lo, hi, p, nb = cluster_bootstrap(
        rows, ["J", "I", "level", "form", "role", "force"], args.n_boot)
    show(f"主效应模型 (cluster bootstrap over seeds, B={nb})", names, pt, lo, hi, p)

    # role × force 交互
    rows2 = [dict(r, rolexforce=f"{r['role']}:{r['force']}") for r in rows]
    n2, p2_, l2, h2, pv2, _ = cluster_bootstrap(
        rows2, ["J", "I", "level", "form", "rolexforce"], args.n_boot)
    show("role × force 交互（各条件相对基线）", n2, p2_, l2, h2, pv2, only=["rolexforce"])

    # 核心：J × condition —— 同权重同数据，模板是否改变授权敏感度斜率
    conds = sorted({r["cond"] for r in rows})
    rows3 = [dict(r) for r in rows]
    for r in rows3:
        for c in conds[1:]:
            r[f"Jx_{c}"] = r["J"] if r["cond"] == c else 0.0
    cols = ["J", "I", "level", "form", "cond"] + [f"Jx_{c}" for c in conds[1:]]
    n3, p3, l3, h3, pv3, _ = cluster_bootstrap(rows3, cols, args.n_boot)
    show(f"J × condition 斜率检验（基线 = {conds[0]}）", n3, p3, l3, h3, pv3,
         only=["J"])

    sig = [t for t, pv in zip(n3, pv3) if t.startswith("Jx_") and pv < 0.05]
    print("\n[干净判读] 同一权重、同一数据、同一份 J，仅模板变化：")
    if sig:
        print(f"  斜率显著随模板变化: {', '.join(sig)}")
        print("  => 模板不只调节'多爱行动'，还改变授权信息参与决策的程度。")
    else:
        print("  所有 J×condition 交互均不显著。")
        print("  => 模板主要移动 baseline call propensity（截距），")
        print("     授权敏感度斜率不受影响。这正是跨模型分析无法干净回答的那个问题。")

    json.dump({"main": dict(zip(n3, p3.tolist())), "p": dict(zip(n3, pv3.tolist()))},
              open(Path(args.fact_dir) / "stats.json", "w"), ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
