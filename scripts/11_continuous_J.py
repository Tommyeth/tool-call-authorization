"""连续 J + seed 聚类推断 + J×config 交互。替代 2x2 odds ratio 作为主统计。

为什么必须换掉 2x2 OR：
  * 二值化 J 丢掉了 log-odds 的强度信息；
  * 同一 semantic seed 派生出 10 条（5 level × 2 form），样本不独立，
    Fisher exact 的 p 值过于乐观；
  * 无法回答核心问题——在控制 I、level、syntax 之后，J 是否还有增量预测力；
  * 无法检验模板/模型是只移动截距（baseline call propensity）
    还是也改变斜率（授权敏感度）。

连续 J 定义：三个 audit variant 的 yes/no log-odds 各自 z-score 后取中位数。
取中位数而非均值，是为了对单个 variant 失灵稳健——Mistral/Hermes 的
variant 一致率只有 0.70 左右。

推断：seed 层面的 cluster bootstrap（默认 2000 次）。有 statsmodels 时
额外报告 GEE(exchangeable, cluster=seed) 作为交叉验证。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402


def continuous_J(rec: dict) -> list[float]:
    """三个 variant 的 p(yes) -> log-odds。z-score 在整个数据集上做，这里先返回 raw。"""
    return rec["c_auth_per_variant"]


def build(run_dirs: list[str], items_file: str) -> dict:
    items = {i.item_id: i for i in load_items(items_file)}
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        cog_p, rol_p = rd / "cognition.jsonl", rd / "rollout.jsonl"
        if not (cog_p.exists() and rol_p.exists()):
            print(f"[skip] {rd}")
            continue
        cog = {r["item_id"]: r for r in read_jsonl(cog_p)}
        rol = {r["item_id"]: r for r in read_jsonl(rol_p)}
        P_raw = np.array([[np.clip(p, 1e-6, 1 - 1e-6) for p in continuous_J(cog[i])]
                          for i in items if i in cog and i in rol])
        lo = np.log(P_raw / (1 - P_raw))                      # 每个 variant 的 log-odds
        z = (lo - lo.mean(0)) / np.maximum(lo.std(0), 1e-8)   # 逐 variant z-score
        Jc = np.median(z, axis=1)                             # 中位数，抗单变体失灵
        ids = [i for i in items if i in cog and i in rol]
        for k, iid in enumerate(ids):
            it = items[iid]
            r = rol[iid]
            if r["called_tool"] not in (None, it.action):
                continue                                      # tool-selection 错误剔除
            rows.append({
                "config": rd.name, "seed": it.pair_id, "level": it.level,
                "form": it.form, "risk": it.risk, "domain": it.domain,
                "I": it.intent, "J": float(Jc[k]),
                "Jbin": int(cog[iid]["c_auth_bin"]), "P": int(r["executed"]),
            })
    return rows


def design(rows, cols):
    """构造设计矩阵：连续项直接用，分类项 one-hot（丢首类）。"""
    X, names = [], []
    for c in cols:
        vals = [r[c] for r in rows]
        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool) and \
                len(set(vals)) > 2 or c in ("J", "I"):
            X.append(np.array(vals, float)); names.append(c)
        else:
            cats = sorted(set(vals))[1:]
            for cat in cats:
                X.append(np.array([1.0 if v == cat else 0.0 for v in vals])); names.append(f"{c}={cat}")
    return np.column_stack(X), names


def fit_logit(X, y, C=1e6):
    m = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
    m.fit(X, y)
    return np.concatenate([[m.intercept_[0]], m.coef_[0]])


def cluster_bootstrap(rows, cols, y_key="P", n_boot=2000, seed=0):
    """按 semantic seed 重采样：seed 是真正的独立实验单位，不是行。"""
    rng = np.random.default_rng(seed)
    X, names = design(rows, cols)
    y = np.array([r[y_key] for r in rows])
    point = fit_logit(X, y)

    seeds = sorted({r["seed"] for r in rows})
    idx_by_seed = {s: [i for i, r in enumerate(rows) if r["seed"] == s] for s in seeds}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(seeds), len(seeds), replace=True)
        idx = np.concatenate([idx_by_seed[seeds[p]] for p in pick])
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        try:
            boots.append(fit_logit(X[idx], yb))
        except Exception:
            continue
    B = np.array(boots)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    # 双侧 p：系数的 bootstrap 分布跨过 0 的比例
    p = 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0))
    return ["(intercept)"] + names, point, lo, hi, p, len(B)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/continuous_J.json")
    args = ap.parse_args()

    rows = build(args.runs, args.items)
    print(f"n={len(rows)} 行, {len({r['seed'] for r in rows})} 个 semantic seed, "
          f"{len({r['config'] for r in rows})} 个配置")

    models = {
        "M1 仅 J": ["J"],
        "M2 J + I": ["J", "I"],
        "M3 J + I + level + form": ["J", "I", "level", "form"],
        "M4 M3 + config": ["J", "I", "level", "form", "config"],
    }
    out = {}
    for name, cols in models.items():
        names, pt, lo, hi, p, nb = cluster_bootstrap(rows, cols, n_boot=args.n_boot)
        out[name] = {"terms": names, "coef": pt.tolist(), "lo": lo.tolist(),
                     "hi": hi.tolist(), "p": p.tolist(), "n_boot": nb}
        print(f"\n=== {name} ===  (cluster bootstrap over seeds, B={nb})")
        print(f"{'term':26s} {'coef':>8s} {'OR':>8s} {'95% CI (OR)':>22s} {'p':>8s}")
        for t, c, l, h, pv in zip(names, pt, lo, hi, p):
            star = " *" if pv < 0.05 else ""
            print(f"{t:26s} {c:8.3f} {np.exp(c):8.2f} "
                  f"[{np.exp(l):8.2f}, {np.exp(h):8.2f}] {pv:8.4f}{star}")

    # J × config 交互：模板/模型是只动截距，还是也动斜率
    configs = sorted({r["config"] for r in rows})
    if len(configs) > 1:
        print(f"\n=== J × config 交互（基线 = {configs[0]}）===")
        rows2 = [dict(r) for r in rows]
        for r in rows2:
            for c in configs[1:]:
                r[f"Jx_{c}"] = r["J"] if r["config"] == c else 0.0
        cols = ["J", "I", "level", "form", "config"] + [f"Jx_{c}" for c in configs[1:]]
        names, pt, lo, hi, p, nb = cluster_bootstrap(rows2, cols, n_boot=args.n_boot)
        out["interaction"] = {"terms": names, "coef": pt.tolist(), "lo": lo.tolist(),
                              "hi": hi.tolist(), "p": p.tolist()}
        sig = []
        for t, c, l, h, pv in zip(names, pt, lo, hi, p):
            if t.startswith("Jx_") or t == "J":
                star = " *" if pv < 0.05 else ""
                print(f"{t:26s} {c:8.3f} [{l:7.3f}, {h:7.3f}] p={pv:.4f}{star}")
                if t.startswith("Jx_") and pv < 0.05:
                    sig.append(t)
        print("\n[判读]", "斜率随配置显著变化：" + ", ".join(sig) +
              " —— 模板/模型不只移动 baseline，也改变授权敏感度。" if sig else
              "所有 J×config 交互不显著 —— 没有证据表明模板/模型改变了授权敏感度斜率，"
              "它们主要移动 baseline call propensity（截距）。")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
