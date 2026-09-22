"""授权方向与动作方向的几何关系：正确的零分布检验。

原来的闸门是 |cos| < 0.5。在 d≈3584 的空间里随机方向的 |cos| 典型值是
1/sqrt(d) ≈ 0.017，所以那个阈值完全没有判别力——0.119 被判为"通过"，
实际上是 ~11 个标准差之外。

这里做三层零分布，从弱到强：
  1. 随机方向零分布 —— 只回答"比随机正交更对齐吗"，最弱
  2. 置换标签零分布 —— 打乱 C 标签重训 probe，控制 probe 拟合本身带来的结构
  3. max-stat 零分布 —— 每次置换取后半层的最大 |cos|，控制"在 L 层里挑最大值"
                        这一多重比较。这才是可以写进论文的那个数。

另报 probe 方向自身的 bootstrap 稳定性：若方向本身在 seed 重采样下就不稳，
任何 cosine 都不值得解释。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def probe_dirs(H, y, layers):
    """逐层 probe 方向（反标准化回原空间），零方差层返回 nan。"""
    D = np.full((len(layers), H.shape[2]), np.nan, dtype=np.float32)
    for k, l in enumerate(layers):
        X = H[:, l, :].astype(np.float32)
        sc = StandardScaler().fit(X)
        if np.all(sc.scale_ < 1e-6):
            continue
        clf = LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000)
        clf.fit(sc.transform(X), y)
        D[k] = clf.coef_[0] / np.maximum(sc.scale_, 1e-8)
    return D


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--max-layers", type=int, default=8,
                    help="后半层等距下采样上限。对偶 solver 复杂度随 n^2 增长，"
                         "400 样本下每次拟合约 2.2s，全层×全置换要数小时。")
    ap.add_argument("--out", default="runs/geometry_null.json")
    args = ap.parse_args()

    items_all = {i.item_id: i for i in load_items(args.items)}
    res = {}

    for rd in args.run_dirs:
        rd = Path(rd)
        if not (rd / "forward.npz").exists():
            print(f"[skip] {rd}")
            continue
        fwd = np.load(rd / "forward.npz", allow_pickle=True)
        order = list(fwd["item_ids"])
        items = [items_all[i] for i in order]
        H = fwd["H"]
        a = unit(fwd["action_dir"].astype(np.float32))
        cog = {r["item_id"]: r for r in read_jsonl(rd / "cognition.jsonl")}
        J = np.array([cog[i]["c_auth_bin"] for i in order])
        d, L = H.shape[2], H.shape[1]
        late = list(range(L // 2, L))          # 后半层，动作决策成形区
        if len(late) > args.max_layers:        # 等距下采样，保留首尾
            late = [late[i] for i in np.linspace(0, len(late) - 1,
                                                 args.max_layers).round().astype(int)]

        obs = np.abs(unit(probe_dirs(H, J, late)) @ a)
        obs_max = float(np.nanmax(obs))
        obs_argmax = late[int(np.nanargmax(obs))]

        # 零分布 1：随机方向
        rng = np.random.default_rng(0)
        R = unit(rng.normal(size=(4000, d)).astype(np.float32))
        rand_cos = np.abs(R @ a)

        # 零分布 2/3：置换标签重训 probe，每次取后半层最大 |cos|
        perm_max = []
        for _ in range(args.n_perm):
            yp = rng.permutation(J)
            c = np.abs(unit(probe_dirs(H, yp, late)) @ a)
            if np.isfinite(c).any():
                perm_max.append(float(np.nanmax(c)))
        perm_max = np.array(perm_max)

        # 方向稳定性：seed 重采样下方向之间的平均 |cos|
        seeds = sorted({i.pair_id for i in items})
        idx_by = {s: [k for k, i in enumerate(items) if i.pair_id == s] for s in seeds}
        best_l = obs_argmax
        boots = []
        for _ in range(30):
            pick = rng.choice(len(seeds), len(seeds), replace=True)
            idx = np.concatenate([idx_by[seeds[p]] for p in pick])
            if len(set(J[idx].tolist())) < 2:
                continue
            boots.append(probe_dirs(H[idx], J[idx], [best_l])[0])
        B = unit(np.array(boots))
        stab = float(np.abs(B @ B.T)[np.triu_indices(len(B), 1)].mean()) if len(B) > 1 else float("nan")

        p_perm = float((perm_max >= obs_max).mean())
        z_rand = (obs_max - rand_cos.mean()) / rand_cos.std()
        z_perm = (obs_max - perm_max.mean()) / max(perm_max.std(), 1e-9)

        res[rd.name] = {
            "d": d, "obs_max_cos": obs_max, "obs_layer": obs_argmax,
            "rand_mean": float(rand_cos.mean()), "rand_std": float(rand_cos.std()),
            "perm_max_mean": float(perm_max.mean()), "perm_max_std": float(perm_max.std()),
            "z_vs_random": float(z_rand), "z_vs_perm_maxstat": float(z_perm),
            "p_perm_maxstat": p_perm, "n_perm": len(perm_max),
            "direction_stability": stab,
        }
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
        r = res[rd.name]
        print(f"\n### {rd.name}  (d={d}, 后半层 L{late[0]}–L{late[-1]})")
        print(f"  观测 max|cos|          {obs_max:.4f} @ L{obs_argmax}")
        print(f"  随机方向零分布          {r['rand_mean']:.4f} ± {r['rand_std']:.4f}"
              f"   (1/sqrt(d)={1/np.sqrt(d):.4f})  => z={z_rand:.1f}")
        print(f"  置换标签 max-stat 零分布 {r['perm_max_mean']:.4f} ± {r['perm_max_std']:.4f}"
              f"   => z={z_perm:.1f}, p={p_perm:.3f}  (B={len(perm_max)})")
        print(f"  probe 方向稳定性        {stab:.3f}  (seed bootstrap 下方向间平均 |cos|)")

    print(f"\n{'='*76}")
    print("判读：随机方向零分布几乎必然显著（高维下任何拟合方向都不随机），")
    print("      能写进论文的是**置换标签 max-stat** 那一列：它同时控制了")
    print("      probe 拟合结构和'在 L 层里挑最大值'的多重比较。")
    print("      方向稳定性低于 ~0.5 时，cosine 数值本身不值得解释。")
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
