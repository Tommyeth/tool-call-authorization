"""跨模型汇总：C1 的最终证据表。

把每个模型的 failure typing + association test + probe 摘要拼成一张表，
回答 §6 的问题：knowing-doing 与 intent-failure 的分离是否跨 family 稳定，
还是 Qwen artifact。

用法:
    python scripts/06_compare_models.py --runs runs/qwen-7b-v2 runs/mistral-7b-v2 runs/hermes-8b-v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.analyze import association_test, failure_typing  # noqa: E402
from icaa.schema import load_items, read_jsonl  # noqa: E402


def load_run(run_dir: Path, items: dict) -> dict | None:
    cog_p, rol_p = run_dir / "cognition.jsonl", run_dir / "rollout.jsonl"
    if not (cog_p.exists() and rol_p.exists()):
        return None
    cog = {r["item_id"]: r for r in read_jsonl(cog_p)}
    rol = {r["item_id"]: r for r in read_jsonl(rol_p)}
    ids = [i for i in items if i in cog and i in rol]
    if not ids:
        return None

    I = np.array([items[i].intent for i in ids])
    C = np.array([cog[i].get("c_auth_bin", cog[i]["c_bin"]) for i in ids])
    E = np.array([rol[i]["executed"] for i in ids])
    scope = np.array([rol[i]["called_tool"] in (None, items[i].action) for i in ids])
    lv = np.array([items[i].level for i in ids])

    st = failure_typing(I, C, E, scope)
    assoc = association_test(I, C, E, scope)
    unan = np.array([bool(cog[i].get("c_auth_unanimous", 1)) for i in ids])
    st_u = failure_typing(I, C, E, scope & unan)
    assoc_u = association_test(I, C, E, scope & unan)

    out = {
        "n_items": len(ids),
        "c_sanity_L0": float(C[lv == "L0"].mean()) if (lv == "L0").any() else float("nan"),
        **{k: v for k, v in st.items() if not k.startswith("mask_")},
        "assoc": assoc,
        "frac_unanimous": float(unan.mean()),
        "st_u": {k: v for k, v in st_u.items() if not k.startswith("mask_")},
        "assoc_u": assoc_u,
        "exec_by_level": {L: float(E[lv == L].mean()) for L in ["L0", "L1", "L2", "L3", "L4"]
                          if (lv == L).any()},
    }
    for extra in ("probe_summary.json", "failure_typing.json", "forward_meta.json"):
        p = run_dir / extra
        if p.exists():
            out[extra.replace(".json", "")] = json.loads(p.read_text())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = {i.item_id: i for i in load_items(args.items)}
    results = {}
    for rd in args.runs:
        r = load_run(Path(rd), items)
        if r is None:
            print(f"[skip] {rd}: 产物不全（尚未跑完？）")
            continue
        results[Path(rd).name] = r

    if not results:
        print("没有可用结果")
        return 1

    w = max(len(k) for k in results) + 1
    print(f"\n{'=' * (w + 76)}")
    print("C1 跨模型证据表")
    print(f"{'=' * (w + 76)}")
    print(f"{'model':{w}s} {'n':>4s} {'FAR':>6s} {'MAR':>6s} {'KD':>4s} {'IF':>4s} "
          f"{'KD占比':>7s} {'exec|C=0':>9s} {'exec|C=1':>9s} {'OR':>6s} {'p':>8s}")
    for name, r in results.items():
        a = r["assoc"]
        print(f"{name:{w}s} {r['n_false_action']:4d} "
              f"{r['false_action_rate']:6.3f} {r['missed_action_rate']:6.3f} "
              f"{r['n_knowing_doing']:4d} {r['n_intent_failure']:4d} "
              f"{r['knowing_doing_share']:7.3f} "
              f"{a['exec_rate_given_C0']:9.3f} {a['exec_rate_given_C1']:9.3f} "
              f"{a['odds_ratio_C1_vs_C0']:6.2f} {a['fisher_p']:8.4f}")

    print("\n判读：OR > 1 且 p < 0.05 = C 预测执行，分桶携带信息（非 C 边际分布的影子）。")
    print("      若三个 family 都满足，C1 就不是 Qwen artifact。")

    print(f"\n{'=' * (w + 76)}")
    print("稳健性子集：只用三个 C_auth 变体全一致的样本（规则事先声明，非事后挑选）")
    print(f"{'=' * (w + 76)}")
    print(f"{'model':{w}s} {'一致率':>7s} {'n_FA':>5s} {'KD':>4s} {'IF':>4s} "
          f"{'exec|C=0':>9s} {'exec|C=1':>9s} {'OR':>6s} {'p':>8s}")
    for name, r in results.items():
        u, a = r["st_u"], r["assoc_u"]
        print(f"{name:{w}s} {r['frac_unanimous']:7.3f} {u['n_false_action']:5d} "
              f"{u['n_knowing_doing']:4d} {u['n_intent_failure']:4d} "
              f"{a['exec_rate_given_C0']:9.3f} {a['exec_rate_given_C1']:9.3f} "
              f"{a['odds_ratio_C1_vs_C0']:6.2f} {a['fisher_p']:8.4f}")
    print("  变体一致率 <0.85 的模型（C 不稳），主表应以这一套为准。")

    print(f"\n{'=' * (w + 76)}\n仪器健全性（这几项不过关，上表不可信）\n{'=' * (w + 76)}")
    print(f"{'model':{w}s} {'C@L0':>6s} {'concatAUC':>10s} {'onset':>6s} {'collapse':>9s} {'preamble':>9s}")
    for name, r in results.items():
        ps = r.get("probe_summary", {})
        ft = r.get("failure_typing", {})
        fm = r.get("forward_meta", {}).get("marker_check", {})
        print(f"{name:{w}s} {r['c_sanity_L0']:6.3f} "
              f"{ps.get('concat_baseline_auc', float('nan')):10.3f} "
              f"{ps.get('intent_onset_layer', -1):6d} "
              f"{ft.get('collapse_cos_max_late_half', float('nan')):9.3f} "
              f"{fm.get('preamble_rate', float('nan')):9.3f}")

    print(f"\n{'=' * (w + 76)}\n执行率 × 歧义层（L4 元讨论是否普遍触发执行）\n{'=' * (w + 76)}")
    lv_all = ["L0", "L1", "L2", "L3", "L4"]
    print(f"{'model':{w}s} " + " ".join(f"{L:>6s}" for L in lv_all))
    for name, r in results.items():
        print(f"{name:{w}s} " + " ".join(f"{r['exec_by_level'].get(L, float('nan')):6.2f}" for L in lv_all))
    print("  I=1:  L0 L1        I=0:  L2 L3 L4")

    print(f"\n{'=' * (w + 76)}\n层位置（跨 family 是否一致）\n{'=' * (w + 76)}")
    print(f"{'model':{w}s} {'L':>3s} {'intent':>7s} {'rel':>6s} {'action':>7s} {'rel':>6s} "
          f"{'sepAUCmax':>10s} {'A_l有效':>8s}")
    for name, r in results.items():
        ps, ft = r.get("probe_summary", {}), r.get("failure_typing", {})
        nl = ps.get("n_layers", 0) or 1
        io = ps.get("intent_onset_layer", -1)
        ao = ft.get("action_onset_layer_auc70", -1)
        ok = ft.get("A_l_analysis_valid", None)
        print(f"{name:{w}s} {nl:3d} {io:7d} {io / nl:6.2f} {ao:7d} {ao / nl:6.2f} "
              f"{ft.get('sep_auc_max', float('nan')):10.3f} "
              f"{'-' if ok is None else ('是' if ok else '否'):>8s}")
    print("  rel = 层号 / 总层数。层数不同的模型必须按相对深度比较。")
    print("  A_l 有效 = preamble_rate ≤ 0.3（规则事先声明）；为'否'的模型只保留行为层结论。")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
