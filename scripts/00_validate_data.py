"""YAML → JSONL 编译 + 结构校验 + 配额报告 + 表面线索闸门。不需要 GPU。

表面线索闸门是硬性的：v0 数据就是因为 intent 与问号完全共线（单特征 AUROC 0.972）
而报废。这里直接把检查内联进编译流程，让它不可能再悄悄回归。
"""

import argparse
from collections import Counter

import numpy as np

import _bootstrap  # noqa: F401

from icaa.schema import compile_pairs, load_actions, write_jsonl


def surface_gate(items) -> bool:
    """返回 True 表示通过。任何单一表面特征都不该逼近 intent。"""
    from sklearn.metrics import roc_auc_score

    y = np.array([i.intent for i in items])
    text = [i.user_turn for i in items]
    feats = {
        "含 '?'": np.array(["?" in t for t in text], float),
        "词数": np.array([len(t.split()) for t in text], float),
        "字符长度": np.array([len(t) for t in text], float),
    }
    print("\n    表面线索闸门（AUROC 取对称最大值，0.5 = 无信息）")
    worst = 0.0
    for name, f in feats.items():
        a = roc_auc_score(y, f)
        a = max(a, 1 - a)
        worst = max(worst, a)
        flag = " <-- 超标" if a > 0.75 else ""
        print(f"      {name:10s} {a:.3f}{flag}")
    if worst > 0.75:
        print(f"    [FAIL] 最强表面线索 AUROC={worst:.3f} > 0.75。"
              "intent 与句式没解耦，probe 结果不可信。")
        return False
    print(f"    [ok] 最强表面线索 AUROC={worst:.3f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["data/pairs/pilot_v1.yaml"],
                    help="可给多个 YAML，合并编译（pair_id 全局唯一）")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--out", default="data/pairs/pilot_v1.jsonl")
    args = ap.parse_args()

    actions = load_actions(args.actions)
    items = []
    seen = set()
    for src in args.pairs:
        part = compile_pairs(src, actions)
        dup = {i.pair_id for i in part} & seen
        if dup:
            raise ValueError(f"{src}: pair_id 与前面的文件重复 {sorted(dup)}")
        seen |= {i.pair_id for i in part}
        items += part
        print(f"    + {src}: {len({i.pair_id for i in part})} scenario / {len(part)} item")
    write_jsonl(items, args.out)

    by_risk = Counter(i.risk for i in items)
    by_action = Counter(i.action for i in items)
    n_pairs = len({i.pair_id for i in items})
    n_levels = len({i.level for i in items})

    print(f"OK  {n_pairs} scenario / {len(items)} item -> {args.out}")
    print(f"    intent=1: {sum(i.intent for i in items)}  intent=0: {sum(1 - i.intent for i in items)}")
    print("    risk    :", dict(by_risk))
    print("    action  :", dict(by_action))

    # 句式 × intent 交叉表：两行都该接近 0.50，否则 probe 会去读句式
    print("\n    句式 × intent（每个 intent 的 int(疑问句) 占比，目标 0.50）")
    for lab in (1, 0):
        sub = [i for i in items if i.intent == lab]
        rate = sum(i.form == "int" for i in sub) / len(sub)
        flag = " <-- 失衡" if abs(rate - 0.5) > 0.1 else ""
        print(f"      intent={lab}  n={len(sub):3d}  int 占比={rate:.2f}{flag}")

    ok = surface_gate(items)

    # 配额提醒（data/SCHEMA.md 末节）
    per_action_scenarios = {a: n // (n_levels * 2) for a, n in by_action.items()}
    thin = [a for a, n in per_action_scenarios.items() if n < 8]
    if thin:
        print(f"\n    [warn] 以下 action 的 scenario 数 <8，per-action 比例无统计意义: {sorted(thin)}")
    if len(items) < 400:
        print(f"    [warn] 仅 {len(items)} item。go/no-go 需要 ≥30 条误执行样本，"
              "通常要 400–600 item 才够。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
