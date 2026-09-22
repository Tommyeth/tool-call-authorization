"""分组泛化检验：probe 是在理解 speech act，还是在记模板？

按 semantic seed 分组只能挡住"记住某个场景"，挡不住"记住 Tell me whether 这个短语"
——它跨全部 40 个 seed 出现。真正的检验是把整类语言构式或整个动作域留出：

  * leave-one-matrix-verb-out —— 训练时没见过 Explain，测试全是 Explain 开头的
  * leave-one-domain-out      —— 训练时没见过 filesystem，测试全是 filesystem 动作

若留出后 AUROC 崩到接近随机，说明 probe 读的是词汇模板；
若保持高位，说明它读的是构式背后的授权语用。

同时报告纯文本 TF-IDF 在同样留出下的成绩作为参照——
若 probe 与 TF-IDF 一起崩，是数据问题；若 probe 撑住而 TF-IDF 崩，
才说明 hidden state 提供了词汇之外的信息。
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# 数据里实际使用的 matrix / speech-act 构式。分类看句首，未命中归 other。
VERB_PATTERNS = [
    ("tell_me", r"^tell me\b"),
    ("explain", r"^explain\b"),
    ("describe", r"^describe\b"),
    ("show_me", r"^show me\b"),
    ("walk_lay_break_spell", r"^(walk me through|lay out|break down|spell out)\b"),
    ("draft_help_write", r"^(draft|help me word|write)\b"),
    ("could_can", r"^(could you|can you)\b"),
    ("wh_question", r"^(what|how|why|who|which)\b"),
    ("modal_question", r"^(should|would|is|are|does|do|will|if)\b"),
]


def verb_group(text: str) -> str:
    t = text.strip().lower()
    for name, pat in VERB_PATTERNS:
        if re.match(pat, t):
            return name
    return "other"


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def probe_lopo(H, y, groups, layer):
    """leave-one-group-out：每次留出一整个组做测试。"""
    X = H[:, layer, :].astype(np.float32)
    pred = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr].tolist())) < 2 or not te.any():
            pred[te] = 0.5
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.decision_function(sc.transform(X[te]))
    return pred


def tfidf_lopo(text, y, groups):
    pred = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr].tolist())) < 2 or not te.any():
            pred[te] = 0.5
            continue
        m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1),
                          LogisticRegression(max_iter=2000))
        m.fit(np.array(text)[tr], y[tr])
        pred[te] = m.predict_proba(np.array(text)[te])[:, 1]
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--out", default="runs/group_holdout.json")
    args = ap.parse_args()

    items_all = {i.item_id: i for i in load_items(args.items)}
    res = {}

    for rd in args.run_dirs:
        rd = Path(rd)
        f = rd / "forward.npz"
        if not f.exists():
            print(f"[skip] {rd}")
            continue
        fwd = np.load(f, allow_pickle=True)
        order = list(fwd["item_ids"])
        items = [items_all[i] for i in order]
        H = fwd["H"]
        y = np.array([i.intent for i in items])
        text = [i.user_turn for i in items]
        vg = np.array([verb_group(i.user_turn) for i in items])
        dom = np.array([i.domain for i in items])
        seed = np.array([i.pair_id for i in items])
        # 用中层（相对深度 ~0.55），跨句式迁移的峰值区
        layer = int(round(0.55 * (H.shape[1] - 1)))

        r = {"layer": layer, "n_verb_groups": len(set(vg)), "n_domains": len(set(dom))}
        for name, g in [("verb", vg), ("domain", dom), ("seed", seed)]:
            r[f"probe_{name}"] = auc(y, probe_lopo(H, y, g, layer))
            r[f"tfidf_{name}"] = auc(y, tfidf_lopo(text, y, g))
        res[rd.name] = r

        print(f"\n### {rd.name}  (layer {layer} / {H.shape[1]-1})")
        print(f"{'留出维度':22s} {'probe':>8s} {'TF-IDF':>8s} {'差值':>7s}")
        for name, label in [("seed", "semantic seed"), ("verb", "matrix verb 构式"),
                            ("domain", "action domain")]:
            p_, t_ = r[f"probe_{name}"], r[f"tfidf_{name}"]
            print(f"  leave-one-{label:20s} {p_:8.3f} {t_:8.3f} {p_ - t_:+7.3f}")

    print(f"\n{'='*72}")
    print("verb 组分布:", {v: int((np.array([verb_group(i.user_turn)
          for i in items_all.values()]) == v).sum()) for v in
          sorted({verb_group(i.user_turn) for i in items_all.values()})})
    print("\n判读：")
    print("  probe 在 verb / domain 留出下仍高 => 读的是授权语用，不是词汇模板")
    print("  probe 与 TF-IDF 一起崩          => 数据本身没有跨构式泛化的信号")
    print("  probe 撑住而 TF-IDF 崩          => hidden state 提供了词汇之外的信息")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
