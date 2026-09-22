"""基线矩阵：执行前预测误调用（over-action detection）。

任务：在 should-not-act 样本（I=0）上，预测模型是否会发出工具调用（P=1）。
所有特征都在生成之前可得，因此结果可直接读作"能否在动手前拦下来"。

基线从弱到强，并标注各自的运行代价：
  surface   问号/长度/首词——排除伪线索
  tfidf     字符 n-gram 文本分类器（经典强文本基线）
  judge     模型自述判断 J（LLM-as-judge，代价是额外一次完整 forward）
  last      末层 hidden state 线性探针
  concat    多层拼接探针（When2Tool 风格）
  best      单层探针，层号由内层 CV 选出（不看测试折）
  oracle    ground-truth I（上界，非可用方法）

两种划分：seed 分组 CV，以及留出整类 matrix-verb 构式的 OOD。
"""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.schema import load_items, read_jsonl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERB = [("tell_me", r"^tell me\b"), ("explain", r"^explain\b"), ("describe", r"^describe\b"),
        ("show_me", r"^show me\b"), ("walk_lay_break_spell", r"^(walk me through|lay out|break down|spell out)\b"),
        ("draft_help_write", r"^(draft|help me word|write)\b"), ("could_can", r"^(could|can) you\b"),
        ("modal_question", r"^(should|would|does|do|is|are|will)\b"), ("wh_question", r"^(what|how|why|who|when|where)\b")]
def verb_of(s):
    for n, p in VERB:
        if re.match(p, s.strip(), re.I): return n
    return "other"

def fit_probe(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    solver = dict(solver="liblinear", dual=True) if Xtr.shape[1] > Xtr.shape[0] else {}
    m = LogisticRegression(C=1.0, max_iter=5000, **solver).fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte))

def evaluate(splits, feat_fn, y):
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2: continue
        pred[te] = feat_fn(tr, te)
    ok = ~np.isnan(pred)
    if len(set(y[ok].tolist())) < 2: return float("nan"), float("nan")
    return roc_auc_score(y[ok], pred[ok]), average_precision_score(y[ok], pred[ok])

def main():
    items = {i.item_id: i for i in load_items("data/pairs/pilot_v2.jsonl")}
    MODELS = ["qwen-7b-v2", "mistral-7b-v2", "hermes-8b-v2", "llama-8b-neutral-v2"]
    COST = {"surface": "0", "tfidf": "0", "judge": "1 extra forward", "last": "0",
            "concat": "0", "best": "0", "oracle": "--"}
    out = {}
    for m in MODELS:
        fwd = np.load(f"runs/{m}/forward.npz", allow_pickle=True)
        order = list(fwd["item_ids"])
        cog = {r["item_id"]: r for r in read_jsonl(f"runs/{m}/cognition.jsonl")}
        rol = {r["item_id"]: r for r in read_jsonl(f"runs/{m}/rollout.jsonl")}
        it = [items[i] for i in order]
        keep = np.array([x.intent == 0 and rol[i]["called_tool"] in (None, x.action)
                         for i, x in zip(order, it)])
        idx = np.where(keep)[0]
        H = fwd["H"][idx]; L = H.shape[1]
        sub = [it[k] for k in idx]; oid = [order[k] for k in idx]
        y = np.array([rol[i]["executed"] for i in oid])
        grp = np.array([x.pair_id for x in sub]); vgrp = np.array([verb_of(x.user_turn) for x in sub])
        txt = [x.user_turn for x in sub]
        Jc = np.array([np.mean(cog[i]["c_auth_per_variant"]) for i in oid])[:, None]
        surf = np.column_stack([[1.0*("?" in s) for s in txt], [len(s.split()) for s in txt],
                                [len(s) for s in txt]])
        mid = list(range(L // 4, L))
        cat = H[:, [mid[k] for k in np.linspace(0, len(mid)-1, 8).round().astype(int)], :]\
                .reshape(len(y), -1).astype(np.float32)
        feats = {"surface": surf, "tfidf": None, "judge": Jc, "last": H[:, -1, :].astype(np.float32),
                 "concat": cat, "best": None, "oracle": np.array([[0.0]] * len(y))}

        def make(name):
            if name == "tfidf":
                def f(tr, te):
                    p = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
                                      LogisticRegression(max_iter=2000))
                    p.fit([txt[i] for i in tr], y[tr])
                    return p.predict_proba([txt[i] for i in te])[:, 1]
                return f
            if name == "best":
                def f(tr, te):
                    inner = list(GroupKFold(3).split(H[tr], y[tr], grp[tr]))
                    best, bl = -1, L // 2
                    # 粗网格：相邻层高度冗余，全层搜索的代价不成比例
                    for l in np.linspace(1, L - 1, 9).round().astype(int):
                        s = []
                        for a, b in inner:
                            if len(set(y[tr][a].tolist())) < 2: continue
                            s.append(roc_auc_score(y[tr][b], fit_probe(
                                H[tr][a, l, :].astype(np.float32), y[tr][a],
                                H[tr][b, l, :].astype(np.float32))))
                        if s and np.mean(s) > best: best, bl = np.mean(s), l
                    return fit_probe(H[tr, bl, :].astype(np.float32), y[tr],
                                     H[te, bl, :].astype(np.float32))
                return f
            if name == "oracle":
                return lambda tr, te: np.zeros(len(te))     # I=0 恒定，无信息 → 0.5
            X = feats[name]
            return lambda tr, te: fit_probe(X[tr], y[tr], X[te])

        res = {}
        seed_sp = list(GroupKFold(5).split(H[:, 0, :], y, grp))
        verb_sp = [(np.where(vgrp != v)[0], np.where(vgrp == v)[0])
                   for v in sorted(set(vgrp)) if (vgrp == v).sum() >= 5]
        for name in ("surface", "tfidf", "judge", "last", "concat", "best"):
            a1, p1 = evaluate(seed_sp, make(name), y)
            a2, p2 = evaluate(verb_sp, make(name), y)
            res[name] = {"auroc_seed": a1, "auprc_seed": p1,
                         "auroc_verb": a2, "auprc_verb": p2, "cost": COST[name]}
        res["_n"] = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean())}
        out[m] = res
        print(f"\n### {m}  n={len(y)}  正类={int(y.sum())}  基率={y.mean():.3f}")
        print(f"{'baseline':10s} {'AUROC(seed)':>12s} {'AUPRC(seed)':>12s} {'AUROC(verb OOD)':>16s} {'代价':>16s}")
        for k in ("surface", "tfidf", "judge", "last", "concat", "best"):
            r = res[k]
            print(f"{k:10s} {r['auroc_seed']:12.3f} {r['auprc_seed']:12.3f} "
                  f"{r['auroc_verb']:16.3f} {r['cost']:>16s}")
    Path("runs/baselines.json").write_text(json.dumps(out, indent=2))
    print("\n-> runs/baselines.json")

if __name__ == "__main__":
    main()
