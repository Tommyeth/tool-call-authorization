"""audit → action cross-context probe transfer。

这是整条路线的生死线。目前已知：在**原始 action forward** 里 I 很早就线性可读。
但 J（模型自述判断）来自**另一次、换了提示词的 forward**，所以还不能说
"原始执行 pass 里存在并使用了同一个授权判断"。

做法：在 audit context 上训练授权方向 w_audit，**完全冻结**，
直接应用到 action context 的 hidden states。不重新拟合、不重新标定。

  迁移成功 => audit 与 action 两种上下文共享可迁移的 authorization representation
  迁移失败 => 二者是 context-specific 表示，论文措辞必须降级为
              "model self-audit and native action policy are behaviorally inconsistent"

对照（缺一不可，否则 transfer AUROC 无法解释）：
  * within-context 上界：同 context 训练+测试
  * random-label 对照：打乱标签重训，测容量记忆
  * embedding 层对照：layer 0 应接近 0.5
  * 按 semantic seed 分组，audit/action 两侧用同一套 seed 划分
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.modeling import load_config, load_model, resolve_model_cfg  # noqa: E402
from icaa.prompts import render_cognition_prompt  # noqa: E402
from icaa.schema import load_actions, load_items, read_jsonl  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


@torch.no_grad()
def audit_hidden(model, tok, device, items, actions, mcfg, variant=0) -> np.ndarray:
    """audit context 最后一个 prompt token 的逐层 hidden states。"""
    out = []
    for it in tqdm(items, desc="audit-forward"):
        p = render_cognition_prompt(it, actions[it.action], tok, mcfg, variant, kind="auth")
        enc = tok(p, return_tensors="pt").to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16))
    return np.stack(out)


def _fit(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, solver="liblinear", dual=True, max_iter=5000)
    clf.fit(sc.transform(X), y)
    return sc, clf


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def transfer_curve(Xtr, ytr, Xte, yte, groups, n_splits=5) -> np.ndarray:
    """按 seed 分组：在 train context 的训练折上拟合，应用到 test context 的对应测试折。"""
    L = Xtr.shape[1]
    pred = np.zeros((len(yte), L))
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    for tr, te in gkf.split(Xtr[:, 0, :], ytr, groups):
        for l in range(L):
            sc, clf = _fit(Xtr[tr, l, :].astype(np.float32), ytr[tr])
            pred[te, l] = clf.decision_function(sc.transform(Xte[te, l, :].astype(np.float32)))
    return np.array([auc(yte, pred[:, l]) for l in range(L)])


def cross_form_transfer(Xtr, Xte, y, form, groups, n_splits=5) -> dict:
    """跨 context + 跨句式 + 跨 seed 的严格迁移。

    audit prompt 与 action prompt 逐字共享用户话语，所以同句式迁移可能只是
    "两个 prompt 里有同样的词"。这里训练只用 imp、测试只用 int（反之亦然），
    且 train/test 的 semantic seed 不重叠——三重隔离后仍能迁移，才排除得掉词汇解释。
    """
    L = Xtr.shape[1]
    out = {}
    for a, b in [("imp", "int"), ("int", "imp")]:
        ma, mb = form == a, form == b
        pred = np.full((len(y), L), np.nan)
        gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        for tr, te in gkf.split(Xtr[:, 0, :], y, groups):
            tr_i = np.intersect1d(tr, np.where(ma)[0])   # 训练：audit + 形式 a
            te_i = np.intersect1d(te, np.where(mb)[0])   # 测试：action + 形式 b + 不同 seed
            if len(set(y[tr_i].tolist())) < 2 or len(te_i) == 0:
                continue
            for l in range(L):
                sc, clf = _fit(Xtr[tr_i, l, :].astype(np.float32), y[tr_i])
                pred[te_i, l] = clf.decision_function(
                    sc.transform(Xte[te_i, l, :].astype(np.float32)))
        ok = ~np.isnan(pred[:, 0])
        out[f"{a}2{b}"] = [auc(y[ok], pred[ok, l]) for l in range(L)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    mcfg = resolve_model_cfg(load_config(), args.model)
    items_all = {i.item_id: i for i in load_items(args.items)}
    actions = load_actions(args.actions)
    fwd = np.load(rd / "forward.npz", allow_pickle=True)
    order = list(fwd["item_ids"])
    items = [items_all[i] for i in order]
    H_act = fwd["H"]

    cache = rd / "audit_hidden.npy"
    if cache.exists():
        H_aud = np.load(cache)
        print(f"复用 {cache}")
    else:
        model, tok, device = load_model(mcfg)
        H_aud = audit_hidden(model, tok, device, items, actions, mcfg)
        np.save(cache, H_aud)
        del model
        torch.cuda.empty_cache()

    assert H_aud.shape == H_act.shape, (H_aud.shape, H_act.shape)
    L = H_act.shape[1]
    I = np.array([it.intent for it in items])
    grp = np.array([it.pair_id for it in items])
    cog = {r["item_id"]: r for r in read_jsonl(rd / "cognition.jsonl")}
    J = np.array([cog[i]["c_auth_bin"] for i in order])

    rng = np.random.default_rng(0)
    I_perm = rng.permutation(I)

    res = {
        "within_action_I": transfer_curve(H_act, I, H_act, I, grp).tolist(),
        "within_audit_I": transfer_curve(H_aud, I, H_aud, I, grp).tolist(),
        "audit_to_action_I": transfer_curve(H_aud, I, H_act, I, grp).tolist(),
        "action_to_audit_I": transfer_curve(H_act, I, H_aud, I, grp).tolist(),
        "audit_to_action_J": transfer_curve(H_aud, J, H_act, J, grp).tolist(),
        "control_permuted_I": transfer_curve(H_aud, I_perm, H_act, I_perm, grp).tolist(),
    }
    form = np.array([it.form for it in items])
    res.update({f"xform_{k}": v for k, v in
                cross_form_transfer(H_aud, H_act, I, form, grp).items()})
    (rd / "audit_transfer.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    print(f"\n{'layer':6s} " + " ".join(f"{k[:14]:>15s}" for k in res))
    for l in range(L):
        print(f"{l:6d} " + " ".join(f"{res[k][l]:15.3f}" for k in res))

    a2a = np.array(res["audit_to_action_I"])
    ctrl = np.array(res["control_permuted_I"])
    within = np.array(res["within_action_I"])
    print(f"\nlayer0 对照 (应 ~0.5): audit→action = {a2a[0]:.3f}")
    print(f"随机标签对照 最大值    : {np.nanmax(ctrl):.3f}")
    print(f"audit→action 峰值      : {np.nanmax(a2a):.3f} @ L{int(np.nanargmax(a2a))}")
    print(f"within-action 上界峰值  : {np.nanmax(within):.3f}")
    for k in ("xform_imp2int", "xform_int2imp"):
        c = np.array(res[k])
        print(f"跨句式 {k:16s}: 峰值 {np.nanmax(c):.3f} @ L{int(np.nanargmax(c))}")
    xf = max(np.nanmax(res["xform_imp2int"]), np.nanmax(res["xform_int2imp"]))
    print(f"\n[跨句式判读] audit(imp)→action(int) 与反向的最好成绩 {xf:.3f}")
    print("  >0.75 => 迁移不能用'两个 prompt 共享词汇'解释，授权表示是真的共享")
    print("  ~0.5  => 同句式的 0.999 主要是词汇平凡解，结论必须降级")

    gap = np.nanmax(a2a) - np.nanmax(ctrl)
    if np.nanmax(a2a) > 0.75 and gap > 0.2:
        print("\n[结论] 迁移成功：audit context 学到的授权方向在原始 action forward 中依然有效，"
              "可以说两种上下文共享可迁移的 authorization representation。")
    else:
        print("\n[结论] 迁移失败或过弱：不能声称原始 forward 内部存在与 audit 一致的授权判断。"
              "论文措辞降级为 behavioral inconsistency。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
