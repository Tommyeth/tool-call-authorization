"""容量对照：用与 intent probe 完全相同的探针，从 h_l 预测"是否执行"。

为什么必须做这个对照：
  intent probe 是 3584 维的训练分类器，A_l 只是 1 个标量（logit lens 读出的
  单一方向）。直接比较两者的 onset layer，"意图早、动作晚"可能纯粹是容量差异，
  而不是真实的时序差异。

对照设计：只取 I=0 且 C=0 的样本（模型都判断"用户没要求"），
在其中区分 E=1（knowing-doing）与 E=0（correct refusal），逐层训 probe。
  * 若这条曲线也在第 20 层左右才抬起来 => 时序差异是真的，
    动作决策确实在后段才形成。
  * 若它早早就高 => 动作结果在早层已可从表征读出，
    只是不在 logit-lens 的那个方向上——结论完全不同。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.analyze import onset_layer  # noqa: E402
from icaa.probe import probe_layerwise_cv  # noqa: E402
from icaa.schema import load_items, read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fwd = np.load(run_dir / "forward.npz", allow_pickle=True)
    items = {i.item_id: i for i in load_items(args.items)}
    order = list(fwd["item_ids"])
    cog = {r["item_id"]: r for r in read_jsonl(run_dir / "cognition.jsonl")}
    rol = {r["item_id"]: r for r in read_jsonl(run_dir / "rollout.jsonl")}

    I = np.array([items[i].intent for i in order])
    C = np.array([cog[i].get("c_auth_bin", cog[i]["c_bin"]) for i in order])
    E = np.array([rol[i]["executed"] for i in order])
    scope = np.array([rol[i]["called_tool"] in (None, items[i].action) for i in order])
    grp = np.array([items[i].pair_id for i in order])
    H = fwd["H"]

    sel = scope & (I == 0) & (C == 0)          # 同 I、同 C，只有 E 不同
    n1, n0 = int((sel & (E == 1)).sum()), int((sel & (E == 0)).sum())
    print(f"对照集: n={int(sel.sum())}  (knowing-doing {n1} vs correct-refusal {n0})")
    if min(n1, n0) < 10:
        print("[warn] 某一侧样本 <10，曲线不可信")

    act = probe_layerwise_cv(H[sel], E[sel], grp[sel])
    ita = probe_layerwise_cv(H, I, grp)        # 同协议下的 intent probe，作参照

    res = {
        "n_kd": n1, "n_refusal": n0,
        "action_probe_auc": [round(float(x), 4) for x in act["auc"]],
        "intent_probe_auc": [round(float(x), 4) for x in ita["auc"]],
        "action_onset_auc70": onset_layer(act["auc"], 0.70),
        "intent_onset_auc70": onset_layer(ita["auc"], 0.70),
        "action_auc_max": float(np.nanmax(act["auc"])),
        "action_auc_argmax": int(np.nanargmax(act["auc"])),
    }
    (run_dir / "action_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    print(f"\n{'layer':6s} {'intent':>8s} {'action':>8s}")
    for l in range(len(act["auc"])):
        print(f"{l:6d} {ita['auc'][l]:8.3f} {act['auc'][l]:8.3f}")
    print(f"\nonset (AUROC>0.70 持续): intent=L{res['intent_onset_auc70']}  "
          f"action=L{res['action_onset_auc70']}")
    print(f"action probe 峰值 {res['action_auc_max']:.3f} @ L{res['action_auc_argmax']}")

    a, i_ = res["action_onset_auc70"], res["intent_onset_auc70"]
    if a >= 0 and i_ >= 0 and a - i_ >= 5:
        print(f"\n[结论] 同容量对照下 action 仍比 intent 晚 {a - i_} 层 —— 时序差异是真的。")
    elif a >= 0 and i_ >= 0:
        print(f"\n[结论] 同容量对照下两者相差仅 {a - i_} 层 —— 之前的'意图早、动作晚'"
              "主要是 logit lens 单方向 vs 高维 probe 的容量假象，不能当作发现。")
    else:
        print("\n[结论] 某一条曲线始终未达 0.70，无法比较 onset。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
