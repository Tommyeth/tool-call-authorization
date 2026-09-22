"""Layer-wise intent probe：意图在第几层变得线性可读。

主协议（holdout）：L0/L4 训 → L1–L3 测。
对照协议（cv）：全 level、按 pair_id 分组的 GroupKFold。
concat baseline：When2Tool 风格多层拼接，用作管线 sanity check。
"""

import argparse
import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from icaa.probe import probe_concat_baseline, probe_layerwise_cv, probe_layerwise_holdout
from icaa.schema import load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v1.jsonl")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fwd = np.load(run_dir / "forward.npz", allow_pickle=True)
    items = {it.item_id: it for it in load_items(args.items)}
    order = [items[i] for i in fwd["item_ids"]]

    H = fwd["H"]
    y = np.array([it.intent for it in order])
    groups = np.array([it.pair_id for it in order])
    levels = np.array([it.level for it in order])
    train_mask = np.isin(levels, ["L0", "L4"])

    ho = probe_layerwise_holdout(H, y, train_mask)
    cv = probe_layerwise_cv(H, y, groups)
    n_layers = H.shape[1]
    mid = list(range(n_layers // 4, n_layers))  # 拼中后层，前几层基本无信号
    cat = probe_concat_baseline(H, y, groups, layers=mid)

    np.savez_compressed(
        run_dir / "probe_intent.npz",
        auc_holdout=ho["auc"], auc_cv=cv["auc"],
        pred_holdout=ho["pred"], pred_cv=cv["pred"],
        direction=ho["direction"], test_mask=ho["test_mask"],
    )

    best_ho = int(np.nanargmax(ho["auc"]))
    best_cv = int(np.nanargmax(cv["auc"]))
    # "意图开始可读" 的层：holdout AUROC 首次并持续超过 0.75
    onset = next((l for l in range(n_layers) if np.all(ho["auc"][l:] > 0.75)), -1)

    summary = {
        "n_layers": n_layers,
        "concat_layers_used": len(cat.get("layers", mid)),
        "holdout_best_layer": best_ho, "holdout_best_auc": float(ho["auc"][best_ho]),
        "cv_best_layer": best_cv, "cv_best_auc": float(cv["auc"][best_cv]),
        "intent_onset_layer": onset,
        "concat_baseline_auc": cat["auc"],
    }
    (run_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("\nper-layer AUROC (holdout L0/L4 -> L1-L3):")
    print("  " + "  ".join(f"{l}:{a:.3f}" for l, a in enumerate(ho["auc"])))

    if cat["auc"] < 0.85:
        print("\n[warn] concat baseline AUROC < 0.85，明显低于 When2Tool 报告的量级。"
              "先查抽取管线（读取位置是否为 prompt 末 token、chat template 是否带上 tools、"
              "dtype 是否被 float16 截断），修好之前的任何 finding 都不可信。")


if __name__ == "__main__":
    main()
