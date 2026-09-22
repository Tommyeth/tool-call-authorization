"""抽 h_l 与 A_l（一次 forward）。需要 GPU。

action marker token 用 L0 样本实测自动推断，并与 configs/models.yaml 的
fallback 值对照——不一致会打 warning 但不中断（不同 checkpoint 起手 token 可能不同）。
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from icaa.forward import run_forward
from icaa.modeling import (
    verify_action_tokens,
    first_token_ids,
    load_config,
    load_model,
    resolve_model_cfg,
)
from icaa.prompts import render_action_prompt
from icaa.schema import load_actions, load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="configs/models.yaml 里的 key")
    ap.add_argument("--items", default="data/pairs/pilot_v1.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--probe-n", type=int, default=16, help="用于自动推断 marker 的 L0 样本数")
    args = ap.parse_args()

    mcfg = resolve_model_cfg(load_config(), args.model)
    run_dir = Path(args.run_dir or f"runs/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)

    actions = load_actions(args.actions)
    items = load_items(args.items)
    model, tok, device = load_model(mcfg)

    # marker 由 chat template 规定，是常量；实测只用来核验，绝不用来定义。
    action_ids = first_token_ids(tok, mcfg.get("action_first_tokens", []))
    if not action_ids:
        raise RuntimeError(
            f"{args.model} 未配置 action_first_tokens。marker 必须显式给出，"
            "不能自动推断——见 modeling.verify_action_tokens 的说明。"
        )

    l0 = [it for it in items if it.level == "L0"][: args.probe_n]
    l0_prompts = [render_action_prompt(it, actions[it.action], tok, mcfg) for it in l0]
    check = verify_action_tokens(model, tok, l0_prompts, device, action_ids)
    print("marker 核验:", json.dumps(check, ensure_ascii=False))
    if not check["marker_seen"]:
        raise RuntimeError(
            f"L0 上从未把 {tok.convert_ids_to_tokens(action_ids)} 作为首 token 生成。"
            "要么 marker 配错，要么模型在明确指令下也不调工具——先查 chat template。"
        )
    if check["preamble_rate"] > 0.3:
        print(f"[warn] preamble rate {check['preamble_rate']:.0%}：模型常先说客套话再调工具，"
              "first-position 的 A_l 会低估动作倾向。这是已知局限，论文需写明。")

    meta = run_forward(
        model, tok, device, items, actions, mcfg, action_ids, run_dir / "forward.npz"
    )
    meta |= {
        "model": args.model,
        "hf_id": mcfg["hf_id"],
        "action_token_ids": action_ids,
        "action_tokens": tok.convert_ids_to_tokens(action_ids),
        "marker_check": check,
    }
    (run_dir / "forward_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
