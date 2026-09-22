"""C(x)：模型自述的动作适当性。独立 forward，不带 tool spec。"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from icaa.elicit import run_elicit
from icaa.modeling import load_config, load_model, resolve_model_cfg
from icaa.schema import load_actions, load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v1.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    mcfg = resolve_model_cfg(load_config(), args.model)
    run_dir = Path(args.run_dir or f"runs/{args.model}")
    actions = load_actions(args.actions)
    items = load_items(args.items)
    model, tok, device = load_model(mcfg)

    meta = run_elicit(model, tok, device, items, actions, mcfg, run_dir / "cognition.jsonl")
    print(json.dumps(meta, indent=2))
    if meta["variant_agreement"] < 0.85:
        print("[warn] 三个改写变体一致率 <0.85：C 在测 prompt 形式而非模型判断。"
              "先改 prompts.COGNITION_VARIANTS 再往下走，否则 failure typing 的两个桶会互相污染。")


if __name__ == "__main__":
    main()
