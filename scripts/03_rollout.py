"""E(x)：真实 greedy rollout。"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from icaa.modeling import load_config, load_model, resolve_model_cfg
from icaa.rollout import run_rollout
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

    meta = run_rollout(model, tok, device, items, actions, mcfg, run_dir / "rollout.jsonl")
    print(json.dumps(meta, indent=2))
    if meta["wrong_tool"]:
        print(f"[note] {meta['wrong_tool']} 条调用了别的 tool，属 selection 错误，"
              "05 会自动从 failure typing 中剔除。")


if __name__ == "__main__":
    main()
