"""只加载 tokenizer（不加载权重），打印渲染后的 prompt 与 marker token。

上 GPU 机器后**第一个**跑这个。它在几秒内暴露最常见的三类坑：
  1. chat template 不接受 tools 参数（gemma-2 就不接受，需走文本回退）
  2. tool spec 根本没进 prompt —— 那 A_l 读的是噪声
  3. action marker 不是单个 token，或 yes/no 解析到了奇怪的 id
"""

import argparse

import _bootstrap  # noqa: F401

from transformers import AutoTokenizer

from icaa.modeling import first_token_ids, load_config, resolve_model_cfg
from icaa.prompts import (
    NO_TOKENS,
    YES_TOKENS,
    render_action_prompt,
    render_cognition_prompt,
)
from icaa.schema import load_actions, load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v1.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--item-id", default=None, help="默认取第一个 L0 和第一个 L3")
    args = ap.parse_args()

    mcfg = resolve_model_cfg(load_config(), args.model)
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    actions = load_actions(args.actions)
    items = load_items(args.items)

    picks = (
        [it for it in items if it.item_id == args.item_id]
        if args.item_id
        else [next(i for i in items if i.level == "L0"), next(i for i in items if i.level == "L3")]
    )

    for it in picks:
        act = actions[it.action]
        p = render_action_prompt(it, act, tok, mcfg)
        print("=" * 78)
        print(f"[action prompt] {it.item_id}  intent={it.intent}  action={it.action}")
        print("=" * 78)
        print(p)
        n_tok = len(tok(p)["input_ids"])
        has_spec = act.tool["name"] in p and "description" in p
        print(f"\n-- {n_tok} tokens | tool spec 进入 prompt: {has_spec}")
        if not has_spec:
            print("   [FAIL] tool spec 没进 prompt。检查 supports_tools_in_template 配置。")

        print("\n" + "-" * 78)
        print(f"[cognition prompt v0] {it.item_id}")
        print("-" * 78)
        print(render_cognition_prompt(it, act, tok, mcfg, 0))

    print("\n" + "=" * 78)
    yes_ids, no_ids = first_token_ids(tok, YES_TOKENS), first_token_ids(tok, NO_TOKENS)
    print(f"yes ids {yes_ids} -> {tok.convert_ids_to_tokens(yes_ids)}")
    print(f"no  ids {no_ids} -> {tok.convert_ids_to_tokens(no_ids)}")
    marker = mcfg.get("action_first_tokens", [])
    mids = first_token_ids(tok, marker)
    print(f"action marker (config fallback) {marker} -> ids {mids} "
          f"-> {tok.convert_ids_to_tokens(mids)}")
    for s in marker:
        pieces = tok.tokenize(s)
        if len(pieces) > 1:
            print(f"   [note] {s!r} 被切成 {pieces}，A_l 实际读的是首片 {pieces[0]!r}。"
                  "01_forward 的实测自动推断会覆盖它，以实测为准。")


if __name__ == "__main__":
    main()
