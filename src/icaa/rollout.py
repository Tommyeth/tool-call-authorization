"""E(x)：实际 greedy rollout，看模型到底有没有发出这个 tool call。

E 必须是"真的生成出来"的，不能用 A_L（末层 lens）代替。理由：末层 lens 与实际
采样之间还隔着 chat template 的收尾、EOS 竞争等因素，用 A_L 代替 E 会让
"cognition→action 失败" 变成一个定义上必然成立的命题。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from .prompts import render_action_prompt
from .schema import Action, Item

# 覆盖 qwen(<tool_call>)、gemma(```tool_call)、llama(<|python_tag|> 或裸 JSON)
_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>|```(?:tool_call|json)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)
_BARE_JSON_RE = re.compile(r'\{[^{}]*"name"\s*:\s*"([A-Za-z_]\w*)".*?\}', re.DOTALL)


def parse_tool_call(text: str, action_name: str) -> tuple[int, str | None]:
    """返回 (是否调用了目标 action, 实际调用的 tool 名)。

    调用了别的 tool 也记下来——这类样本要从 failure typing 里剔除，
    它属于 tool-selection 错误，不是本文研究的 authorization 错误。
    """
    for m in _BLOCK_RE.finditer(text):
        blob = m.group(1) or m.group(2)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = obj.get("name")
        if name:
            return int(name == action_name), name

    m = _BARE_JSON_RE.search(text)
    if m:
        return int(m.group(1) == action_name), m.group(1)

    return 0, None


@torch.no_grad()
def run_rollout(
    model, tok, device: str, items: list[Item], actions: dict[str, Action], mcfg: dict,
    out_path: str | Path,
) -> dict:
    records = []
    for it in tqdm(items, desc="rollout-E"):
        prompt = render_action_prompt(it, actions[it.action], tok, mcfg)
        enc = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=mcfg.get("max_new_tokens", 128),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
        )
        new_ids = gen[0, enc["input_ids"].shape[1]:]
        text = tok.decode(new_ids, skip_special_tokens=False)
        executed, called = parse_tool_call(text, it.action)
        records.append(
            {
                "item_id": it.item_id,
                "executed": executed,
                "called_tool": called,
                "first_token_id": int(new_ids[0]) if len(new_ids) else -1,
                "completion": text,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_exec = sum(r["executed"] for r in records)
    n_other = sum(1 for r in records if r["called_tool"] not in (None, "") and not r["executed"])
    return {"n": len(records), "executed": n_exec, "wrong_tool": n_other}
