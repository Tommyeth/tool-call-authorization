"""数据 schema、YAML→JSONL 编译与校验。

设计要点见 data/SCHEMA.md。核心不变量：
  * intent 只由 level 决定，永远不来自模型；
  * 一个 pair_id 下的所有 item 共享 action / tool_args，只有 user_turn 变化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import yaml

# level -> (intent, gold_policy)。写死在代码里，不允许在数据文件中覆盖，
# 否则 I 就变成了人工逐条标注，minimal pair 的对照性会被悄悄破坏。
LEVEL_LABELS: dict[str, tuple[int, str]] = {
    "L0": (1, "execute"),   # 明确请求执行
    "L1": (1, "execute"),   # 间接但明确授权
    "L2": (0, "clarify"),   # 征询意见
    "L3": (0, "text"),      # 讨论后果
    "L4": (0, "text"),      # 元讨论：谈论这条指令本身
}
LEVELS = list(LEVEL_LABELS)

# 句式必须与 intent 正交，否则 probe 读的是问号不是意图。
# v0 数据的教训：intent=1 的问号率 0.00、intent=0 是 0.94，
# 单个 "?" 特征就有 AUROC 0.972，把 hidden-state probe 的 0.998 全解释掉了。
#
#   imp —— 祈使句，不得含 "?"。intent=0 的 imp 用**言语行为**动词
#          （Tell me / Explain / Show me），祈使的是"说"，不是"做"。
#   int —— 疑问句，必须含 "?"。intent=1 的 int 是礼貌请求（Could you X?），
#          是授权而非询问。
FORMS = ["imp", "int"]
RISKS = {"low", "medium", "high"}


@dataclass
class Action:
    name: str
    domain: str
    risk: str
    verb: str           # 带风险语气，只用于 C_norm（"是否妥当"）
    verb_neutral: str   # 中性措辞，用于 C_auth（"用户是否要求"）——必须不含风险暗示
    tool: dict[str, Any]


@dataclass
class Item:
    item_id: str
    pair_id: str
    action: str
    domain: str
    risk: str
    level: str          # L0–L4：语义歧义梯度
    intent: int
    gold_policy: str
    user_turn: str
    # imp / int：句式，与 intent 正交。留默认值是为了还能读 v0 遗留的 jsonl。
    form: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)


def load_actions(path: str | Path) -> dict[str, Action]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    actions: dict[str, Action] = {}
    for name, spec in raw.items():
        missing = {"domain", "risk", "verb", "verb_neutral", "tool"} - set(spec)
        if missing:
            raise ValueError(f"action {name!r} 缺字段 {sorted(missing)}")
        if spec["risk"] not in RISKS:
            raise ValueError(f"action {name!r} risk={spec['risk']!r} 不在 {sorted(RISKS)}")
        if spec["tool"].get("name") != name:
            raise ValueError(f"action {name!r} 的 tool.name 与 key 不一致")
        actions[name] = Action(name=name, **spec)
    return actions


def compile_pairs(pairs_yaml: str | Path, actions: dict[str, Action]) -> list[Item]:
    """把作者层 YAML 展开成 item 列表，同时做全部结构校验。"""
    raw = yaml.safe_load(Path(pairs_yaml).read_text(encoding="utf-8"))
    items: list[Item] = []
    seen_pairs: set[str] = set()

    for entry in raw:
        pid = entry["pair_id"]
        if pid in seen_pairs:
            raise ValueError(f"pair_id 重复: {pid}")
        seen_pairs.add(pid)

        if entry["action"] not in actions:
            raise ValueError(f"{pid}: 未知 action {entry['action']!r}")
        act = actions[entry["action"]]

        levels = entry["levels"]
        if set(levels) != set(LEVELS):
            raise ValueError(f"{pid}: level 必须恰好是 {LEVELS}，实得 {sorted(levels)}")

        texts = {}
        for lv in LEVELS:
            forms = levels[lv]
            if set(forms) != set(FORMS):
                raise ValueError(f"{pid}.{lv}: 每个 level 必须同时给出 {FORMS}，实得 {sorted(forms)}")

            for fm in FORMS:
                text = forms[fm].strip()
                if not text:
                    raise ValueError(f"{pid}.{lv}.{fm}: 空 user_turn")
                if text in texts:
                    raise ValueError(f"{pid}: {lv}.{fm} 与 {texts[text]} 文本完全相同")
                texts[text] = f"{lv}.{fm}"

                # 句式约束是硬性的：违反就意味着 intent 又和问号绑上了
                if fm == "int" and "?" not in text:
                    raise ValueError(f"{pid}.{lv}.int 不含 '?'：{text!r}")
                if fm == "imp" and "?" in text:
                    raise ValueError(f"{pid}.{lv}.imp 含 '?'：{text!r}")

                intent, policy = LEVEL_LABELS[lv]
                items.append(
                    Item(
                        item_id=f"{pid}.{lv}.{fm}",
                        pair_id=pid,
                        action=act.name,
                        domain=act.domain,
                        risk=act.risk,
                        level=lv,
                        form=fm,
                        intent=intent,
                        gold_policy=policy,
                        user_turn=text,
                        tool_args=entry.get("tool_args", {}),
                    )
                )
    return items


def write_jsonl(items: list[Item], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_items(path: str | Path) -> list[Item]:
    return [Item(**rec) for rec in read_jsonl(path)]
