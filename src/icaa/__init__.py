"""Intent–Cognition–Action Alignment pilot.

I 来自数据构造，C 来自模型自述（独立 forward），A 来自 logit lens，E 来自真实 rollout。
四者的证据来源互不重叠——这是整套分析非循环的前提，改动任何一环前先读 data/SCHEMA.md。
"""

__all__ = ["schema", "prompts", "modeling", "forward", "elicit", "rollout", "probe", "analyze"]
