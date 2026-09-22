"""模型加载、logit lens、marker token 解析。

logit lens 的定义（A_l 的来源）：
    logits_l = lm_head( final_norm( h_l ) )
即把第 l 层的 residual stream 直接投到词表。这是标准 tuned-lens 的无训练版本，
不引入任何需要拟合的参数——这一点很重要：A_l 必须与 probe 学到的东西无关，
否则 C 和 A 就成了同一个向量上的两个分类器（见 analyze.collapse_check）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path: str | Path = "configs/models.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_model_cfg(cfg: dict, key: str) -> dict:
    if key not in cfg["models"]:
        raise KeyError(f"未知模型 {key!r}，可选：{sorted(cfg['models'])}")
    return {**cfg.get("defaults", {}), **cfg["models"][key]}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(mcfg: dict):
    device = pick_device(mcfg.get("device", "auto"))
    dtype = DTYPES[mcfg.get("dtype", "bfloat16")]
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kwargs: dict[str, Any] = {"device_map": device if device != "cpu" else None}
    # gemma-2 的 attention soft-capping 与 SDPA/FlashAttention 不兼容，
    # 用默认后端会静默产生错误的 attention 输出——而 logit lens 读的正是 residual stream。
    if mcfg.get("attn_implementation"):
        kwargs["attn_implementation"] = mcfg["attn_implementation"]

    try:  # transformers>=5 用 dtype，旧版用 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["hf_id"], torch_dtype=dtype, **kwargs
        )
    if kwargs["device_map"] is None:
        model = model.to(device)
    model.eval()
    return model, tok, device


# --------------------------------------------------------------------------- lens


def _final_norm(model):
    for attr in ("model.norm", "model.final_layernorm", "transformer.ln_f"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"找不到 final norm，请为 {type(model).__name__} 补一条 attr 路径")


class LogitLens:
    """把任意层的 hidden state 投到词表。"""

    def __init__(self, model):
        self.norm = _final_norm(model)
        self.head = model.get_output_embeddings()
        # gemma-2 在 lm_head 之后有 softcapping，不加会让概率整体偏大
        self.softcap = getattr(model.config, "final_logit_softcapping", None)

    @torch.no_grad()
    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> logits (..., vocab)"""
        logits = self.head(self.norm(h.to(self.head.weight.dtype)))
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits


# ------------------------------------------------------------------- marker tokens


def first_token_ids(tok, strings: list[str]) -> list[int]:
    """取每个字符串的首 token id，去重后返回。"""
    ids = []
    for s in strings:
        enc = tok.encode(s, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


@torch.no_grad()
def verify_action_tokens(
    model, tok, prompts: list[str], device: str, marker_ids: list[int]
) -> dict:
    """核验 config 给的 tool-call marker，并测量 preamble rate。

    **autodetect 不得用来定义 marker。** marker 由模型 chat template 规定
    （Qwen 是 `<tool_call>`），是文档化的常量。早期版本用"首 token 众数前二"
    自动推断，在含礼貌问句的 L0 上把 `Sure` 也收了进来——而 `Sure` 同样开启
    纯文本回答，计入 A_l 等于把客套话当成动作倾向。

    返回:
      top_first_tokens —— 实测首 token 分布，用于人工核对
      marker_first_rate —— 首 token 直接就是 marker 的比例
      preamble_rate    —— 首 token 不是 marker 的比例。模型先说 "Sure," 再调工具
                          时，first-position 的 A_l 会低估动作倾向，这是必须在
                          论文里写明的已知局限。
    """
    counter: Counter[int] = Counter()
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        counter[int(model(**enc).logits[0, -1].argmax())] += 1

    n = max(1, sum(counter.values()))
    hit = sum(c for tid, c in counter.items() if tid in set(marker_ids))
    top = counter.most_common(5)
    return {
        "top_first_tokens": [
            {"id": tid, "token": tok.convert_ids_to_tokens([tid])[0], "count": c}
            for tid, c in top
        ],
        "marker_first_rate": hit / n,
        "preamble_rate": 1 - hit / n,
        "marker_seen": hit > 0,
    }
