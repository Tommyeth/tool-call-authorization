"""一次 forward 同时产出 h_l 和 A_l。

为什么合在一起：两者必须来自**同一个前向、同一个位置**（prompt 最后一个 token，
生成第一个 token 之前）。分两次跑会因为 padding / 截断差异引入不可控偏移，
而后面 §11 的核心论证依赖 "A_l 在第几层翻转" 这种层级精度的比较。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .modeling import LogitLens
from .prompts import render_action_prompt
from .schema import Action, Item


@torch.no_grad()
def run_forward(
    model,
    tok,
    device: str,
    items: list[Item],
    actions: dict[str, Action],
    mcfg: dict,
    action_token_ids: list[int],
    out_path: str | Path,
) -> dict:
    lens = LogitLens(model)
    action_ids = torch.tensor(action_token_ids, device=device)

    H_all, A_all, M_all = [], [], []

    for it in tqdm(items, desc="forward"):
        prompt = render_action_prompt(it, actions[it.action], tok, mcfg)
        enc = tok(prompt, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)

        # hidden_states 长度 L+1：[0] 是 embedding 输出，[l] 是第 l 层之后
        hs = torch.stack([h[0, -1] for h in out.hidden_states])  # (L+1, d)
        logits = lens(hs)                                        # (L+1, vocab)
        probs = logits.float().softmax(-1)

        p_action = probs[:, action_ids].sum(-1)                  # A_l

        # margin：最强 action marker 与最强 non-action token 的 logit 差。
        # 比纯概率更能反映"倾向翻转"的时刻，且不受 softmax 温度影响。
        lg = logits.float()
        best_action = lg[:, action_ids].max(-1).values
        masked = lg.clone()
        masked[:, action_ids] = -float("inf")
        margin = best_action - masked.max(-1).values

        H_all.append(hs.float().cpu().numpy().astype(np.float16))
        A_all.append(p_action.cpu().numpy())
        M_all.append(margin.cpu().numpy())

    # unembedding 里的 "action 方向"：给 analyze.collapse_check 用。
    # 存 d 维向量而不是整个 W_U（后者动辄 1GB）。
    W_U = lens.head.weight
    action_dir = (W_U[action_ids].float().mean(0) - W_U.float().mean(0)).cpu().numpy()

    payload = {
        "item_ids": np.array([it.item_id for it in items]),
        "H": np.stack(H_all),          # (N, L+1, d) float16
        "A": np.stack(A_all),          # (N, L+1) 概率
        "A_margin": np.stack(M_all),   # (N, L+1) logit margin
        "action_token_ids": np.array(action_token_ids),
        "action_dir": action_dir,      # (d,)
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return {"n_items": len(items), "n_layers": payload["H"].shape[1], "d": payload["H"].shape[2]}
