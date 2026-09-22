"""冻结实验 provenance。实例销毁后这些信息无法重建。

模板效应是本文的核心发现，因此 **chat template 本身就是实验条件**，
必须像超参数一样被记录：模板原文、渲染后的 prompt 全文、模型 commit、
库版本、解码参数。少了任何一项，"FAR 0.996 vs 0.100" 都无法复现。
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icaa.modeling import load_config, resolve_model_cfg  # noqa: E402
from icaa.prompts import (  # noqa: E402
    AGENT_SYSTEM,
    COGNITION_AUTH_VARIANTS,
    COGNITION_NORM_VARIANTS,
    render_action_prompt,
    render_cognition_prompt,
)
from icaa.schema import load_actions, load_items  # noqa: E402


def sha(s) -> str:
    # 部分 tokenizer（如 Hermes）的 chat_template 是 {name: template} 的 dict
    if not isinstance(s, str):
        s = json.dumps(s, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="models.yaml 里的 key")
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--out", default="runs/provenance.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import transformers
    import torch
    from huggingface_hub import HfApi

    cfg = load_config()
    actions = load_actions(args.actions)
    items = load_items(args.items)
    # 每个配置都用同一条样本渲染，便于逐字 diff
    probe_item = next(i for i in items if i.level == "L3" and i.form == "imp")

    api = HfApi()
    snap = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip(),
        },
        "data": {
            "items_file": args.items,
            "n_items": len(items),
            "n_seeds": len({i.pair_id for i in items}),
            "items_sha": sha(Path(args.items).read_text()),
        },
        "prompts": {
            "AGENT_SYSTEM": AGENT_SYSTEM,
            "COGNITION_AUTH_VARIANTS": COGNITION_AUTH_VARIANTS,
            "COGNITION_NORM_VARIANTS": COGNITION_NORM_VARIANTS,
        },
        "probe_item": {"item_id": probe_item.item_id, "user_turn": probe_item.user_turn},
        "configs": {},
    }

    for key in args.configs:
        m = resolve_model_cfg(cfg, key)
        tok = AutoTokenizer.from_pretrained(m["hf_id"])
        act_prompt = render_action_prompt(probe_item, actions[probe_item.action], tok, m)
        aud_prompt = render_cognition_prompt(
            probe_item, actions[probe_item.action], tok, m, 0, kind="auth")
        try:
            rev = api.model_info(m["hf_id"]).sha
        except Exception:
            rev = None
        snap["configs"][key] = {
            "hf_id": m["hf_id"],
            "revision": rev,
            "settings": {k: v for k, v in m.items() if k != "hf_id"},
            "chat_template": tok.chat_template if isinstance(tok.chat_template, str)
            else json.dumps(tok.chat_template, ensure_ascii=False),
            "chat_template_sha": sha(tok.chat_template or ""),
            "rendered_action_prompt": act_prompt,
            "rendered_action_prompt_sha": sha(act_prompt),
            "rendered_audit_prompt": aud_prompt,
            "n_tokens_action": len(tok(act_prompt)["input_ids"]),
        }
        print(f"  {key:20s} rev={str(rev)[:12]} tmpl_sha={sha(tok.chat_template or '')} "
              f"tokens={len(tok(act_prompt)['input_ids'])}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(snap, ensure_ascii=False, indent=2))
    print(f"\n-> {args.out}")

    # 模板差异是核心实验条件，单独摘出来便于人工核对
    print("\n模板 sha 分组（同 sha = 同模板条件）")
    groups: dict[str, list[str]] = {}
    for k, v in snap["configs"].items():
        groups.setdefault(v["chat_template_sha"], []).append(k)
    for s, ks in groups.items():
        print(f"  {s}: {ks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
