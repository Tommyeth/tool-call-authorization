"""Re-extract audit (variant 1) hidden states with explicit item IDs.

The historical ``runs/<model>-v2/audit_hidden.npy`` has no embedded item IDs; its
alignment with ``forward.npz`` relies on the extraction order. This script
re-extracts the same audit prompt (cognition variant 0, kind ``auth``) at the
pinned checkpoint revision, stores the states together with item IDs and
prompt hashes, and (``--compare``) checks the historical cache against the new
one: per-item cosine similarity per layer and a nearest-neighbour identity test.

Extraction (GPU):
  python scripts/33_audit_cache_with_ids.py --model qwen-7b --out-dir runs/revision-20260911/audit_ids/qwen-7b-v2
Comparison (CPU, local):
  python scripts/33_audit_cache_with_ids.py --model qwen-7b --out-dir runs/revision-20260911/audit_ids/qwen-7b-v2 \
      --compare runs/qwen-7b-v2/audit_hidden.npy --compare-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.modeling import load_config, resolve_model_cfg  # noqa: E402
from icaa.prompts import COGNITION_AUTH_VARIANTS, render_cognition_prompt  # noqa: E402
from icaa.schema import load_actions, load_items  # noqa: E402

# Recorded revisions (Supplement Table S2 / runs/provenance.json).
FALLBACK_REVISIONS = {
    "qwen-7b": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistral-7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "hermes-8b": "896ea440e5a9e6070e3d8a2774daf2b481ab425b",
    "llama-8b": "0e9e39f249a16976918f6564b8830bc894c89659",
    "llama-8b-neutral": "0e9e39f249a16976918f6564b8830bc894c89659",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recorded_revision(key: str) -> str:
    prov = ROOT / "runs/provenance.json"
    if prov.exists():
        cfg = json.loads(prov.read_text()).get("configs", {}).get(key, {})
        if cfg.get("revision"):
            return cfg["revision"]
    return FALLBACK_REVISIONS[key]


def extract(args, mcfg, items, actions, out: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = recorded_revision(args.model)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], revision=revision,
                                                 torch_dtype=torch.bfloat16).to("cuda").eval()
    prompts = [render_cognition_prompt(it, actions[it.action], tok, mcfg, 0, kind="auth") for it in items]
    states, start = [], time.time()
    with torch.inference_mode():
        for index, prompt in enumerate(prompts):
            enc = tok(prompt, return_tensors="pt").to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            states.append(torch.stack([h[0, -1] for h in hs]).float().cpu().numpy().astype(np.float16))
            if (index + 1) % 50 == 0:
                print(json.dumps({"model": args.model, "done": index + 1, "seconds": round(time.time() - start, 1)}), flush=True)
    H = np.stack(states)
    ids = np.asarray([it.item_id for it in items])
    np.savez_compressed(out / "audit_hidden_ids.npz", H=H, item_ids=ids,
                        prompt_sha256=np.asarray([sha(p) for p in prompts]))
    meta = {
        "model_key": args.model, "hf_id": mcfg["hf_id"], "revision": revision,
        "variant_index": 0, "variant_template_sha256": sha(COGNITION_AUTH_VARIANTS[0]),
        "chat_template_sha256": sha(tok.chat_template if isinstance(tok.chat_template, str)
                                    else json.dumps(tok.chat_template or {}, sort_keys=True, ensure_ascii=False)),
        "shape": list(H.shape), "dtype": "float16 (from bfloat16 forward)",
        "position": "final prompt position after add_generation_prompt", "n_items": len(items),
        "item_order": "data/pairs/pilot_v2.jsonl order (identical to forward.npz item_ids)",
        "data_sha256": hashlib.sha256((ROOT / "data/pairs/pilot_v2.jsonl").read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
        "elapsed_seconds": round(time.time() - start, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"saved": str(out / "audit_hidden_ids.npz"), **{k: meta[k] for k in ["shape", "elapsed_seconds"]}}))


def compare(args, items, out: Path):
    new = np.load(out / "audit_hidden_ids.npz", allow_pickle=True)
    H_new = new["H"].astype(np.float32)
    ids = list(new["item_ids"])
    if ids != [it.item_id for it in items]:
        raise ValueError("New cache item order differs from data order")
    H_old = np.load(args.compare, mmap_mode="r")
    if H_old.shape != H_new.shape:
        raise ValueError(f"Shape mismatch: historical {H_old.shape} vs new {H_new.shape}")
    H_old = np.asarray(H_old, dtype=np.float32)
    n, L, _ = H_new.shape

    def unit(x):
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)

    per_layer = []
    for layer in range(L):
        a, b = unit(H_old[:, layer]), unit(H_new[:, layer])
        diag = np.sum(a * b, axis=1)
        sim = a @ b.T
        nn_old_to_new = np.argmax(sim, axis=1)
        nn_new_to_old = np.argmax(sim, axis=0)
        off = sim.copy(); np.fill_diagonal(off, -np.inf)
        per_layer.append({
            "layer": layer, "mean_diag_cosine": float(diag.mean()), "min_diag_cosine": float(diag.min()),
            "identity_matches_old_to_new": int((nn_old_to_new == np.arange(n)).sum()),
            "identity_matches_new_to_old": int((nn_new_to_old == np.arange(n)).sum()),
            "min_margin_diag_minus_best_other": float((diag - off.max(axis=1)).min()),
        })
    mid = int(round(0.55 * (L - 1)))
    mismatched = [ids[i] for i in range(n)
                  if np.argmax(unit(H_old[:, mid]) @ unit(H_new[i, mid])) != i]
    report = {
        "historical_cache": str(args.compare), "new_cache": str(out / "audit_hidden_ids.npz"),
        "historical_sha256": hashlib.sha256(Path(args.compare).read_bytes()).hexdigest(),
        "n_items": n, "n_layers": L, "mid_layer": mid,
        "all_layers_identity_aligned": all(r["identity_matches_old_to_new"] == n and r["identity_matches_new_to_old"] == n
                                           for r in per_layer if r["layer"] > 0),
        "mid_layer_summary": per_layer[mid], "last_layer_summary": per_layer[-1],
        "mid_layer_mismatched_item_ids": mismatched, "per_layer": per_layer,
        "interpretation": ("Layer 0 is the embedding of a shared final token and is excluded from the identity test. "
                           "Values differ numerically because hardware/software differ; the test is whether each "
                           "historical row is nearest to the new row of the same item."),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["all_layers_identity_aligned", "mid_layer_summary", "last_layer_summary", "mid_layer_mismatched_item_ids"]}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="configs/models.yaml key, e.g. qwen-7b, hermes-8b")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--items", default="data/pairs/pilot_v2.jsonl")
    ap.add_argument("--actions", default="data/actions.yaml")
    ap.add_argument("--compare", type=Path, help="Historical audit_hidden.npy to compare against")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    mcfg = resolve_model_cfg(load_config(), args.model)
    items = load_items(args.items)
    actions = load_actions(args.actions)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.compare_only:
        extract(args, mcfg, items, actions, out)
    if args.compare is not None:
        compare(args, items, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
