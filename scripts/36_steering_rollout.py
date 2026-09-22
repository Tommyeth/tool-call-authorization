"""Activation-steering rollouts on the original action prompt (protocol: compute/intervention-20260911/PROTOCOL.md).

For each condition, a constant vector alpha * g * d is added to the residual stream at the output of one
decoder layer, at every position (prompt and generated tokens), during greedy generation. Directions and the
layer come from scripts/35_steering_directions.py (audit states only). Conditions: baseline (no hook) and
alpha in {-2,-1,1,2} for d_auth, d_form (orthogonalised control), d_rand0, d_rand1 (random controls) at the
same shift norm. No tool is dispatched; outputs are parsed text proposals.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.modeling import load_config, resolve_model_cfg  # noqa: E402
from icaa.prompts import render_action_prompt  # noqa: E402
from icaa.rollout import parse_tool_call  # noqa: E402
from icaa.schema import load_actions, load_items  # noqa: E402

FALLBACK_REVISIONS = {"qwen-7b": "a09a35458c702b33eeacc393d103063234e8bc28",
                      "hermes-8b": "896ea440e5a9e6070e3d8a2774daf2b481ab425b"}


def recorded_revision(key):
    prov = ROOT / "runs/provenance.json"
    if prov.exists():
        rev = json.loads(prov.read_text()).get("configs", {}).get(key, {}).get("revision")
        if rev:
            return rev
    return FALLBACK_REVISIONS[key]


def degenerate(ids):
    if len(ids) < 20:
        return False
    uniq = len(set(ids)) / len(ids)
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    from collections import Counter
    top = Counter(grams).most_common(1)[0][1] if grams else 0
    return bool(uniq < 0.3 or top / max(1, len(grams)) > 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True); ap.add_argument("--directions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--limit", type=int)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2, -1, 1, 2])
    ap.add_argument("--directions-used", nargs="+", default=["d_auth", "d_form", "d_rand0", "d_rand1"])
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    mcfg = resolve_model_cfg(load_config(), args.model); rev = recorded_revision(args.model)
    items = load_items(ROOT / "data/pairs/pilot_v2.jsonl"); items = items[:args.limit] if args.limit else items
    actions = load_actions(ROOT / "data/actions.yaml")
    dz = np.load(args.directions); layer_idx = int(dz["layer"]) - 1; gap = float(dz["gap"])
    torch.manual_seed(0); torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], revision=rev, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], revision=rev, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to("cuda").eval()
    layers = model.model.layers
    assert 0 <= layer_idx < len(layers), (layer_idx, len(layers))
    state = {"delta": None}

    def hook(module, inputs, output):
        if state["delta"] is None:
            return output
        if isinstance(output, tuple):
            return (output[0] + state["delta"],) + tuple(output[1:])
        return output + state["delta"]
    handle = layers[layer_idx].register_forward_hook(hook)

    prompts = {it.item_id: render_action_prompt(it, actions[it.action], tok, mcfg) for it in items}
    conditions = [{"direction": "none", "alpha": 0.0}]
    for dname in args.directions_used:
        for a in args.alphas:
            conditions.append({"direction": dname, "alpha": float(a)})
    meta = {"model_key": args.model, "hf_id": mcfg["hf_id"], "revision": rev, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "attention": "sdpa", "batch_size": args.batch_size,
            "seed": 0, "do_sample": False, "max_new_tokens": args.max_new_tokens, "n_items": len(items),
            "prompt_renderer": "icaa.prompts.render_action_prompt (original representation prompt)",
            "directions_file": str(args.directions), "directions_sha256": hashlib.sha256(args.directions.read_bytes()).hexdigest(),
            "hook_decoder_layer_index": layer_idx, "cache_index": layer_idx + 1, "gap_g": gap,
            "shift": "alpha * g * unit direction added to the layer output at all positions",
            "conditions": conditions, "chat_template_sha256": hashlib.sha256(json.dumps(tok.chat_template, sort_keys=True).encode()).hexdigest()
            if not isinstance(tok.chat_template, str) else hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            "data_sha256": hashlib.sha256((ROOT / "data/pairs/pilot_v2.jsonl").read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    eos = model.generation_config.eos_token_id; eos_ids = set(eos if isinstance(eos, list) else [eos])
    summary = []
    for c in conditions:
        tag = f"{c['direction']}_a{c['alpha']:+.0f}" if c["direction"] != "none" else "baseline_a0"
        f = out / f"rollout_{tag}.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        done = {r["item_id"] for r in rows}
        if c["direction"] == "none":
            state["delta"] = None
        else:
            vec = torch.tensor(dz[c["direction"]], dtype=torch.float32) * (c["alpha"] * gap)
            state["delta"] = vec.to("cuda", dtype=torch.bfloat16)
        pending = [it for it in items if it.item_id not in done]; start = time.time()
        with f.open("a") as fh, torch.inference_mode():
            for pos in range(0, len(pending), args.batch_size):
                batch = pending[pos:pos + args.batch_size]
                enc = tok([prompts[it.item_id] for it in batch], padding=True, return_tensors="pt").to("cuda")
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, pad_token_id=tok.pad_token_id)
                new = gen[:, enc["input_ids"].shape[1]:]
                texts = tok.batch_decode(new, skip_special_tokens=False)
                for it, t, ids_t in zip(batch, texts, new):
                    ids = ids_t.tolist(); first_eos = next((k for k, v in enumerate(ids) if v in eos_ids), None)
                    actual = ids[:first_eos + 1] if first_eos is not None else ids
                    ok, called = parse_tool_call(t, it.action)
                    rec = {"item_id": it.item_id, "direction": c["direction"], "alpha": c["alpha"],
                           "shift_norm": abs(c["alpha"]) * gap if c["direction"] != "none" else 0.0,
                           "executed": ok, "called_tool": called, "completion": t, "generated_token_ids": actual,
                           "ended_with_eos": first_eos is not None,
                           "truncated": first_eos is None and len(actual) >= args.max_new_tokens,
                           "degenerate": degenerate(actual),
                           "prompt_sha256": hashlib.sha256(prompts[it.item_id].encode()).hexdigest()}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); rows.append(rec)
                fh.flush()
                print(json.dumps({"condition": tag, "completed": len(rows), "total": len(items),
                                  "seconds": round(time.time() - start, 1)}), flush=True)
        by = {r["item_id"]: r for r in rows}
        y = np.array([by[it.item_id]["executed"] for it in items]); I = np.array([it.intent for it in items])
        summary.append({**c, "tag": tag, "n": len(y), "FAR": float(y[I == 0].mean()), "MAR": float(1 - y[I == 1].mean()),
                        "n_truncated": int(sum(by[it.item_id]["truncated"] for it in items)),
                        "n_degenerate": int(sum(by[it.item_id]["degenerate"] for it in items))})
        print(json.dumps(summary[-1]), flush=True)
    handle.remove()
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "COMPLETE").write_text(str(time.time()))


if __name__ == "__main__":
    main()
