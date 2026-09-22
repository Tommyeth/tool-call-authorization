"""Matched-input text baselines for the paired construction holdout.

Script 25 encodes only ``user_turn`` for TF-IDF and MiniLM, whereas the probes
read hidden states of the complete rendered action prompt (system instruction +
tool schema + user turn). This script re-runs the text baselines on the same 23
paired construction folds but with the model-agnostic *neutral* rendering text
as input:

    AGENT_SYSTEM + "\n\n" + _tools_as_text(action) + "\n\n" + user_turn

This is the plain-text content of the Llama-neutral action prompt before the
chat template is applied. It narrows the input-context gap; it does not match
encoder capacity, pretraining, or the exact chat-template tokens.

Folds, classifiers, regularization grid, and score conventions are imported
from scripts/25_paired_holdout.py so that the comparison stays like-for-like.
MiniLM is optional (``--model-dir``); TF-IDF runs without any download.

Usage:
  python scripts/32_matched_input_text_baseline.py --c-values 0.1 1 10
  python scripts/32_matched_input_text_baseline.py --model-dir <MiniLM snapshot dir>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.prompts import AGENT_SYSTEM, _tools_as_text  # noqa: E402
from icaa.schema import load_actions, load_items  # noqa: E402

spec = importlib.util.spec_from_file_location("paired25", ROOT / "scripts/25_paired_holdout.py")
p25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p25)

SCHEMA_VERSION = "matched-input-text-baseline-v1"


def full_prompt_text(item, action) -> str:
    return f"{AGENT_SYSTEM}\n\n{_tools_as_text(action)}\n\n{item.user_turn}"


def minilm_embeddings(model_dir: Path, texts: list[str], max_length: int, batch_size: int,
                      threads: int, seed: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    revision_marker = model_dir / "revision.txt"
    actual = (revision_marker.read_text().strip() if revision_marker.exists()
              else model_dir.name)
    if actual != p25.MINILM_REVISION:
        raise ValueError(f"Unexpected MiniLM revision: {actual}")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    if model.config.hidden_size != 384 or model.config.num_hidden_layers != 6:
        raise ValueError("MiniLM architecture mismatch")
    out, n_truncated = [], 0
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            lengths = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in batch]
            n_truncated += sum(length > max_length for length in lengths)
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    X = np.concatenate(out).astype(np.float32)
    if X.shape != (len(texts), 384) or not np.isfinite(X).all():
        raise ValueError("Invalid MiniLM embeddings")
    return X, n_truncated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=Path, default=ROOT / "data/pairs/pilot_v2.jsonl")
    ap.add_argument("--actions", type=Path, default=ROOT / "data/actions.yaml")
    ap.add_argument("--reference-splits", type=Path,
                    default=ROOT / "runs/revision-20260910/paired_holdout/splits.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/revision-20260911/matched_text")
    ap.add_argument("--model-dir", type=Path, help="Optional local MiniLM snapshot (fixed revision)")
    ap.add_argument("--c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512,
                    help="MiniLM truncation length; the full prompt exceeds the 256 used for user turns")
    args = ap.parse_args()

    items = load_items(args.items)
    actions = load_actions(args.actions)
    ids = [it.item_id for it in items]
    y = np.asarray([it.intent for it in items], dtype=int)
    groups = np.asarray([p25._holdout.verb_group(it.user_turn) for it in items])
    schemes, excluded = p25.make_splits(y, groups, ids)
    # Only the paired scheme is of interest; keep the LOPO diagnostic out to avoid re-reporting it.
    schemes = {"paired_construction": schemes["paired_construction"]}
    if args.reference_splits.exists():
        ref = json.loads(args.reference_splits.read_text())
        ref_ids = [f["fold_id"] for f in ref["schemes"]["paired_construction"]]
        if ref_ids != [f["fold_id"] for f in schemes["paired_construction"]]:
            raise ValueError("Paired folds differ from the archived script-25 splits")
        ref_test = [f["test_item_ids"] for f in ref["schemes"]["paired_construction"]]
        if ref_test != [f["test_item_ids"] for f in schemes["paired_construction"]]:
            raise ValueError("Paired fold memberships differ from the archived splits")
        splits_match = True
    else:
        splits_match = None

    texts = np.asarray([full_prompt_text(it, actions[it.action]) for it in items])
    user_only = np.asarray([it.user_turn for it in items])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.out_dir / "scores.jsonl"
    results = {}
    with score_path.open("w", encoding="utf-8") as stream:
        results["tfidf_user_turn_recheck"] = p25.evaluate(
            "tfidf_user_turn_recheck", "tfidf", None, user_only, y, ids, schemes,
            args.c_values, args.seed, stream)
        results["tfidf_full_prompt"] = p25.evaluate(
            "tfidf_full_prompt", "tfidf", None, texts, y, ids, schemes,
            args.c_values, args.seed, stream)
        minilm_meta = None
        if args.model_dir is not None:
            # Like-for-like sanity check against the archived user-turn MiniLM result (max_length 256).
            X_user, _ = minilm_embeddings(args.model_dir, user_only.tolist(), 256,
                                          args.batch_size, args.threads, args.seed)
            results["minilm_user_turn_recheck"] = p25.evaluate(
                "minilm_user_turn_recheck", "minilm", X_user, None, y, ids, schemes,
                args.c_values, args.seed, stream)
            X, n_trunc = minilm_embeddings(args.model_dir, texts.tolist(), args.max_length,
                                           args.batch_size, args.threads, args.seed)
            np.savez_compressed(args.out_dir / "minilm_full_prompt.npz", embeddings=X,
                                item_ids=np.asarray(ids))
            minilm_meta = {"model": p25.MINILM_MODEL, "revision": p25.MINILM_REVISION,
                           "max_length": args.max_length, "n_truncated_inputs": int(n_trunc),
                           "pooling": "attention-mask mean, L2 normalized"}
            results["minilm_full_prompt"] = p25.evaluate(
                "minilm_full_prompt", "minilm", X, None, y, ids, schemes,
                args.c_values, args.seed, stream)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_definition": "AGENT_SYSTEM + '\\n\\n' + _tools_as_text(action) + '\\n\\n' + user_turn "
                            "(neutral rendering text before chat template)",
        "data_sha256": hashlib.sha256(args.items.read_bytes()).hexdigest(),
        "ordered_full_text_sha256": p25.json_hash(texts.tolist()),
        "n_items": len(items), "n_paired_folds": len(schemes["paired_construction"]),
        "splits_match_archived_script25": splits_match,
        "n_excluded_pairs": len(excluded),
        "c_values": args.c_values, "seed": args.seed,
        "minilm": minilm_meta,
        "results": results,
        "limitations": [
            "Matches input context (system instruction + schema + user turn), not encoder capacity, "
            "pretraining objective, or exact chat-template tokens.",
            "The schema block is constant within a semantic seed, so it carries seed identity, not label information.",
            "Folds overlap and are not independent; macro AUROC is descriptive.",
        ],
    }
    p25.write_json(args.out_dir / "metrics.json", summary)
    for name, res in results.items():
        for c_key, per in res.items():
            s = per["paired_construction"]["summary"]
            print(f"{name:26s} {c_key:7s} macro={s['macro_auroc']:.4f} "
                  f"min={s['min_fold_auroc']:.4f} n_folds={s['n_dual_class_test_folds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
