"""Compare the archived 128-token Qwen run with the 512-token rerun, item by item.

Inputs are the two factorial directories (rollout_<condition>.jsonl per condition,
cluster_stats/stats.json from script 28). Outputs a JSON summary:
  * per-condition FAR/MAR at 128 vs 512 tokens (legacy parser, fixed 240/160 denominators),
  * truncation counts at both budgets,
  * item-level transitions among the 1,033 outputs that hit the 128 cap
    (non-call -> call, call -> non-call, unchanged), split by authorization label,
  * the eight matched contrasts at both budgets (from the two stats.json files),
  * a prefix check: whether the first 128 generated tokens of the 512 run equal the
    archived 128-token generation (greedy decoding on different hardware need not agree).
No tool is dispatched; all outcomes are parsed text proposals.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.schema import load_items  # noqa: E402


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rates(rows, label):
    y = np.asarray([rows[i]["executed"] for i in label]); I = np.asarray([label[i] for i in label])
    return {"FAR": float(y[I == 0].mean()), "MAR": float(1 - y[I == 1].mean()),
            "false_actions": int(y[I == 0].sum()), "missed_actions": int((1 - y[I == 1]).sum()),
            "n_truncated": int(sum(r.get("truncated", False) for r in rows.values()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "runs/revision-20260910/factorial-qwen")
    ap.add_argument("--extended", type=Path, default=ROOT / "runs/revision-20260911/factorial-qwen-512")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/revision-20260911/budget_sensitivity.json")
    args = ap.parse_args()
    items = load_items(ROOT / "data/pairs/pilot_v2.jsonl")
    label = {it.item_id: it.intent for it in items}
    base_prov = json.loads((args.base / "provenance.json").read_text())
    ext_prov = json.loads((args.extended / "provenance.json").read_text())
    conditions = [f"{c['role']}_{c['force']}" for c in base_prov["conditions"]]
    assert conditions == [f"{c['role']}_{c['force']}" for c in ext_prov["conditions"]]
    per_condition, transitions = {}, {"authorized": {}, "unauthorized": {}}
    prefix_equal, prefix_total, changed_outcome_total = 0, 0, 0
    for tag in conditions:
        b = {r["item_id"]: r for r in read_jsonl(args.base / f"rollout_{tag}.jsonl")}
        e = {r["item_id"]: r for r in read_jsonl(args.extended / f"rollout_{tag}.jsonl")}
        assert set(b) == set(e) == set(label), tag
        for iid in b:
            assert b[iid]["prompt_sha256"] == e[iid]["prompt_sha256"], ("prompt mismatch", tag, iid)
        rb, re_ = rates(b, label), rates(e, label)
        trans = {}
        for iid, rec in b.items():
            key = "authorized" if label[iid] == 1 else "unauthorized"
            bt = rec["generated_token_ids"]; et = e[iid]["generated_token_ids"]
            prefix_total += 1; prefix_equal += int(et[:len(bt)] == bt or (len(et) < len(bt) and bt[:len(et)] == et))
            if rec.get("truncated"):
                move = f"{int(rec['executed'])}->{int(e[iid]['executed'])}"
                trans[key] = trans.get(key, {}); trans[key][move] = trans[key].get(move, 0) + 1
            changed_outcome_total += int(rec["executed"] != e[iid]["executed"])
        per_condition[tag] = {"budget_128": rb, "budget_512": re_,
                              "FAR_change_pp": round(100 * (re_["FAR"] - rb["FAR"]), 2),
                              "MAR_change_pp": round(100 * (re_["MAR"] - rb["MAR"]), 2),
                              "transitions_among_128_truncated": trans}
        for key in trans:
            for move, n in trans[key].items():
                transitions[key][move] = transitions[key].get(move, 0) + n
    contrasts = {}
    for name, d in [("budget_128", args.base), ("budget_512", args.extended)]:
        stats_path = d / "cluster_stats" / "stats.json"
        if stats_path.exists():
            s = json.loads(stats_path.read_text())
            contrasts[name] = {c["name"]: {m: {k: c[m][k] for k in ("estimate_percentage_points", "ci_lower_percentage_points", "ci_upper_percentage_points")}
                                           for m in ("FAR", "MAR") if m in c}
                               for c in s.get("contrasts", []) if c["name"].startswith("matched:")}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base": {"dir": str(args.base.relative_to(ROOT)), "max_new_tokens": base_prov["max_new_tokens"], "gpu": base_prov.get("gpu")},
        "extended": {"dir": str(args.extended.relative_to(ROOT)), "max_new_tokens": ext_prov["max_new_tokens"], "gpu": ext_prov.get("gpu"),
                     "torch": ext_prov.get("torch"), "transformers": ext_prov.get("transformers")},
        "n_conditions": len(conditions), "n_records_per_budget": 400 * len(conditions),
        "total_truncated_128": sum(v["budget_128"]["n_truncated"] for v in per_condition.values()),
        "total_truncated_512": sum(v["budget_512"]["n_truncated"] for v in per_condition.values()),
        "total_false_actions": {"128": sum(v["budget_128"]["false_actions"] for v in per_condition.values()),
                                "512": sum(v["budget_512"]["false_actions"] for v in per_condition.values())},
        "total_missed_actions": {"128": sum(v["budget_128"]["missed_actions"] for v in per_condition.values()),
                                 "512": sum(v["budget_512"]["missed_actions"] for v in per_condition.values())},
        "records_with_changed_outcome": changed_outcome_total,
        "greedy_prefix_agreement": {"equal_first_128_tokens": prefix_equal, "total": prefix_total,
                                    "note": "Different hardware (A10 vs RTX 5090) and kernels; disagreement here is numerical, not a prompt difference."},
        "transitions_among_128_truncated_outputs": transitions,
        "per_condition": per_condition, "matched_contrasts": contrasts,
        "interpretation": "Budget sensitivity of the proposal rates; the 128-token archived run remains the primary result. "
                          "Truncated records stay in every denominator at both budgets.",
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["total_truncated_128", "total_truncated_512", "total_false_actions",
                                               "total_missed_actions", "records_with_changed_outcome", "greedy_prefix_agreement",
                                               "transitions_among_128_truncated_outputs"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
