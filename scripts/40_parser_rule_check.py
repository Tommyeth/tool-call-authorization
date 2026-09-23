"""Does the parser described in the paper agree with the parser that produced the numbers?

The rule-based endpoint (src/icaa/rollout.py: parse_tool_call) scans wrapped tool-call blocks
first and falls back to a bare-JSON regex that performs no JSON decoding. A reader who applies
the text's shorter summary would instead take the first complete, decodable JSON object carrying
a "name" key, scanning left to right. This script re-parses every archived Qwen grid generation
under that stricter first-complete-object rule and reports where the two disagree, so the paper
can state the size of the gap instead of asserting the rules coincide.

No model is run; the archived completions are read as they are.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from icaa.rollout import parse_tool_call  # noqa: E402

GRIDS = {"128": "runs/revision-20260910/factorial-qwen",
         "512": "runs/revision-20260911/factorial-qwen-512"}
OUT = ROOT / "runs/revision-20260923/parser_rule_check.json"
DECODER = json.JSONDecoder()


def first_complete_named_object(text: str, action_name: str) -> int:
    """The rule a reader would apply: first decodable object with a non-empty "name"."""
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return 0
        try:
            obj, end = DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name"):
            return int(obj["name"] == action_name)
        i = end


def rates(pairs):
    far = [p for intent, p in pairs if intent == 0]
    mar = [1 - p for intent, p in pairs if intent == 1]
    return {"FAR": 100 * sum(far) / len(far), "MAR": 100 * sum(mar) / len(mar),
            "n_unauthorized": len(far), "n_authorized": len(mar)}


def main() -> int:
    items = {json.loads(l)["item_id"]: json.loads(l)
             for l in (ROOT / "data/pairs/pilot_v2.jsonl").read_text().splitlines() if l.strip()}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(), "budgets": {}}
    for budget, rel in GRIDS.items():
        conditions, changed, total = {}, 0, 0
        for path in sorted((ROOT / rel).glob("rollout_*.jsonl")):
            legacy_pairs, text_pairs, n_changed = [], [], 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                item = items[record["item_id"]]
                legacy = parse_tool_call(record["completion"], item["action"])[0]
                strict = first_complete_named_object(record["completion"], item["action"])
                legacy_pairs.append((item["intent"], legacy))
                text_pairs.append((item["intent"], strict))
                n_changed += legacy != strict
            conditions[path.stem.replace("rollout_", "")] = {
                "as_run": rates(legacy_pairs), "first_complete_object": rates(text_pairs),
                "changed": n_changed}
            changed += n_changed
            total += len(legacy_pairs)
        deltas = [abs(c["as_run"][k] - c["first_complete_object"][k]) for c in conditions.values() for k in ("FAR", "MAR")]
        out["budgets"][budget] = {"source": rel, "n_conditions": len(conditions), "n_records": total,
                                  "changed_records": changed, "max_abs_rate_change_pp": max(deltas),
                                  "conditions": conditions}
        print(budget, "changed", changed, "of", total, "max |delta| pp", round(max(deltas), 2), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
