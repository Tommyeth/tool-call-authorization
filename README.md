# Authorization readouts, instruction clauses, and generation budgets in LLM tool calling

Research code for probing whether language models represent *authorization* (a user requesting an
action versus merely discussing it) and how tool-calling behaviour depends on instruction wording,
placement, and generation budget.

This repository contains the core experiment and analysis code only. Data, model outputs, cached
hidden states, figure scripts, and manuscript files are not included.

## Layout

| Path | Contents |
|---|---|
| `src/icaa/` | Item schema, prompt rendering, model loading, forward passes, probing, rollout parsing |
| `scripts/` | Numbered experiment and analysis steps and two shell runners |
| `configs/models.yaml` | Model identifiers and chat-template settings |

Key steps: `24_nested_transfer.py` (audit-to-action transfer with audit-only layer selection),
`25_paired_holdout.py` and `32_matched_input_text_baseline.py` (construction hold-outs and text
baselines), `27_revision_factorial.py` / `28_factorial_revision_stats.py` (instruction grid and
paired seed-bootstrap contrasts; `--max-new-tokens` sets the budget), `33_audit_cache_with_ids.py`
(ID-embedded hidden-state extraction), `35`–`37` (activation intervention with equal-norm controls), `39_revision_controls.py`
(text baselines and shuffled-label control for audit-to-action transfer, zero-shot audit judgment on
construction folds, and steering shift size).

## Setup

```bash
pip install -r requirements.txt
```

Scripts expect the item file at `data/pairs/pilot_v2.jsonl`, action schemas at `data/actions.yaml`,
and write to `runs/`. Gated checkpoints need Hugging Face access configured in your environment
(for example `huggingface-cli login`); no credentials are stored here.

All model tool calls are generated text proposals; no tool is executed.
