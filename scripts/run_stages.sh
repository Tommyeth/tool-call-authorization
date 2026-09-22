#!/usr/bin/env bash
# 只跑指定阶段，复用已有产物。用于修了某一环之后避免整轮重跑。
#   用法: setsid nohup bash scripts/run_stages.sh qwen-7b forward probe typing </dev/null &>/dev/null &
#
# 阶段名: forward | elicit | rollout | probe | typing

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false

MODEL=${1:-qwen-7b}; shift
STAGES=("$@")
ITEMS=${ITEMS:-data/pairs/pilot_v1.jsonl}
RUN_DIR=${RUN_DIR:-runs/${MODEL}}
LOG="${RUN_DIR}.log"
mkdir -p "$RUN_DIR"
exec >>"$LOG" 2>&1

echo "########## STAGES [${STAGES[*]}] ${MODEL} $(date -u +%F' '%T) ##########"

for s in "${STAGES[@]}"; do
  echo "=== STAGE ${s} start $(date -u +%T) ==="
  case "$s" in
    forward) python scripts/01_forward.py --model "$MODEL" --items "$ITEMS" --run-dir "$RUN_DIR" ;;
    elicit)  python scripts/02_elicit.py  --model "$MODEL" --items "$ITEMS" --run-dir "$RUN_DIR" ;;
    rollout) python scripts/03_rollout.py --model "$MODEL" --items "$ITEMS" --run-dir "$RUN_DIR" ;;
    probe)   python scripts/04_probe.py          --run-dir "$RUN_DIR" --items "$ITEMS" ;;
    typing)  python scripts/05_failure_typing.py --run-dir "$RUN_DIR" --items "$ITEMS" ;;
    *) echo "unknown stage: $s"; exit 1 ;;
  esac
  code=$?
  [ $code -ne 0 ] && { echo "=== FAILED stage=${s} exit=${code} $(date -u +%T) ==="; exit 1; }
  echo "=== STAGE ${s} done $(date -u +%T) ==="
done

echo "########## STAGES DONE ${MODEL} $(date -u +%T) ##########"
