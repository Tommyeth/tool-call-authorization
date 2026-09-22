#!/usr/bin/env bash
# 单模型全流程。所有输出（含 stderr）进 runs/{model}.log，带阶段标记与心跳，
# 便于从外部 tail 监控进度与报错。
#   用法: nohup bash scripts/run_all.sh qwen-7b >/dev/null 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

MODEL=${1:-qwen-7b}
ITEMS=${2:-data/pairs/pilot_v1.jsonl}
RUN_DIR="runs/${MODEL}"
LOG="runs/${MODEL}.log"
mkdir -p "$RUN_DIR"
exec >>"$LOG" 2>&1

STAGE="init"
finish() {
  local code=$?
  [ -n "${HB_PID:-}" ] && kill "$HB_PID" 2>/dev/null
  if [ $code -ne 0 ]; then
    echo "=== FAILED stage=${STAGE} exit=${code} $(date -u +%T) ==="
  fi
  exit $code
}
trap finish EXIT

# 心跳：每 10 分钟报一次显存与最近一条 tqdm 进度，避免长阶段完全静默
( while true; do
    sleep 600
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    prog=$(tail -c 2000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -E "it/s|s/it" | tail -1)
    echo "[hb $(date -u +%H:%M:%S)] stage=${STAGE} gpu=${gpu} ${prog}"
  done ) &
HB_PID=$!

run() {
  STAGE="$1"; shift
  echo "=== STAGE ${STAGE} start $(date -u +%T) ==="
  "$@" || { echo "=== FAILED stage=${STAGE} $(date -u +%T) ==="; exit 1; }
  echo "=== STAGE ${STAGE} done $(date -u +%T) ==="
}

echo "########## RUN ${MODEL} $(date -u +%F' '%T) ##########"

run download python - "$MODEL" <<'PY'
import sys, yaml
from huggingface_hub import snapshot_download
cfg = yaml.safe_load(open("configs/models.yaml"))
hf_id = cfg["models"][sys.argv[1]]["hf_id"]
print(f"downloading {hf_id}", flush=True)
p = snapshot_download(hf_id, ignore_patterns=["*.pth", "*.onnx", "original/*"])
print("cached at", p, flush=True)
PY

run forward  python scripts/01_forward.py --model "$MODEL" --items "$ITEMS"
run elicit   python scripts/02_elicit.py  --model "$MODEL" --items "$ITEMS"
run rollout  python scripts/03_rollout.py --model "$MODEL" --items "$ITEMS"
run probe    python scripts/04_probe.py          --run-dir "$RUN_DIR" --items "$ITEMS"
run typing   python scripts/05_failure_typing.py --run-dir "$RUN_DIR" --items "$ITEMS"

echo "########## ALL DONE ${MODEL} $(date -u +%F' '%T) ##########"
