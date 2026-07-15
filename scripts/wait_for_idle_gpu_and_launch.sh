#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_DIR CONFIG [PHASES]" >&2
  exit 2
fi

ROOT="${LLM_SC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$1"
CONFIG="$2"
PHASES="${3:-phase1 phase2 phase3 figures}"
GPU_INDEX="${CUDA_VISIBLE_DEVICES:-0}"
POLL_SECONDS="${LLM_SC_IDLE_POLL_SECONDS:-60}"
REQUIRED_SAMPLES="${LLM_SC_IDLE_REQUIRED_SAMPLES:-3}"
MAX_USED_MIB="${LLM_SC_IDLE_MAX_USED_MIB:-4096}"
MAX_UTIL="${LLM_SC_IDLE_MAX_UTIL:-10}"

cd "$ROOT"
mkdir -p "$OUT/.state" "$OUT/logs"
exec 8>"$OUT/.state/wait.lock"
if ! flock -n 8; then
  echo "another idle waiter owns $OUT" >&2
  exit 5
fi

idle_samples=0
while true; do
  IFS=, read -r used util < <(
    nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits |
      tr -d ' '
  )
  printf '%s gpu=%s used_mib=%s util=%s idle_samples=%s/%s\n' \
    "$(date --iso-8601=seconds)" "$GPU_INDEX" "$used" "$util" "$idle_samples" "$REQUIRED_SAMPLES" |
    tee -a "$OUT/logs/wait.log"
  if (( used <= MAX_USED_MIB && util <= MAX_UTIL )); then
    idle_samples=$((idle_samples + 1))
  else
    idle_samples=0
  fi
  if (( idle_samples >= REQUIRED_SAMPLES )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

export LLM_SC_CONFIG="$CONFIG"
export LLM_SC_PHASES="$PHASES"
exec bash scripts/launch_structured_qwen25_formal.sh "$OUT"
