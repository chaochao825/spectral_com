#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

ROOT="${LLM_SC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="$1"

cd "$ROOT"
mkdir -p "$OUT/logs" "$OUT/.state"
exec 9>"$OUT/.state/run.lock"
if ! flock -n 9; then
  echo "another formal run owns $OUT" >&2
  exit 5
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLM_SC_OUTPUT_DIR="$OUT"
export LLM_SC_LOCAL_FILES_ONLY="${LLM_SC_LOCAL_FILES_ONLY:-1}"
export LLM_SC_SVD_FAIL_FAST=1
export LLM_SC_DATASET_BACKUP_ROOT="${LLM_SC_DATASET_BACKUP_ROOT:-$HOME/dataset_backup}"
export LLM_SC_DATA_OFFLINE=1
export LLM_SC_ZERO_SHOT_OFFLINE=1

bash scripts/run_structured_qwen25_formal.sh >>"$OUT/logs/formal.log" 2>&1
code=$?
echo "$code" >"$OUT/.state/exit_code"
exit "$code"
