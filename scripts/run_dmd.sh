#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m llm_spectral_dynamics.run_experiment \
  --config configs/default.yaml \
  --sites "resid_post" \
  --num-sequences "${LLM_SD_NUM_SEQUENCES:-256}" \
  --seq-len "${LLM_SD_SEQ_LEN:-256}" \
  --batch-size "${LLM_SD_BATCH_SIZE:-2}"

