#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

COMMON_ARGS=(
  --config configs/default.yaml
  --models "gpt2,EleutherAI/pythia-70m"
  --variants "pretrained,random"
  --sites "resid_post,attn_out,mlp_out"
  --num-sequences "${LLM_SD_NUM_SEQUENCES:-512}"
  --seq-len "${LLM_SD_SEQ_LEN:-256}"
  --batch-size "${LLM_SD_BATCH_SIZE:-2}"
  --output-dir "${LLM_SD_OUTPUT_DIR:-results}"
)

if [[ "${LLM_SD_SMOKE:-0}" == "1" ]]; then
  if [[ "${LLM_SD_FRESH_OUTPUT:-0}" == "1" ]]; then
    COMMON_ARGS+=(--fresh-output)
  fi
  python -m llm_spectral_dynamics.run_experiment \
    "${COMMON_ARGS[@]}" \
    --models "synthetic-gpt2,synthetic-pythia-70m" \
    --num-sequences "${LLM_SD_NUM_SEQUENCES:-24}" \
    --seq-len "${LLM_SD_SEQ_LEN:-32}" \
    --synthetic-smoke
else
  if [[ "${LLM_SD_FRESH_OUTPUT:-0}" == "1" ]]; then
    COMMON_ARGS+=(--fresh-output)
  fi
  python -m llm_spectral_dynamics.run_experiment "${COMMON_ARGS[@]}"
fi
