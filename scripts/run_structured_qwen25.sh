#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${LLM_SC_PYTHON:-python}"
CONFIG="${LLM_SC_CONFIG:-configs/structured_qwen25_1p5b.yaml}"
OUT="${LLM_SC_OUTPUT_DIR:-results/structured_qwen25_1p5b}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

COMMON=(--config "$CONFIG" --output-dir "$OUT")

if [[ "${LLM_SC_LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  COMMON+=(--local-files-only)
fi

if [[ "${LLM_SC_SMOKE:-0}" == "1" ]]; then
  export LLM_SC_ALLOW_FALLBACK="${LLM_SC_ALLOW_FALLBACK:-1}"
  SMOKE_COMMON=("${COMMON[@]}" --layers "${LLM_SC_LAYERS:-0}" --modules "${LLM_SC_MODULES:-down_proj}" --compression-ratios "${LLM_SC_RATIOS:-4}" --residual-fractions "${LLM_SC_RESIDUALS:-0,0.02}")
  "$PYTHON" -m llm_spectral_dynamics.structured.phase1 "${SMOKE_COMMON[@]}" --max-matrices "${LLM_SC_MAX_MATRICES:-1}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase2 "${SMOKE_COMMON[@]}" --sample-limit "${LLM_SC_SAMPLE_LIMIT:-128}" --calibration-sequences "${LLM_SC_CALIBRATION_SEQUENCES:-2}" --max-modules "${LLM_SC_MAX_MODULES:-1}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase3 "${SMOKE_COMMON[@]}" --eval-limit "${LLM_SC_EVAL_LIMIT:-2}" --zero-shot-limit "${LLM_SC_ZERO_SHOT_LIMIT:-2}" ${LLM_SC_SKIP_ZERO_SHOT:+--skip-zero-shot}
  "$PYTHON" -m llm_spectral_dynamics.structured.phase4 "${COMMON[@]}" \
    --layers "${LLM_SC_LAYERS:-0}" \
    --modules "${LLM_SC_MODULES:-down_proj}" \
    --methods "${LLM_SC_PHASE4_METHODS:-structured,structured_lora,lora,mora,fourierft,bca}" \
    --budgets "${LLM_SC_PHASE4_BUDGETS:-65536}" \
    --task-conditions "${LLM_SC_PHASE4_TASKS:-natural}" \
    --train-steps "${LLM_SC_PHASE4_TRAIN_STEPS:-1}" \
    --eval-limit "${LLM_SC_PHASE4_EVAL_LIMIT:-1}" \
    --max-modules "${LLM_SC_MAX_MODULES:-1}" \
    --max-runs "${LLM_SC_PHASE4_MAX_RUNS:-6}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase5 "${COMMON[@]}" \
    --layers "${LLM_SC_LAYERS:-0}" \
    --modules "${LLM_SC_MODULES:-down_proj}" \
    --rotations "${LLM_SC_PHASE5_ROTATIONS:-none,hadamard,learned_butterfly}" \
    --bit-widths "${LLM_SC_PHASE5_BITS:-4,3,2}" \
    --max-matrices "${LLM_SC_PHASE5_MAX_MATRICES:-1}"
else
  "$PYTHON" -m llm_spectral_dynamics.structured.phase1 "${COMMON[@]}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase2 "${COMMON[@]}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase3 "${COMMON[@]}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase4 "${COMMON[@]}"
  "$PYTHON" -m llm_spectral_dynamics.structured.phase5 "${COMMON[@]}"
fi

"$PYTHON" figures/plot_structured_qwen25.py --result-dir "$OUT"
