#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LLM_SD_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${LLM_SD_PYTHON:-python}"
OUT_ROOT="${LLM_SD_OUTPUT_ROOT:-$ROOT/results/large_435}"
RUNTIME_ROOT="${LLM_SD_RUNTIME_ROOT:-$ROOT/trash/runtime}"
QWEN15_MODEL="${LLM_SD_QWEN15_MODEL:-/data6/user20111239/Qwen1.5-MoE-A2.7B}"
QWEN57_MODEL="${LLM_SD_QWEN57_MODEL:-/data6/user20111239/Qwen2-57B-A14B-Instruct}"
LLAMA70_MODEL="${LLM_SD_LLAMA70_MODEL:-/data6/user24111736/meta-llama/Llama-2-70b-chat-hf}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/hf_home}"
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT/tmp}"
mkdir -p "$OUT_ROOT" "$TMPDIR" "$HF_HOME"

BASE_ARGS=(
  --config configs/default.yaml
  --variants pretrained
  --sites "resid_post,attn_out,mlp_out"
  --sample-limit "${LLM_SD_SAMPLE_LIMIT:-2048}"
  --bootstrap-samples "${LLM_SD_BOOTSTRAP_SAMPLES:-32}"
  --dynamic-max-sequences "${LLM_SD_DYNAMIC_MAX_SEQUENCES:-16}"
  --dynamic-pca-rank "${LLM_SD_DYNAMIC_PCA_RANK:-32}"
  --local-files-only
  --low-cpu-mem-usage
)

if [[ "${LLM_SD_FRESH_OUTPUT:-0}" == "1" ]]; then
  BASE_ARGS+=(--fresh-output)
fi

(
  export CUDA_VISIBLE_DEVICES="${LLM_SD_QWEN15_CUDA_VISIBLE_DEVICES:-0}"
  "$PYTHON" -m llm_spectral_dynamics.run_experiment \
    "${BASE_ARGS[@]}" \
    --batch-size "${LLM_SD_QWEN15_BATCH_SIZE:-4}" \
    --torch-dtype bfloat16 \
    --models "$QWEN15_MODEL" \
    --layers 0,12,23 \
    --num-sequences "${LLM_SD_QWEN15_NUM_SEQUENCES:-64}" \
    --seq-len "${LLM_SD_QWEN15_SEQ_LEN:-128}" \
    --output-dir "$OUT_ROOT/qwen15_moe"
)

"$PYTHON" -m llm_spectral_dynamics.run_experiment \
  "${BASE_ARGS[@]}" \
  --batch-size 1 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --models "$QWEN57_MODEL" \
  --layers 0,14,27 \
  --num-sequences "${LLM_SD_QWEN57_NUM_SEQUENCES:-16}" \
  --seq-len "${LLM_SD_QWEN57_SEQ_LEN:-128}" \
  --output-dir "$OUT_ROOT/qwen57_a14b"

LLAMA_OUT="$OUT_ROOT/llama70_smoke"
mkdir -p "$LLAMA_OUT"
MIN_FREE="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {m=$1} $1<m {m=$1} END {print m+0}')"
if [[ "$MIN_FREE" -ge "${LLM_SD_LLAMA_MIN_FREE_MB:-51200}" ]]; then
  if ! "$PYTHON" -m llm_spectral_dynamics.run_experiment \
      "${BASE_ARGS[@]}" \
      --batch-size 1 \
      --device-map auto \
      --torch-dtype float16 \
      --models "$LLAMA70_MODEL" \
      --layers 0,40,79 \
      --num-sequences "${LLM_SD_LLAMA_NUM_SEQUENCES:-4}" \
      --seq-len "${LLM_SD_LLAMA_SEQ_LEN:-64}" \
      --output-dir "$LLAMA_OUT"; then
    {
      echo "# Llama-2-70B Smoke Failed"
      echo
      echo "The gated Llama smoke was attempted because minimum free GPU memory was ${MIN_FREE} MiB."
      echo "The command failed; this does not fail the Qwen large-model validation."
    } > "$LLAMA_OUT/FAILED.md"
  fi
else
  {
    echo "# Llama-2-70B Smoke Skipped"
    echo
    echo "Minimum free GPU memory was ${MIN_FREE} MiB, below ${LLM_SD_LLAMA_MIN_FREE_MB:-51200} MiB."
    echo "This skip does not fail the Qwen large-model validation."
  } > "$LLAMA_OUT/SKIPPED.md"
fi

"$PYTHON" scripts/summarize_results.py "$OUT_ROOT/qwen15_moe"
"$PYTHON" scripts/summarize_results.py "$OUT_ROOT/qwen57_a14b"
if [[ -f "$LLAMA_OUT/SKIPPED.md" ]]; then
  cat "$LLAMA_OUT/SKIPPED.md"
elif [[ -f "$LLAMA_OUT/FAILED.md" ]]; then
  cat "$LLAMA_OUT/FAILED.md"
else
  "$PYTHON" scripts/summarize_results.py "$LLAMA_OUT"
fi
