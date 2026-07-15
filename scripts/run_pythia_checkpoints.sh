#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

for REV in step0 step1000 step10000 step143000; do
  python -m llm_spectral_dynamics.run_experiment \
    --config configs/default.yaml \
    --models "EleutherAI/pythia-70m" \
    --variants "pretrained" \
    --revision "${REV}" \
    --sites "resid_post,attn_out,mlp_out" \
    --output-dir "results/pythia_checkpoints/${REV}" \
    --num-sequences "${LLM_SD_NUM_SEQUENCES:-512}" \
    --seq-len "${LLM_SD_SEQ_LEN:-256}"
done
