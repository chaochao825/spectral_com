#!/usr/bin/env bash
set -euo pipefail

ROOT="${LLM_SC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${LLM_SC_PYTHON:-python}"
CONFIG="${LLM_SC_CONFIG:-configs/structured_qwen25_1p5b.yaml}"
OUT="${LLM_SC_OUTPUT_DIR:-results/structured_qwen25_1p5b_formal}"
STATE_DIR="$OUT/.state"
LOG_DIR="$OUT/logs"

cd "$ROOT"
mkdir -p "$STATE_DIR" "$LOG_DIR"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export LLM_SC_LOCAL_FILES_ONLY=1
export LLM_SC_SVD_FAIL_FAST=1
export LLM_SC_DATA_OFFLINE=1
export LLM_SC_ZERO_SHOT_OFFLINE=1
export LLM_SC_ZERO_SHOT_STRICT=1
export LLM_SC_DATASET_BACKUP_ROOT="${LLM_SC_DATASET_BACKUP_ROOT:-$HOME/dataset_backup}"

COMMON=(--config "$CONFIG" --output-dir "$OUT")
COMMON+=(--local-files-only)
DATASET_BACKUP_ROOT="${LLM_SC_DATASET_BACKUP_ROOT:-$HOME/dataset_backup}"
PHASES="${LLM_SC_PHASES:-phase1 phase2 phase3 phase4 phase5 figures}"

phase_enabled() {
  [[ " $PHASES " == *" $1 "* ]]
}

compute_fingerprint() {
  local file_digest environment_digest
  file_digest="$(
    {
      printf '%s\0' "$CONFIG" "$PHASES" figures/plot_structured_qwen25.py scripts/run_structured_qwen25_formal.sh scripts/launch_structured_qwen25_formal.sh
      find src/llm_spectral_dynamics -type f -name '*.py' -print0 | sort -z
      if [[ -d "$DATASET_BACKUP_ROOT" ]]; then
        find "$DATASET_BACKUP_ROOT" -maxdepth 3 -type f -print0 | sort -z
      fi
    } | xargs -0 sha256sum | sha256sum | awk '{print $1}'
  )"
  environment_digest="$(
    "$PYTHON" -c 'import importlib.metadata as m, sys; print(sys.executable); print(sys.version); print("\n".join(sorted(str(d.metadata.get("Name")) + "==" + d.version for d in m.distributions() if d.metadata.get("Name"))))' |
      sha256sum | awk '{print $1}'
  )"
  printf '%s\n%s\n' "$file_digest" "$environment_digest" | sha256sum | awk '{print $1}'
}

RUN_FINGERPRINT="$(compute_fingerprint)"

assert_fingerprint_unchanged() {
  local context="$1"
  local current_fingerprint
  current_fingerprint="$(compute_fingerprint)"
  if [[ "$current_fingerprint" != "$RUN_FINGERPRINT" ]]; then
    echo "input fingerprint changed during formal run ($context); refusing mixed-input results" >&2
    exit 5
  fi
}

validate_output() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    return 1
  fi
  if [[ "$path" == *.csv ]]; then
    awk 'NR > 1 && $0 !~ /^[[:space:]]*$/ { found = 1; exit } END { exit found ? 0 : 1 }' "$path"
  fi
}

phase_artifacts() {
  local phase="$1"
  local expected="$2"
  case "$phase" in
    phase1)
      printf '%s\n' \
        "$OUT/phase1/metrics/layer_spectrum_metrics.csv" \
        "$OUT/phase1/metrics/spectra.csv" \
        "$OUT/phase1/metrics/approximation_errors.csv" \
        "$OUT/phase1/metrics/residual_metrics.csv" \
        "$OUT/manifests/phase1_manifest.json"
      ;;
    phase2)
      printf '%s\n' "$OUT/phase2/metrics/activation_reconstruction.csv"
      ;;
    phase3)
      printf '%s\n' \
        "$OUT/phase3/metrics/compression_performance.csv" \
        "$OUT/phase3/metrics/zero_shot.csv" \
        "$OUT/manifests/phase3_replacements.json"
      ;;
    phase4)
      printf '%s\n' \
        "$OUT/phase4/metrics/peft_performance.csv" \
        "$OUT/phase4/metrics/update_spectrum.csv" \
        "$OUT/manifests/phase4_adapters.json"
      ;;
    phase5)
      printf '%s\n' \
        "$OUT/phase5/metrics/rotation_outliers.csv" \
        "$OUT/phase5/metrics/quantization_errors.csv" \
        "$OUT/phase5/metrics/structured_quantization.csv"
      ;;
    figures)
      printf '%s\n' "$expected"
      if phase_enabled phase1; then
        printf '%s\n' \
          "$OUT/figures/spectral_decay_by_module.pdf" \
          "$OUT/figures/structured_approximation_error.pdf" \
          "$OUT/figures/residual_budget_and_concentration.pdf" \
          "$OUT/figures/layer_type_structure_heatmap.pdf"
      fi
      if phase_enabled phase2; then
        printf '%s\n' "$OUT/figures/phase2_weight_vs_activation_error.pdf"
      fi
      if phase_enabled phase3; then
        printf '%s\n' "$OUT/figures/phase3_compression_vs_perplexity.pdf"
      fi
      if phase_enabled phase4; then
        printf '%s\n' "$OUT/figures/phase4_peft_budget_and_rank.pdf"
      fi
      if phase_enabled phase5; then
        printf '%s\n' "$OUT/figures/phase5_rotation_quantization.pdf"
      fi
      ;;
    *)
      printf '%s\n' "$expected"
      ;;
  esac
}

validate_phase_output() {
  local phase="$1"
  local expected="$2"
  local artifact
  while IFS= read -r artifact; do
    validate_output "$artifact" || return 1
  done < <(phase_artifacts "$phase" "$expected")
}

phase_output_checksum() {
  local phase="$1"
  local expected="$2"
  local artifact
  while IFS= read -r artifact; do
    sha256sum "$artifact"
  done < <(phase_artifacts "$phase" "$expected") | sha256sum | awk '{print $1}'
}

marker_matches() {
  local marker="$1"
  local phase="$2"
  local output="$3"
  local marker_fingerprint marker_checksum output_checksum
  read -r marker_fingerprint marker_checksum <"$marker" || return 1
  [[ "$marker_fingerprint" == "$RUN_FINGERPRINT" ]] || return 1
  validate_phase_output "$phase" "$output" || return 1
  output_checksum="$(phase_output_checksum "$phase" "$output")"
  [[ "$marker_checksum" == "$output_checksum" ]]
}

run_phase() {
  local phase="$1"
  local expected="$2"
  shift 2
  local marker="$STATE_DIR/$phase.done"
  local log="$LOG_DIR/$phase.log"
  assert_fingerprint_unchanged "$phase before"
  if [[ -f "$marker" ]]; then
    if marker_matches "$marker" "$phase" "$OUT/$expected"; then
      echo "skip $phase: matching marker and validated output exist"
      return
    fi
    echo "refusing stale or invalid marker for $phase; use a new output directory" >&2
    exit 3
  fi
  echo "start $phase $(date --iso-8601=seconds)"
  "$@" 2>&1 | tee -a "$log"
  assert_fingerprint_unchanged "$phase after"
  if ! validate_phase_output "$phase" "$OUT/$expected"; then
    echo "missing or empty expected output for $phase: $OUT/$expected" >&2
    exit 4
  fi
  marker_tmp="$STATE_DIR/$phase.done.tmp.$$"
  output_checksum="$(phase_output_checksum "$phase" "$OUT/$expected")"
  printf '%s %s\n' "$RUN_FINGERPRINT" "$output_checksum" >"$marker_tmp"
  mv "$marker_tmp" "$marker"
  echo "done $phase $(date --iso-8601=seconds)"
}

phase_enabled phase1 && run_phase phase1 phase1/metrics/layer_spectrum_metrics.csv "$PYTHON" -m llm_spectral_dynamics.structured.phase1 "${COMMON[@]}"
phase_enabled phase2 && run_phase phase2 phase2/metrics/activation_reconstruction.csv "$PYTHON" -m llm_spectral_dynamics.structured.phase2 "${COMMON[@]}"
phase_enabled phase3 && run_phase phase3 phase3/metrics/compression_performance.csv "$PYTHON" -m llm_spectral_dynamics.structured.phase3 "${COMMON[@]}"
phase_enabled phase4 && run_phase phase4 phase4/metrics/peft_performance.csv "$PYTHON" -m llm_spectral_dynamics.structured.phase4 "${COMMON[@]}"
phase_enabled phase5 && run_phase phase5 phase5/metrics/structured_quantization.csv "$PYTHON" -m llm_spectral_dynamics.structured.phase5 "${COMMON[@]}"
phase_enabled figures && run_phase figures report.md "$PYTHON" figures/plot_structured_qwen25.py --result-dir "$OUT"
