# Spectral Dynamics and Structured Residuals

This repository studies two connected questions in Transformer models:

1. How activation covariance, effective rank, spectral tails, and token-time dynamics change across depth and sublayers.
2. Whether weight and quantization residuals contain sparse, low-rank, block-circulant, or Monarch-like structure that can be exploited under a matched memory budget.

The repository contains the current implementation, tests, run scripts, notebooks, figures, and all archived experiment results. Model weights and datasets are intentionally excluded.

## Research Tracks

### Activation spectral dynamics

- Streamed, centered covariance with float64 Welford/Chan updates.
- Eigenspectra at `resid_post`, `attn_out`, and `mlp_out` sites.
- Effective rank, participation ratio, spectral entropy, explained variance, condition number, anisotropy, outlier score, and power-law fits.
- Pretrained versus random-init controls, shared-axis overlays, and metric delta tables.
- Token-lag autocorrelation, PCA dynamics, DMD, KV-cache spectra, and PCA interventions.
- Local Hugging Face model paths, offline loading, mixed precision, and `device_map=auto` for multi-GPU models.

### Structured residual compression

- Weight spectra and equal-budget low-rank, block-circulant, and Monarch-like approximations.
- Activation reconstruction, perplexity, and zero-shot evaluation.
- Quantization plus residual decomposition:

  ```text
  W ~= Q(W) + S_res + L_res
  ```

- Sparse/low-rank residual stacking, OASR candidates, Hessian-weighted orthogonality, additivity tests, and selector diagnostics.
- Structured and low-rank adapters, rotations, and quantization experiments.

See the [Chinese current-methods and results overview](docs/current_methods_and_results_zh.md) for a concise project-level judgment, [current methods and results](docs/methods_and_results.md) for the English technical summary, and [the 2026-07-16 remote merge and research audit](docs/remote_merge_and_research_audit_20260716.md) for provenance and novelty qualification. The current method is specified in [the 2026-07-17 two-stage interaction-aware update](docs/interaction_aware_two_stage_update_20260717.md), with [an official-baseline execution contract](docs/official_baseline_execution_contract_20260717.md) for SLiM/OBR/QERA/EoRA. The vision extension is covered by [vision residual rank and locality](docs/vision_residual_rank_locality.md) and the newer [PTQ, attention locality, and cache experiment plan](docs/vision_ptq_attention_locality_plan_20260717.md). Verified server-35 model paths are recorded in [the model registry](configs/model_paths_435.yaml).

### Vision residual rank and attention locality

- A high `resid_post` effective rank does not prove that the attention update remains high-rank: the identity residual path can preserve an existing subspace while `attn_out` becomes low-rank or redundant.
- Global attention can contract non-shared patch modes quickly when its mixing operator has a large spectral gap. Fixed local windows slow global mixing but can first collapse diversity within each window, so global rank and within-window rank must be reported separately.
- The compression-relevant signal combines update rank, novelty outside the current residual subspace, spatial support, output sensitivity, cache reuse, and runtime traffic. Low-rank but high-novelty updates may encode useful specialization; low-rank and low-novelty updates are stronger compression candidates.
- The proposed visual experiment is `PTQ x attention route x cache codec`, matched jointly on natural checkpoint bytes, runtime traffic, and peak VRAM, then selected by independent validation quality and measured latency.

This is currently a source-grounded analysis and execution contract, not a completed visual-model benchmark.

## Current Evidence

| Experiment | Scope | Current interpretation |
|---|---|---|
| `results/mvp_real` | GPT-2 and Pythia-70M, pretrained/random, real text | Pretrained/random differences are visible in matched metrics but are subtle in raw log-log plots; attention outputs have lower effective rank than FFN/residual outputs in this sample. |
| `results/large_435_download_20260604_1026` | Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B, Llama-2-70B | Multi-GPU, local-path spectral collection is feasible. Qwen1.5 is a reduced feasibility run; Qwen57 and Llama70 cover the planned layer/site smoke settings. |
| `results/structured_qwen25_1p5b_goal_smoke_20260606_024653` | Qwen2.5-1.5B structured compression | Low-rank was the best tested weight approximation for `down_proj`; structured adapters and rotation/quantization signals remain preliminary. |
| `results/structured_qwen25_1p5b_formal_20260610_194113` | Complete Qwen2.5-1.5B five-phase run | Direct structured replacements failed badly: baseline PPL was `13.85` and the best compressed row was about `1.061e4`. The adapter result is in-sample because training and evaluation reuse the validation prefix. |
| `results/compression_orthogonality_mvp_20260623_v7` | Orthogonality/additivity MVP | Provides Hessian-cosine, additivity, loss-landscape, and goal-audit artifacts; it is a methodology check, not a large-model conclusion. |
| `results/compare_7b_dam_residual_stack_20260707` | Qwen2-7B layer subsets and Pythia controls | `Q+S+L` produced a positive signal on the six-module Qwen2-7B attention+MLP subset, but the evaluation is too small for a general or SOTA claim. |
| `results/two_stage_heterogeneous_pythia70m_natural_match_smoke_20260717` | Pythia-70M, one MLP tensor, independent train/validation/test | Exact-natural QSL/no-joint matching works. Both selected the same Q+L endpoint at 627,712 natural bytes, so the joint-value claim correctly remains false. |
| `results/large_model_interaction_aware_v4_qwen3b_sentinel_ea77d12_20260717` | Qwen2.5-3B, layer-0 `gate_proj`, independent train/validation/test | The heterogeneous two-stage allocator completed on a 3B model. QSL and exact-natural no-joint both selected the same W4/group-128 RTN plus rank-72 FP16 low-rank endpoint at 13,506,496 natural bytes and test NLL 2.550458, so the joint-value claim remains false. |

The complete result directory map and qualification notes are in [results/README.md](results/README.md). Historical reports retain only the paths and run metadata that were recorded at execution time; several early runs do not contain a complete package lock, model/data revision, or command manifest. The environment used for the current publication checks is recorded separately in [the 210 validation manifest](environments/validation_210_20260716.yaml) and must not be treated as the original experiment environment.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,research]"
```

The large-model scripts assume local model directories and can run fully offline. They do not download or commit model weights. `scripts/run_large_435.sh` is project-relative and accepts `LLM_SD_QWEN15_MODEL`, `LLM_SD_QWEN57_MODEL`, and `LLM_SD_LLAMA70_MODEL` overrides. Runtime caches default to ignored `trash/runtime`; set `LLM_SD_RUNTIME_ROOT` when a different high-capacity scratch path is preferred.

## Entry Points

Dependency-light synthetic smoke:

```bash
LLM_SD_SMOKE=1 bash scripts/run_mvp.sh
```

Real activation-spectrum MVP:

```bash
bash scripts/run_mvp.sh
```

Server-35 large-model validation:

```bash
bash scripts/run_large_435.sh
```

Structured Qwen2.5 phases:

```bash
bash scripts/run_structured_qwen25.sh
```

Interaction-aware 3B/7B staged validation:

```bash
python scripts/run_large_scale_hessian_suite.py \
  --config configs/large_model_interaction_aware_v4_20260717.json \
  --dry-run
```

Unit tests:

```bash
python scripts/run_unit_tests.py
```

Publication and package checks:

```bash
python scripts/check_publishable_tree.py
python -m pip wheel . --no-deps -w trash/wheel-smoke/dist
```

## Output Contract

Activation runs write metrics, eigenvalue payloads, comparison plots, and a report under their configured output directory. Structured runs write per-phase CSV/JSON artifacts, figures, manifests, and reports. Existing managed outputs are moved to an output-local `trash/` directory only when `--fresh-output` is explicitly used.

## Interpretation Rules

- Synthetic smoke data validates plumbing; it is not evidence about pretrained representations.
- Covariance spectra over pooled tokens, per-image token rank, attention-matrix rank, and weight rank are different objects and must not be compared as if they were interchangeable.
- Pretrained/random comparisons require matched model family, layer, site, data, token count, centering, and estimator.
- Large-model pretrained-only runs establish feasibility and depth/site trends, not pretrained/random causality.
- Small perplexity or zero-shot subsets are screening tests. Final claims require broader models, tasks, seeds, and confidence intervals.
- Absolute perplexities from different run directories are not comparable unless evaluation examples, model/tokenizer revisions, seeds, and scoring settings are matched; use within-run deltas for the archived Qwen2-7B layer-subset comparison.
