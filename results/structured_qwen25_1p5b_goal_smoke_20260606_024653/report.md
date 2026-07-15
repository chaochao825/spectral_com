# Structured Qwen2.5-1.5B Compression Report

- Baseline perplexity: 7.649.
- Best compressed PPL row: stage=down_proj, ratio=4.0, residual=0.0, ppl=9.875.
- Completed replacement stages: down_proj.
- Mean reported zero-shot accuracy: 0.4444.

## Phase 1
- Best weight-error structure by module type: down_proj: low_rank@4.0x.

## Phase 2
- Best activation reconstruction row: module=down_proj, method=low_rank, residual=sparse, activation_error=0.6874.

## Phase 4
- Adapter methods covered: bca, fourierft, lora, mora, structured, structured_lora.
- Best adapter row: method=structured, rank=n/a, params=53760, budget_per_module=65536, ppl=6.715.

## Phase 5
- Best input-channel outlier reduction row: rotation=hadamard, before=1.237, after=1.079, norm_change=7.121e-05.
- Lowest direct quantization-error row: rotation=hadamard, bits=4, error=0.1657.

Detailed CSV artifacts are under `phase1/metrics`, `phase2/metrics`, `phase3/metrics`, `phase4/metrics`, and `phase5/metrics`; figures are under `figures/`.
