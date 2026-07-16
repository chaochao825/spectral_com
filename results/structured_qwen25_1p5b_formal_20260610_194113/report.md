# Structured Qwen2.5-1.5B Compression Report

- Baseline perplexity: 13.85.
- Best compressed PPL row: stage=down_proj, ratio=2.0, residual=0.04, ppl=1.061e+04.
- Completed replacement stages: down_proj, o_proj, qkv_proj, up_gate_proj.
- Mean reported zero-shot accuracy: 0.3651.

## Phase 1
- Best weight-error structure by module type: down_proj: low_rank@2.0x; gate_proj: low_rank@2.0x; k_proj: low_rank@2.0x; o_proj: low_rank@2.0x; q_proj: low_rank@2.0x; up_proj: low_rank@2.0x; v_proj: low_rank@2.0x.

## Phase 2
- Best activation reconstruction row: module=k_proj, method=low_rank, residual=sparse, activation_error=0.007756.

## Phase 4
- Adapter methods covered: bca, fourierft, lora, mora, structured, structured_lora.
- Best adapter row: method=structured, rank=n/a, params=322560, budget_per_module=131072, ppl=5.763.

## Phase 5
- Best input-channel outlier reduction row: rotation=hadamard, before=1.237, after=1.079, norm_change=7.121e-05.
- Lowest direct quantization-error row: rotation=hadamard, bits=4, error=0.1649.

Detailed CSV artifacts are under `phase1/metrics`, `phase2/metrics`, `phase3/metrics`, `phase4/metrics`, and `phase5/metrics`; figures are under `figures/`.
