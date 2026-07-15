# Method Revision After Split-Data Fair Benchmark

This note updates the original Compression Orthogonality Landscape objective after the stricter split-data benchmark. The important change is conceptual: the current evidence supports Hessian cross-terms as a diagnostic framework, but not yet as a standalone competitive compression recipe.

## Original Objective Restated

The pasted goal defines the contribution as a loss-landscape diagnostic framework, not a new triple-compression pipeline. The paper-strength claim must be:

- whether Hessian cross-terms predict which Q/S/R perturbations are complementary or conflicting;
- whether additivity error follows the same pattern;
- whether order gaps can be explained by Hessian overlap and singular-spectrum changes;
- whether a layer-wise selection rule can improve over fixed-order baselines at the same budget.

This means a pretty landscape or a successful Q+S+R stack is insufficient. The framework must report `rho_H`, additivity error, order gap, and real PPL/accuracy degradation correlations.

## Current Evidence Against The Strong Selector Claim

The strictest artifact is:

- `results/pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3`

It uses disjoint text windows: 32 calibration texts, 64 PPL-evaluation texts, and 64 LoRA-recovery texts. Zero-shot uses ARC-Easy and HellaSwag with 100 validation examples each.

### Same-Budget QSR Result

At selected-layer memory ratio 0.133, the Hessian-guided QSR selector minimizes predicted local Hessian cost but does not minimize held-out PPL degradation:

| Strategy | Selection rule | Predicted Hessian cost | PPL delta | Mean zero-shot delta |
| --- | --- | ---: | ---: | ---: |
| `qsr_rotated_wanda_whitened` | fixed Q->S->R | 21.2457 | +57.64% | -0.045 |
| `rqs_rotated_wanda_whitened` | fixed R->Q->S | 14.3136 | +59.54% | -0.020 |
| `hessian_guided_qsr_budget` | layer-wise min Hessian cost | 12.3022 | +64.10% | -0.025 |
| `slim_like_srq_proxy` | fixed SLiM-like proxy | 13.5985 | +67.45% | -0.035 |

The result is mixed but not competitive enough for the strong claim. Hessian-guided QSR is better than the SLiM-like proxy in this split run, but it is worse than both fixed rotated QSR/RQS recipes at the same memory. The selector therefore cannot be presented as a reliable same-budget QSR improvement.

### SPQ-Prior Result

At matched SPQ layer prior, memory ratio 0.196, and equal tiny LoRA recovery budget, Hessian-guided SPQ improves PPL but worsens mean zero-shot accuracy:

| Strategy | Recovery | PPL delta | Mean zero-shot delta |
| --- | --- | ---: | ---: |
| `spq_like_rsq_no_lora` | none | +12.31% | +0.030 |
| `hessian_guided_spq_no_lora` | none | +11.01% | +0.010 |
| `spq_like_rsq_lora` | LoRA rank4, 5 steps | +13.53% | +0.035 |
| `hessian_guided_spq_lora` | LoRA rank4, 5 steps | +9.74% | +0.015 |

This is the best current positive result, but it is recipe-conditioned and PPL-only. It shows that Hessian guidance can refine a sensible fixed layer prior; it does not show Pareto dominance.

## Why The Current Method Fails

### 1. The Hessian proxy is too local for direct recipe selection

The current `rho_H` and predicted cost use a layer-local Gauss-Newton proxy from activation covariance `X^T X`, not the exact full-model Hessian. In the split-data run:

| Predictor | Target | Spearman rho |
| --- | --- | ---: |
| `abs(rho_H)` | `abs(additivity_error)` | 0.2727 |
| `abs(rho_H)` | PPL degradation | 0.1888 |
| Taylor/cross-term prediction | loss degradation | 0.3287 |
| Frobenius delta sum | loss degradation | 0.8811 |
| trace-only cost | loss degradation | 0.7413 |

This is not enough to use local Hessian cost as the only selector objective. It is useful for detecting catastrophic candidates and interpreting conflicts, but it mis-ranks several non-catastrophic choices.

### 2. The selector optimizes one proxy, but the benchmark is multi-objective

The selector minimizes calibration Hessian cost. The reported benchmark cares about held-out NLL/PPL and zero-shot accuracy. SPQ-guided rows show the mismatch clearly: PPL improves, but zero-shot mean is lower than fixed SPQ-like.

The conclusion is not that Hessian overlap is useless. The conclusion is that a single scalar Hessian cost is not aligned with a Pareto objective unless it is combined with task-aware and activation-reconstruction terms.

### 3. The unconstrained QSR search space is too aggressive

The QSR budget uses 4-bit quantization, keep=0.8 pruning, and rank=0.5 low-rank replacement, giving memory ratio 0.133. In the same run, `r_only_whitened_rank0p5` alone has +53.92% PPL delta. This means the stack already contains a high-risk operation. Changing order cannot fully repair an over-aggressive low-rank component.

### 4. The selector lacks structural priors used by strong recipes

SPQ works because it does not apply every operation everywhere. It uses layer-type specialization: low-rank on attention projections, pruning on MLP layers, global quantization, then recovery. The fair result agrees with that design lesson. Hessian-guided SPQ is better than unconstrained Hessian-guided QSR because the SPQ layer prior restricts the search to more plausible candidates.

### 5. Rotation and outlier handling are under-modeled

In the QSR selector, two larger layers choose plain RTN instead of rotated RTN because plain RTN has lower local predicted Hessian cost. Held-out PPL disagrees. This suggests that the proxy misses rotation's outlier-smoothing and generalization effect, especially in high-impact projections.

## Revised Selector Design

The next selector should not replace fixed recipes with a free-form Hessian search. It should be a constrained, multi-objective selector:

```text
candidate_score(layer, candidate) =
    z(local_taylor_cross_cost)
  + lambda_act * z(activation_reconstruction_error)
  + lambda_trace * z(trace_or_frobenius_sensitivity)
  + lambda_order * z(predicted_order_gap)
  + lambda_outlier * unrotated_quantization_penalty
  + lambda_zs * calibration_zero_shot_proxy
  + memory_penalty(candidate, target_budget)
```

The constraints should be explicit:

- keep SPQ/SLiM-like layer priors as default candidates;
- allow overrides only when Hessian overlap or order-gap evidence crosses a threshold;
- pre-register memory ratios and recovery budgets;
- choose candidates only from calibration data;
- report the Pareto frontier over PPL delta, zero-shot delta, memory ratio, and recovery cost.

## Revised Experimental Matrix

The next fair run should separate four effects that are currently entangled:

| Ablation | Question |
| --- | --- |
| fixed SPQ prior vs no prior | Is the layer-type prior more important than Hessian selection? |
| method choice fixed vs method choice guided | Is the gain from choosing RTN/rotated RTN, Wanda/magnitude, SVD/whitened SVD? |
| order fixed vs order guided | Does Hessian overlap actually improve non-commutative order? |
| no recovery vs equal LoRA recovery | Does the selector survive under the same recovery budget? |

Each row should be evaluated on the same held-out protocol:

- sliding-window WikiText-2 or C4 PPL where available;
- ARC-Easy, HellaSwag, PIQA, and WinoGrande if local data is available;
- signed PPL/NLL delta, clipped degradation, mean and per-task zero-shot delta;
- theoretical selected-layer memory and deployable memory when packed kernels exist.

## Paper Claim Update

Supported now:

- Hessian/cross-term tables and heatmaps provide useful diagnostics beyond visualization.
- Order gaps are real, and singular-spectrum changes partly explain them.
- Strong layer priors matter. Hessian guidance can improve PPL inside an SPQ-like prior under matched memory and recovery budget.

Not supported yet:

- Hessian-guided QSR is a competitive standalone compression method.
- `rho_H` alone is a robust predictor of real PPL or zero-shot degradation.
- A claim that the current guided method is Pareto-dominant over fixed SPQ-like or stronger external baselines.
- The lossless frontier proves deployable or paper-strength lossless stacking. It remains a search diagnostic until split-data, zero-shot, and larger-window validation passes.

The correct paper positioning is therefore: "Compression Orthogonality Landscape" is a diagnostic and constrained selection framework for explaining and improving ensemble compression recipes, not a claim that unconstrained Hessian-guided QSR is already the best compressor.

## Figure

The diagnostic figure is generated by `figures/plot_selector_diagnostic.py` and saved as:

- `figures/selector_failure_diagnostic.png`
- `figures/selector_failure_diagnostic.pdf`

It visualizes the key negative and qualified-positive results: Hessian cost mis-ranks same-budget QSR rows, while SPQ-prior guidance improves PPL without becoming Pareto-dominant.
