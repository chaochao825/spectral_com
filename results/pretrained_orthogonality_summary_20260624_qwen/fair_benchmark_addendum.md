# Fair Fixed-Config Benchmark Addendum

This addendum addresses a methodological issue in the earlier lossless frontier result: selecting the best row from the same short PPL window can make clipped PPL drop look like zero. The benchmark below evaluates fixed, predeclared recipes plus predeclared calibration-only Hessian selectors. No row is selected by final PPL or zero-shot accuracy.

## Protocol

Run artifact:

- `results/pretrained_orthogonality_pythia70m_fair_benchmark_4mods_arc_hella100_20260627`
- Guided-selection update: `results/pretrained_orthogonality_pythia70m_fair_benchmark_guided_4mods_arc_hella100_20260627`
- Extended recipe update: `results/pretrained_orthogonality_pythia70m_fair_benchmark_extended_4mods_arc_hella100_lora5_20260627`
- Split-data extended update: `results/pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3`

Configuration:

- Model: `EleutherAI/pythia-70m`
- Selected modules: first-layer `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`
- PPL evaluation: 1,016 tokens from the same local zero-shot-backup text source
- Zero-shot tasks: ARC-Easy and HellaSwag, 100 validation examples per task
- Selection rules: fixed recipes use predeclared Q/S/R methods and order; Hessian-guided rows search only on calibration Hessian cost.
- Guided Q+S+R search space: per layer, 6 Q/S/R orders x 2 Q methods x 2 S methods x 2 R methods at fixed 4-bit, keep=0.8, rank=0.5.
- Guided SPQ search space: SPQ layer prior is fixed first, then each selected layer searches permutations of its SPQ operations and same-budget method variants by calibration Hessian cost only.
- LoRA recovery budget in the extended recipe update: fixed and guided SPQ-like LoRA rows both use 5 steps and rank 4.
- Split-data update: 160 text rows were loaded from the same local zero-shot backup source and split sequentially into 32 calibration texts, 64 PPL-evaluation texts, and 64 LoRA-recovery texts. This supersedes the shared-window LoRA rows for fairness claims.
- PIQA note: not included in this run because the 236 server has local backups for ARC-Easy and HellaSwag but not PIQA.

## Q/S/R Subset Summary

| Strategy | Family | Memory ratio | Signed PPL delta | Mean acc. | Mean acc. delta |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | baseline | 1.000 | 0.00% | 0.305 | 0.000 |
| q_only_rtn_4bit | Q-only | 0.250 | +4.56% | 0.315 | +0.010 |
| q_only_rotated_4bit | Q-only | 0.250 | +3.45% | 0.350 | +0.045 |
| s_only_magnitude_keep0p8 | S-only | 0.800 | +0.05% | 0.320 | +0.015 |
| s_only_wanda_keep0p8 | S-only | 0.800 | +0.29% | 0.305 | +0.000 |
| r_only_svd_rank0p5 | R-only | 0.667 | +523.11% | 0.350 | +0.045 |
| r_only_whitened_rank0p5 | R-only | 0.667 | +53.92% | 0.295 | -0.010 |
| qsr_naive_rtn_magnitude_svd | Q+S+R | 0.133 | +536.52% | 0.315 | +0.010 |
| qsr_rotated_wanda_whitened | Q+S+R | 0.133 | +57.64% | 0.260 | -0.045 |
| rqs_rotated_wanda_whitened | Q+S+R | 0.133 | +59.54% | 0.285 | -0.020 |
| hessian_guided_qsr_budget | Hessian-guided Q+S+R | 0.133 | +64.10% | 0.280 | -0.025 |

Per-task zero-shot details are in `metrics/fair_benchmark_zero_shot.csv`. The main machine-readable table is `metrics/fair_benchmark.csv`. The guided run adds `metrics/fair_benchmark_selection.csv`, `figures/fair_benchmark_summary.png`, and `figures/fair_benchmark_guided_competitiveness.png`.

## Extended Recipe Summary

The split-data extended run adds SLiM-like and SPQ-like recipes. Low-loss/frontier rows are not included here because those earlier rows choose candidates by benchmark drop and are search diagnostics, not fair no-result-selection baselines.

| Strategy | Family | Recovery | Memory ratio | Signed PPL delta | Mean acc. delta | Selection rule |
| --- | --- | --- | ---: | ---: | ---: | --- |
| s_only_magnitude_keep0p8 | S-only reference | none | 0.800 | +0.05% | +0.015 | fixed |
| s_only_wanda_keep0p8 | S-only reference | none | 0.800 | +0.29% | +0.000 | fixed |
| q_only_rotated_4bit | Q-only reference | none | 0.250 | +3.45% | +0.045 | fixed |
| qsr_rotated_wanda_whitened | fixed Q+S+R | none | 0.133 | +57.64% | -0.045 | fixed |
| rqs_rotated_wanda_whitened | fixed Q+S+R | none | 0.133 | +59.54% | -0.020 | fixed |
| hessian_guided_qsr_budget | Hessian-guided Q+S+R | none | 0.133 | +64.10% | -0.025 | calibration-only method/order selector |
| slim_like_srq_proxy | SLiM-like proxy | none | 0.133 | +67.45% | -0.035 | fixed |
| spq_like_rsq_no_lora | SPQ-like layer prior | none | 0.196 | +12.31% | +0.030 | fixed |
| hessian_guided_spq_no_lora | Hessian-guided SPQ-like | none | 0.196 | +11.01% | +0.010 | calibration-only method/order selector |
| spq_like_rsq_lora | SPQ-like layer prior | LoRA rank4/5 steps | 0.196 | +13.53% | +0.035 | fixed + equal recovery |
| hessian_guided_spq_lora | Hessian-guided SPQ-like | LoRA rank4/5 steps | 0.196 | +9.74% | +0.015 | calibration-only method/order selector + equal recovery |

The new visualization is `../pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3/figures/fair_benchmark_extended_competitiveness.png`.

## Hessian-Guided Competitiveness Check

The Hessian-guided selector is competitive only against the weakest naive stack, not against the strongest fixed or single-method baselines in this fair benchmark.

At the same nominal Q+S+R memory ratio of 0.133:

| Strategy | Selection rule | Predicted Hessian cost | Signed PPL delta | Mean acc. delta |
| --- | --- | ---: | ---: | ---: |
| qsr_rotated_wanda_whitened | fixed Q->S->R | 21.2457 | +57.64% | -0.045 |
| rqs_rotated_wanda_whitened | fixed R->Q->S | 14.3136 | +59.54% | -0.020 |
| hessian_guided_qsr_budget | layer-wise min Hessian cost | 12.3022 | +64.10% | -0.025 |

The selected layers were:

| Layer | Order | Q | S | R | Predicted cost |
| --- | --- | --- | --- | --- | ---: |
| L0:dense | rqs | rotated_rtn | wanda | whitened_svd | 0.0612 |
| L0:query_key_value | rqs | rtn | wanda | whitened_svd | 6.5833 |
| L0:dense_4h_to_h | rsq | rotated_rtn | wanda | whitened_svd | 0.0676 |
| L0:dense_h_to_4h | rsq | rtn | wanda | whitened_svd | 5.5900 |

This explains the result: the selector mostly chooses R-first orders, but in the two larger/high-impact layers it chooses plain RTN rather than rotated RTN because the local Hessian proxy predicts a lower cost. That reduces the summed calibration Hessian cost, but it does not transfer to lower split-window PPL. Across all fair rows, predicted Hessian cost still separates catastrophic plain SVD/naive-QSR failures from reasonable methods. Within the non-catastrophic methods it is weak, and within the three same-budget QSR rows it is mis-ranked for split-window PPL.

For SPQ-like recipes, the result is more favorable to Hessian guidance on PPL only. At matched SPQ layer prior and memory ratio 0.196, Hessian-guided method/order selection improves no-LoRA PPL delta from +12.31% to +11.01%, and with equal tiny LoRA recovery from +13.53% to +9.74% on disjoint recovery/evaluation text windows. It is not Pareto-dominant: mean zero-shot delta is lower than fixed SPQ-like both without LoRA (+0.010 vs +0.030) and with LoRA (+0.015 vs +0.035). This is still not a global win over high-memory single-method references, and the PPL improvement should be attributed to method+order selection over the fixed SPQ-like recipe rather than order selection alone.

## Interpretation

Under the split-data benchmark, the stacked Q+S+R configurations do **not** beat the single-method quality references on PPL or mean zero-shot accuracy. Those references use higher memory ratios, so this is not a same-memory dominance comparison: `s_only_magnitude_keep0p8` has memory 0.800 and +0.05% PPL, while Q+S+R rows use memory 0.133 and SPQ-like rows use memory 0.196. The fair claim is therefore about tradeoff and recipe competitiveness, not absolute quality at equal memory.

The Q+S+R stacks achieve the lowest nominal memory ratio (0.133), but the aggressive fixed stack is not lossless: `qsr_rotated_wanda_whitened` has +57.64% signed PPL delta and -4.5 percentage points mean accuracy delta; `rqs_rotated_wanda_whitened` has +59.54% signed PPL delta and -2.0 percentage points mean accuracy delta.

The guided Q+S+R stack has lower predicted Hessian cost than both fixed rotated stacks, but it is worse on PPL and only between them on mean zero-shot accuracy. This means the earlier frontier result should be treated as a search/diagnostic result, not as a fair benchmark claim. A paper-strength claim should separate selection from evaluation and must validate any Hessian selection rule on split-window PPL/accuracy, not just calibration loss.

## Required Next Fair Experiment

1. Pre-register memory budgets, e.g. 0.80, 0.50, 0.25, and 0.133 selected-layer memory ratio.
2. Choose each method's configuration from calibration-only criteria, not final benchmark PPL or accuracy.
3. Improve the selector objective before claiming competitiveness: add a rotation prior or activation reconstruction term, penalize unrotated RTN in high-impact layers, and validate whether layer-local Hessian cost should be normalized by layer output variance or downstream sensitivity.
4. Evaluate on a larger PPL window and at least ARC-Easy, HellaSwag, and PIQA after caching PIQA locally.
5. Report signed PPL/NLL deltas, clipped degradation, per-task accuracy, mean accuracy, and memory ratio for every method.
