# Pretrained Orthogonality Detailed Conclusion

This document extends the earlier conclusion with per-configuration details and adds a larger model validation run. The larger run uses `EleutherAI/pythia-160m` (162,322,944 parameters), compared with the earlier `EleutherAI/pythia-70m` runs (70,426,624 parameters).

Important scope notes: PPL/calibration uses local ARC-Easy/HellaSwag backup text because WikiText-2 download timed out; provenance is recorded per run in `metrics/text_source_metadata.csv`. The overlap metric is a layer-local Hessian/Gauss-Newton proxy based on activation covariance `X^T X`, not the full model Hessian. Additivity rows test linearized perturbation sums `W + Delta_i + Delta_j`; executable compression-order effects are in `order_gap.csv`.

## Experiment Configurations

| Run | Model | Params | Purpose | Modules | Compression | Methods | Eval / zero-shot | Text source |
|---|---|---:|---|---:|---|---|---|---|
| 70M_default_4bit_6mods | EleutherAI/pythia-70m | 70,426,624 | mild compression sanity check with fewer modules; tests whether the framework is useful before aggressive degradation | 6 | q 4-bit, s keep 0.5, r rank 0.5 | Q=rtn, S=wanda, R=whitened_svd | 1016 PPL tokens; arc_easy,hellaswag @ 8 examples/task | zero_shot_backup:arc_easy,hellaswag |
| 70M_mid_3bit_12mods | EleutherAI/pythia-70m | 70,426,624 | main 70M configuration; same 12-module coverage used for stronger correlation and strategy comparisons | 12 | q 3-bit, s keep 0.5, r rank 0.5 | Q=rtn, S=wanda, R=whitened_svd | 1016 PPL tokens; arc_easy,hellaswag @ 8 examples/task | zero_shot_backup:arc_easy,hellaswag |
| 70M_strong_2bit_12mods | EleutherAI/pythia-70m | 70,426,624 | stress test; checks whether overlap/additivity signals become clearer under larger perturbations | 12 | q 2-bit, s keep 0.4, r rank 0.4 | Q=rtn, S=wanda, R=whitened_svd | 1016 PPL tokens; arc_easy,hellaswag @ 8 examples/task | zero_shot_backup:arc_easy,hellaswag |
| 160M_mid_3bit_12mods | EleutherAI/pythia-160m | 162,322,944 | larger-parameter validation run with the same mid configuration as 70M_mid | 12 | q 3-bit, s keep 0.5, r rank 0.5 | Q=rtn, S=wanda, R=whitened_svd | 1016 PPL tokens; arc_easy,hellaswag @ 8 examples/task | zero_shot_backup:arc_easy,hellaswag |

## Result Summary

| Run | rho additivity | rho PPL | rho zero-shot | Taylor vs loss | Frobenius vs loss | Spectrum vs order gap | Baseline PPL | Hessian PPL | Fixed default PPL | SLiM-proxy PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70M_default_4bit_6mods | 0.1249 | 0.2941 | 0.2877 | -0.0526 | 0.1290 | -0.4685 | 68.59 | 97.05 | 105.4 | 102.3 |
| 70M_mid_3bit_12mods | 0.1449 | 0.1748 | -0.0503 | 0.7246 | 0.5001 | 0.1409 | 68.59 | 353.4 | 398.1 | 455.2 |
| 70M_strong_2bit_12mods | 0.4842 | 0.2165 | 0.1164 | 0.5367 | 0.4857 | 0.4261 | 68.59 | 3.607e+07 | 1.042e+08 | 5.899e+07 |
| 160M_mid_3bit_12mods | 0.0654 | 0.2932 | 0.0400 | 0.4610 | 0.1012 | 0.3635 | 45.97 | 112.5 | 135.7 | 113.5 |

## Interpretation

- The larger Pythia-160M run does not overturn the earlier mixed conclusion. It strengthens the claim that Taylor/cross-term style prediction can beat simple Frobenius and trace-only baselines in a larger model setting: `0.4610` vs `0.1012` and `0.1441` on the matched mid configuration.
- The larger run also preserves the utility signal for layer-wise selection by PPL: Hessian-guided layer-wise PPL `112.5270` is slightly better than fixed default `135.6822` and SLiM-like proxy `113.5150`. The margin over the SLiM-like proxy is small, so this should be presented as preliminary.
- The weak point remains `rho_H` as a direct predictor of linearized additivity and real task degradation. In Pythia-160M mid, `rho_H` vs additivity is only `0.0654`, and `rho_H` vs zero-shot degradation is `0.0400`. This means the framework currently has stronger evidence as a Taylor/cross-term loss diagnostic and layer-wise selector than as a standalone universal degradation predictor.
- Order sensitivity remains observable, but the best explanatory variable varies by setting. For Pythia-160M mid, singular spectrum deltas correlate with order gap better than symmetric overlap (`0.3635` vs `-0.0139`).

## Files

- `summary.csv`: machine-readable cross-run metrics.
- `experiment_configurations.csv`: exact per-run setup details.
- `figures/correlation_by_config_and_model.png`: correlation evidence across all configs.
- `figures/strategy_ppl_by_config_and_model.png`: strategy PPL comparison.
- `figures/model_scale_mid_config_comparison.png`: 70M vs 160M matched mid-config comparison.
