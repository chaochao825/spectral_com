# Pretrained Small-LLM Compression Orthogonality

- Model: `EleutherAI/pythia-160m`
- Target modules: `query_key_value, dense, dense_h_to_4h, dense_4h_to_h`; selected count: 12
- Text source: `zero_shot_backup:arc_easy,hellaswag`; split policy: `disjoint_sequential_text_windows` (calib=256, eval=512, recovery=512 texts).
- Compression settings: q=rtn/bits4, s=wanda/keep0.8, r=whitened_svd/rank0.5
- Baseline PPL: 54.6310; NLL: 4.0006; zero-shot mean: nan

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 12 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.3452 (n=36). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = -0.3514 (n=36); zero-shot degradation = nan (n=0).
- Taylor/cross-term prediction vs actual loss degradation = 0.5732 (n=36); Frobenius baseline = 0.7254 (n=36); trace-only baseline = 0.5655 (n=36).
- Order gap explanation: R-first conditional overlap = -0.0826 (n=24); singular entropy shift = 0.1939 (n=24); symmetric overlap = 0.2530 (n=24).
- Highest |rho_H| row: L0:dense_4h_to_h pair=qs |rho_H|=0.2919, |A_ij|=0.2601.
- Largest order gap: L6:query_key_value rq vs qr abs loss gap=0.0108.
- Best compressed strategy by PPL: hessian_layerwise PPL=71.4840, degradation=16.8530; baseline PPL=54.6310.

## Method-Coverage Notes

This run is a pretrained-LLM framework experiment, not a claim that the native script reimplements every external baseline.
Text provenance is recorded in `metrics/text_source_metadata.csv`; zero-shot additivity and strategy evaluations use the same per-task example limit so degradation correlations are comparable.
When `--include-fair-benchmark` is enabled, fair benchmark zero-shot scores are in `metrics/fair_benchmark_zero_shot.csv` even if top-level strategy zero-shot was disabled.
Unavailable external baselines in this environment:
- q/gptq: auto-gptq package is not installed in the current environment
- q/awq: AWQ/AutoAWQ package is not installed in the current environment
- s/sparsegpt: SparseGPT package/integration is not installed in the current environment

Native baselines included: RTN quantization, Hadamard rotated RTN proxy, magnitude pruning, Wanda-style activation-aware pruning, vanilla SVD, and activation-whitened SVD proxy.
The `slim_like_srq_proxy` row is a fixed triple-compression recipe proxy; it is not the official SLiM implementation.

## Artifacts

- `metrics/hessian_cosine.csv` and `figures/hessian_cosine_heatmap.png`
- `metrics/additivity.csv`, `metrics/order_gap.csv`, `metrics/correlations.csv`
- `metrics/strategy_performance.csv`, `metrics/layerwise_selection.csv`, `metrics/method_status.csv`
- `metrics/spq_recipe_diagnostics.csv` when `--include-spq-strategies` is enabled
- `metrics/rotation_quantization.csv` and `figures/rotation_quantization_summary.png` when `--include-rotation-analysis` is enabled
- `metrics/low_loss_triple_candidates.csv` when `--include-low-loss-triple` is enabled
- `metrics/lossless_frontier_candidates.csv`, `metrics/lossless_frontier_summary.csv`, and `figures/lossless_frontier_summary.png` when `--include-lossless-frontier` is enabled
- `metrics/fair_benchmark.csv`, `metrics/fair_benchmark_zero_shot.csv`, `metrics/fair_benchmark_selection.csv`, and `figures/fair_benchmark_summary.png` when `--include-fair-benchmark` is enabled
- `figures/pretrained_goal_dashboard.png`
- `figures/largest_order_gap_singular_spectrum.png`
