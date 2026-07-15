# Pretrained Small-LLM Compression Orthogonality

- Model: `EleutherAI/pythia-70m`
- Target modules: `query_key_value, dense, dense_h_to_4h, dense_4h_to_h`; selected count: 4
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (64 texts).
- Compression settings: q=rtn/bits4, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 69.1944; NLL: 4.2369; zero-shot mean: nan

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 4 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.5594 (n=12). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.0210 (n=12); zero-shot degradation = nan (n=0).
- Taylor/cross-term prediction vs actual loss degradation = 0.5175 (n=12); Frobenius baseline = 0.8252 (n=12); trace-only baseline = 0.7832 (n=12).
- Order gap explanation: R-first conditional overlap = -0.4762 (n=8); singular entropy shift = -0.6190 (n=8); symmetric overlap = -0.1190 (n=8).
- Highest |rho_H| row: L0:query_key_value pair=sr |rho_H|=0.0563, |A_ij|=0.7362.
- Largest order gap: L0:dense rq vs qr abs loss gap=0.0329.
- Best compressed strategy by PPL: hessian_layerwise PPL=93.4597, degradation=24.2653; baseline PPL=69.1944.

## Method-Coverage Notes

This run is a pretrained-LLM framework experiment, not a claim that the native script reimplements every external baseline.
PPL/calibration data provenance is recorded in `metrics/text_source_metadata.csv`; zero-shot additivity and strategy evaluations use the same per-task example limit so degradation correlations are comparable.
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
- `metrics/fair_benchmark.csv`, `metrics/fair_benchmark_zero_shot.csv`, and `figures/fair_benchmark_summary.png` when `--include-fair-benchmark` is enabled
- `figures/pretrained_goal_dashboard.png`
- `figures/largest_order_gap_singular_spectrum.png`
