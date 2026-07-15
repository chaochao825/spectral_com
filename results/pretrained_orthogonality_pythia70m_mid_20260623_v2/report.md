# Pretrained Small-LLM Compression Orthogonality

- Model: `EleutherAI/pythia-70m`
- Target modules: `query_key_value, dense, dense_h_to_4h, dense_4h_to_h`; selected count: 12
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (64 texts).
- Compression settings: q=rtn/bits3, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 68.5878; NLL: 4.2281; zero-shot mean: 0.1875

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 12 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.1449 (n=36). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.1748 (n=36); zero-shot degradation = -0.0503 (n=36).
- Taylor/cross-term prediction vs actual loss degradation = 0.7246 (n=36); Frobenius baseline = 0.5001 (n=36); trace-only baseline = 0.4592 (n=36).
- Order gap explanation: R-first conditional overlap = -0.2513 (n=24); singular entropy shift = 0.1409 (n=24); symmetric overlap = 0.0887 (n=24).
- Highest |rho_H| row: L3:dense pair=qs |rho_H|=0.3669, |A_ij|=0.1535.
- Largest order gap: L3:dense_4h_to_h rq vs qr abs loss gap=0.0576.
- Best compressed strategy by PPL: hessian_layerwise PPL=353.3619, degradation=284.7741; baseline PPL=68.5878.

## Method-Coverage Notes

This run is a pretrained-LLM framework experiment, not a claim that the native script reimplements every external baseline.
PPL/calibration data provenance is recorded in `metrics/text_source_metadata.csv`; zero-shot additivity and strategy evaluations use the same per-task example limit so degradation correlations are comparable.
Unavailable external baselines in this environment:
- q/gptq: auto-gptq package is not installed in the current environment
- q/awq: AWQ/AutoAWQ package is not installed in the current environment
- s/sparsegpt: SparseGPT package/integration is not installed in the current environment

Native baselines included: RTN quantization, magnitude pruning, Wanda-style activation-aware pruning, vanilla SVD, and activation-whitened SVD proxy.
The `slim_like_srq_proxy` row is a fixed triple-compression recipe proxy; it is not the official SLiM implementation.

## Artifacts

- `metrics/hessian_cosine.csv` and `figures/hessian_cosine_heatmap.png`
- `metrics/additivity.csv`, `metrics/order_gap.csv`, `metrics/correlations.csv`
- `metrics/strategy_performance.csv`, `metrics/layerwise_selection.csv`, `metrics/method_status.csv`
- `figures/pretrained_goal_dashboard.png`
- `figures/largest_order_gap_singular_spectrum.png`
