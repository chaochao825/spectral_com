# Pretrained Small-LLM Compression Orthogonality

- Model: `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B`
- Target modules: `q_proj, o_proj, up_proj, down_proj`; selected count: 12
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (32 texts).
- Compression settings: q=rtn/bits3, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 12.3748; NLL: 2.5157; zero-shot mean: 0.5000

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 12 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.3876 (n=36). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.0875 (n=36); zero-shot degradation = 0.1286 (n=36).
- Taylor/cross-term prediction vs actual loss degradation = 0.4798 (n=36); Frobenius baseline = 0.1869 (n=36); trace-only baseline = 0.4103 (n=36).
- Order gap explanation: R-first conditional overlap = -0.4078 (n=24); singular entropy shift = 0.5261 (n=24); symmetric overlap = -0.1252 (n=24).
- Highest |rho_H| row: L14:up_proj pair=qs |rho_H|=0.3596, |A_ij|=0.2403.
- Largest order gap: L27:down_proj rq vs qr abs loss gap=0.1874.
- Best compressed strategy by PPL: slim_like_srq_proxy PPL=17.1197, degradation=4.7449; baseline PPL=12.3748.

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
