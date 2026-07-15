# Pretrained Small-LLM Compression Orthogonality

- Model: `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B`
- Target modules: `q_proj, o_proj`; selected count: 2
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (8 texts).
- Compression settings: q=rtn/bits4, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 36.6479; NLL: 3.6014; zero-shot mean: 0.5000

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 2 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.7143 (n=6). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.6000 (n=6); zero-shot degradation = nan (n=6).
- Taylor/cross-term prediction vs actual loss degradation = 0.8286 (n=6); Frobenius baseline = 0.8857 (n=6); trace-only baseline = 0.8857 (n=6).
- Order gap explanation: R-first conditional overlap = 0.4000 (n=4); singular entropy shift = -0.4000 (n=4); symmetric overlap = -0.4000 (n=4).
- Highest |rho_H| row: L0:q_proj pair=sr |rho_H|=0.0741, |A_ij|=1.5161.
- Largest order gap: L0:o_proj rs vs sr abs loss gap=0.0171.
- Best compressed strategy by PPL: fixed_qsr_naive PPL=36.1564, degradation=-0.4916; baseline PPL=36.6479.

## Rotation-Quantization Evidence

- Hadamard rotated RTN is evaluated as `fixed_qsr_rotated_q` with q=rotated_rtn, s=wanda, r=whitened_svd; PPL=36.6135, degradation=-0.0345.
- Compared with `fixed_qsr_default`, rotated-Q delta PPL=0.0511 under the same bits/keep/rank settings.
- `metrics/rotation_quantization.csv` records RTN vs rotated RTN relative weight error, Hessian self cost, and input-channel max/median outlier ratios.

## Low-Loss Triple-Stack Evidence

- `low_loss_triple_stack` applies all three operations with order=rqs, q=rotated_rtn, s=wanda, r=whitened_svd, bits=8, keep=0.9950, rank=0.9950.
- Benchmark-drop criterion: metric=ppl (requested=ppl), drop=0.0244%, threshold=1.0000%, pass=True.
- Result: PPL=36.6569, PPL degradation=0.0089, zero-shot=0.5000.
- `metrics/low_loss_triple_candidates.csv` records every evaluated conservative Q+S+R candidate.

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
- `figures/pretrained_goal_dashboard.png`
- `figures/largest_order_gap_singular_spectrum.png`
