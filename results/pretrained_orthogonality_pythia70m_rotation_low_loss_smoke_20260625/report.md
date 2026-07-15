# Pretrained Small-LLM Compression Orthogonality

- Model: `EleutherAI/pythia-70m`
- Target modules: `query_key_value, dense, dense_h_to_4h, dense_4h_to_h`; selected count: 4
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (16 texts).
- Compression settings: q=rtn/bits4, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 76.7669; NLL: 4.3408; zero-shot mean: 0.5000

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 4 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.1189 (n=12). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.0280 (n=12); zero-shot degradation = -0.3585 (n=12).
- Taylor/cross-term prediction vs actual loss degradation = 0.5385 (n=12); Frobenius baseline = 0.4755 (n=12); trace-only baseline = 0.5804 (n=12).
- Order gap explanation: R-first conditional overlap = -0.1190 (n=8); singular entropy shift = 0.0000 (n=8); symmetric overlap = 0.1429 (n=8).
- Highest |rho_H| row: L0:query_key_value pair=sr |rho_H|=0.0552, |A_ij|=0.0206.
- Largest order gap: L0:dense_h_to_4h rs vs sr abs loss gap=0.0693.
- Best compressed strategy by PPL: low_loss_triple_stack PPL=75.6100, degradation=-1.1569; baseline PPL=76.7669.

## Rotation-Quantization Evidence

- Hadamard rotated RTN is evaluated as `fixed_qsr_rotated_q` with q=rotated_rtn, s=wanda, r=whitened_svd; PPL=88.1160, degradation=11.3491.
- Compared with `fixed_qsr_default`, rotated-Q delta PPL=-0.2384 under the same bits/keep/rank settings.
- `metrics/rotation_quantization.csv` records RTN vs rotated RTN relative weight error, Hessian self cost, and input-channel max/median outlier ratios.

## Low-Loss Triple-Stack Evidence

- `low_loss_triple_stack` applies all three operations with order=rqs, q=rtn, s=wanda, r=whitened_svd, bits=8, keep=0.9950, rank=0.9950.
- Benchmark-drop criterion: metric=ppl (requested=ppl), drop=0.0000%, threshold=1.0000%, pass=True.
- Result: PPL=75.6100, PPL degradation=-1.1569, zero-shot=0.5000.
- `metrics/low_loss_triple_candidates.csv` records every evaluated conservative Q+S+R candidate.

## SPQ-Like Recipe Evidence

- Fixed SPQ-like no-LoRA uses attention R+Q, MLP S+Q, and Q-only for other selected linear modules with q=rtn, s=wanda, r=svd.
- No-LoRA comparison: fixed SPQ-like PPL=91.8167, Hessian-guided-SPQ PPL=82.8372; delta guided-fixed=-8.9795. Both use the same nominal bits/keep/rank budget.
- `metrics/spq_recipe_diagnostics.csv` records the SPQ-applicable pair rho_H, fixed/reversed predicted Hessian costs, and Hessian-guided order/method choices per layer.

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
