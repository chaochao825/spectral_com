# Pretrained Small-LLM Compression Orthogonality

- Model: `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B`
- Target modules: `q_proj, o_proj, up_proj, down_proj`; selected count: 8
- Text source used for PPL/calibration: `zero_shot_backup:arc_easy,hellaswag` (16 texts).
- Compression settings: q=rtn/bits3, s=wanda/keep0.5, r=whitened_svd/rank0.5
- Baseline PPL: 30.6148; NLL: 3.4215; zero-shot mean: 0.5000

## Goal-Criterion Evidence

- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over 8 pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.
- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = 0.2974 (n=24). Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.
- Real degradation: Spearman(|rho_H|, PPL degradation) = 0.1461 (n=24); zero-shot degradation = nan (n=24).
- Taylor/cross-term prediction vs actual loss degradation = 0.6487 (n=24); Frobenius baseline = 0.4443 (n=24); trace-only baseline = 0.5687 (n=24).
- Order gap explanation: R-first conditional overlap = -0.6765 (n=16); singular entropy shift = 0.6412 (n=16); symmetric overlap = -0.5676 (n=16).
- Highest |rho_H| row: L0:down_proj pair=qs |rho_H|=0.2996, |A_ij|=0.0766.
- Largest order gap: L27:down_proj rq vs qr abs loss gap=0.2638.
- Best compressed strategy by PPL: slim_like_srq_proxy PPL=43.5229, degradation=12.9081; baseline PPL=30.6148.

## SPQ-Like Recipe Evidence

- Fixed SPQ-like no-LoRA uses attention R+Q, MLP S+Q, and Q-only for other selected linear modules with q=rtn, s=wanda, r=svd.
- No-LoRA comparison: fixed SPQ-like PPL=64.7770, Hessian-guided-SPQ PPL=50.4522; delta guided-fixed=-14.3248. Both use the same nominal bits/keep/rank budget.
- LoRA-recovered comparison (3 steps, rank 4): fixed SPQ-like PPL=62.5219, Hessian-guided-SPQ PPL=49.7825; delta guided-fixed=-12.7394.
- `metrics/spq_recipe_diagnostics.csv` records the SPQ-applicable pair rho_H, fixed/reversed predicted Hessian costs, and Hessian-guided order/method choices per layer.

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
- `metrics/spq_recipe_diagnostics.csv` when `--include-spq-strategies` is enabled
- `figures/pretrained_goal_dashboard.png`
- `figures/largest_order_gap_singular_spectrum.png`
