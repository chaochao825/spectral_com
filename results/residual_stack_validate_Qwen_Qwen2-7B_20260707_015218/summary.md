# Residual-Stack Validation

- Model: `Qwen/Qwen2-7B`
- Selected modules: 6 (`model.layers.0.mlp.down_proj, model.layers.0.mlp.up_proj, model.layers.0.self_attn.o_proj, model.layers.14.mlp.down_proj, model.layers.14.mlp.up_proj, model.layers.14.self_attn.o_proj`)
- Mode: `residual_stack_validate`; target memory ratio: 0.2580; q base: 4-bit.
- Text source: `zero_shot_backup:arc_easy,hellaswag`; calib=16 texts, eval=16 texts.
- Dense baseline PPL: 45.5759; NLL: 3.8194; zero-shot mean: 0.7500.

## Strategy Results

| strategy | memory | PPL | signed PPL delta | zero-shot | zero-shot delta | <= target |
|---|---:|---:|---:|---:|---:|---|
| Q only | 0.2500 | 45.9601 | +0.3841 | 0.7500 | +0.0000 | True |
| Q+L same budget | 0.2580 | 45.6047 | +0.0288 | 0.7500 | +0.0000 | True |
| Q+S same budget | 0.2580 | 45.2999 | -0.2760 | 0.7500 | +0.0000 | True |
| Q+S+L same budget | 0.2580 | 45.2952 | -0.2807 | 0.7500 | +0.0000 | True |
| Residual-stack selector | 0.2580 | 45.2999 | -0.2760 | 0.7500 | +0.0000 | True |
| Sequential QSR matched | 0.2579 | 47.1122 | +1.5363 | 0.7500 | +0.0000 | True |
| Fixed SPQ-like matched | 0.2580 | 45.6666 | +0.0907 | 0.7500 | +0.0000 | True |
| Hessian-guided SPQ matched | 0.2580 | 45.8652 | +0.2893 | 0.7500 | +0.0000 | True |
| Paper L->Q factor matched | 0.2580 | 50.1146 | +4.5387 | 0.7500 | +0.0000 | True |
| Paper DAM closed matched | 0.2580 | 49.6686 | +4.0926 | 0.7500 | +0.0000 | True |
| Paper DAM activation-grid matched | 0.2580 | 49.8985 | +4.3226 | 0.7500 | +0.0000 | True |

## Evidence Notes

- Same-budget rule: a residual-stack win is counted only when `nominal_memory_ratio <= target_memory_ratio`; residual rows use additive component accounting and baseline rows use a memory-only keep/rank grid matched under the same target.
- Selector memory summary: selected=0.2580, global feasible=True, filter fallback layers=0.
- Selected candidate mix: {"q_s": 6}.
- Conditional-overlap diagnostic: Spearman(positive rho, activation gain vs Q-only) = 0.1077 (n=54); negative values mean lower conflict tends to give larger activation gain.
- DAM comparison rows are DAM-like proxies implemented from the paper equations, not an official repository: `dam_closed` uses Eq.21-style balancing, and `dam_activation_grid` selects the diagonal exponent using calibration activation reconstruction.
- Conservative verdict for this run: positive on PPL for Q+S+L vs Q+L at the recorded budget.

## Artifacts

- `metrics/residual_stack_candidates.csv`
- `metrics/residual_stack_selection.csv`
- `metrics/dam_factor_selection.csv` when `--include-dam-comparison` is enabled
- `metrics/residual_stack_strategy.csv`
- `metrics/residual_stack_zero_shot.csv`
- `selected_recipe.json`
- `figures/memory_ppl_frontier.png`
- `figures/candidate_activation_error_by_layer.png`
- `figures/conditional_overlap_heatmap.png`
- `figures/residual_structure_scatter.png`

This experiment validates a framework hypothesis only; it is not a SOTA claim.
