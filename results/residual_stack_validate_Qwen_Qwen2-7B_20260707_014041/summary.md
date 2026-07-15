# Residual-Stack Validation

- Model: `Qwen/Qwen2-7B`
- Selected modules: 3 (`model.layers.0.self_attn.o_proj, model.layers.14.self_attn.o_proj, model.layers.27.self_attn.o_proj`)
- Mode: `residual_stack_validate`; target memory ratio: 0.2580; q base: 4-bit.
- Text source: `zero_shot_backup:arc_easy,hellaswag`; calib=8 texts, eval=8 texts.
- Dense baseline PPL: 67.1155; NLL: 4.2064; zero-shot mean: 0.7500.

## Strategy Results

| strategy | memory | PPL | signed PPL delta | zero-shot | zero-shot delta | <= target |
|---|---:|---:|---:|---:|---:|---|
| Q only | 0.2500 | 67.0159 | -0.0996 | 0.7500 | +0.0000 | True |
| Q+L same budget | 0.2578 | 66.5785 | -0.5370 | 0.7500 | +0.0000 | True |
| Q+S same budget | 0.2580 | 67.5588 | +0.4433 | 0.7500 | +0.0000 | True |
| Q+S+L same budget | 0.2577 | 67.2807 | +0.1652 | 0.7500 | +0.0000 | True |
| Residual-stack selector | 0.2580 | 67.5588 | +0.4433 | 0.7500 | +0.0000 | True |
| Sequential QSR matched | 0.2580 | 67.1672 | +0.0517 | 0.7500 | +0.0000 | True |
| Fixed SPQ-like matched | 0.2550 | 67.8340 | +0.7185 | 0.7500 | +0.0000 | True |
| Hessian-guided SPQ matched | 0.2550 | 68.0849 | +0.9694 | 0.7500 | +0.0000 | True |
| Paper L->Q factor matched | 0.2580 | 75.4997 | +8.3842 | 0.7500 | +0.0000 | True |
| Paper DAM closed matched | 0.2580 | 71.6608 | +4.5453 | 0.7500 | +0.0000 | True |
| Paper DAM activation-grid matched | 0.2580 | 70.9489 | +3.8334 | 0.7500 | +0.0000 | True |

## Evidence Notes

- Same-budget rule: a residual-stack win is counted only when `nominal_memory_ratio <= target_memory_ratio`; residual rows use additive component accounting and baseline rows use a memory-only keep/rank grid matched under the same target.
- Selector memory summary: selected=0.2580, global feasible=True, filter fallback layers=0.
- Selected candidate mix: {"q_s": 3}.
- Conditional-overlap diagnostic: Spearman(positive rho, activation gain vs Q-only) = -0.0998 (n=27); negative values mean lower conflict tends to give larger activation gain.
- DAM comparison rows are DAM-like proxies implemented from the paper equations, not an official repository: `dam_closed` uses Eq.21-style balancing, and `dam_activation_grid` selects the diagonal exponent using calibration activation reconstruction.
- Conservative verdict for this run: not positive against Q+L at the recorded budget.

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
