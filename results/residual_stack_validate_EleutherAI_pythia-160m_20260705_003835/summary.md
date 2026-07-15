# Residual-Stack Validation

- Model: `EleutherAI/pythia-160m`
- Selected modules: 4 (`gpt_neox.layers.0.attention.dense, gpt_neox.layers.0.mlp.dense_4h_to_h, gpt_neox.layers.0.mlp.dense_h_to_4h, gpt_neox.layers.6.attention.dense`)
- Mode: `residual_stack_validate`; target memory ratio: 0.2580; q base: 4-bit.
- Text source: `zero_shot_backup:arc_easy,hellaswag`; calib=16 texts, eval=16 texts.
- Dense baseline PPL: 78.8096; NLL: 4.3670; zero-shot mean: 0.5000.

## Strategy Results

| strategy | memory | PPL | signed PPL delta | zero-shot | zero-shot delta | <= target |
|---|---:|---:|---:|---:|---:|---|
| Q only | 0.2500 | 79.0052 | +0.1956 | 0.5000 | +0.0000 | True |
| Q+L same budget | 0.2568 | 79.5121 | +0.7024 | 0.5000 | +0.0000 | True |
| Q+S same budget | 0.2580 | 79.1933 | +0.3837 | 0.5000 | +0.0000 | True |
| Q+S+L same budget | 0.2572 | 78.0618 | -0.7478 | 0.5000 | +0.0000 | True |
| Residual-stack selector | 0.2575 | 80.9982 | +2.1886 | 0.5000 | +0.0000 | True |
| Sequential QSR matched | 0.2580 | 79.1391 | +0.3295 | 0.5000 | +0.0000 | True |
| Fixed SPQ-like matched | 0.2580 | 78.5499 | -0.2598 | 0.5000 | +0.0000 | True |
| Hessian-guided SPQ matched | 0.2580 | 80.5068 | +1.6972 | 0.5000 | +0.0000 | True |

## Evidence Notes

- Same-budget rule: a residual-stack win is counted only when `nominal_memory_ratio <= target_memory_ratio`; residual rows use additive component accounting and baseline rows use a memory-only keep/rank grid matched under the same target.
- Selector memory summary: selected=0.2575, global feasible=True, filter fallback layers=0.
- Selected candidate mix: {"q_l": 2, "q_s": 1, "q_s_l": 1}.
- Conditional-overlap diagnostic: Spearman(positive rho, activation gain vs Q-only) = 0.1274 (n=64); negative values mean lower conflict tends to give larger activation gain.
- Conservative verdict for this run: positive on PPL for Q+S+L vs Q+L at the recorded budget.

## Artifacts

- `metrics/residual_stack_candidates.csv`
- `metrics/residual_stack_selection.csv`
- `metrics/residual_stack_strategy.csv`
- `metrics/residual_stack_zero_shot.csv`
- `selected_recipe.json`
- `figures/memory_ppl_frontier.png`
- `figures/candidate_activation_error_by_layer.png`
- `figures/conditional_overlap_heatmap.png`
- `figures/residual_structure_scatter.png`

This experiment validates a framework hypothesis only; it is not a SOTA claim.
