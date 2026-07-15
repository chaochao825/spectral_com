# SPQ-Like Execution Summary

This file records the implemented SPQ-like baselines and the first low-budget execution results. These are smoke/diagnostic runs, not final deployment-scale SPQ reproduction runs. LoRA recovery used only 3-5 steps, so it verifies the comparison path rather than saturated recovery quality.

## Implemented Strategies

- `spq_like_rsq_no_lora`: attention modules use `R->Q`, MLP modules use `S->Q`, other selected linear modules use `Q`; fixed methods are `q=rtn`, `s=wanda`, `r=svd`.
- `hessian_guided_spq_no_lora`: same nominal bit/keep/rank budget and same SPQ layer-type prior, but chooses order and same-budget method variants by local Hessian cost.
- `spq_like_rsq_lora` and `hessian_guided_spq_lora`: same two strategies with equal LoRA rank/steps/lr recovery budgets.
- `metrics/spq_recipe_diagnostics.csv` explains the fixed recipe per layer with SPQ pair `rho_H`, fixed/reversed predicted Hessian cost, and guided choices.

## Result Table

| Run | Mods | Tokens | LoRA steps | Base PPL | Fixed no-LoRA | Guided no-LoRA | Guided-Fixed | Fixed LoRA | Guided LoRA | Guided-Fixed LoRA | rho add. | rho PPL | Taylor-loss | Frob-loss | Spec-order | Order changes | Method changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pythia-70M SPQ smoke | 12.0000 | 508.0 | 5.0000 | 59.8666 | 1194.7 | 259.7 | -935.0 | 1013.4 | 235.6 | -777.8 | 0.2296 | 0.1851 | 0.6839 | 0.4126 | 0.3626 | 2.0000 | 7.0000 |
| Qwen2.5-1.5B SPQ smoke | 8.0000 | 254.0 | 3.0000 | 30.6148 | 64.7770 | 50.4522 | -14.3248 | 62.5219 | 49.7825 | -12.7394 | 0.2974 | 0.1461 | 0.6487 | 0.4443 | 0.6412 | 2.0000 | 6.0000 |

## Interpretation

- In both smoke runs, Hessian-guided-SPQ improves over the fixed SPQ-like recipe under the same nominal compression budget.
- The improvement survives equal tiny LoRA recovery budgets: Pythia guided-vs-fixed LoRA PPL delta is -777.8, and Qwen guided-vs-fixed LoRA PPL delta is -12.7394.
- The diagnostic explanation is consistent with the original framework: Taylor/cross-term correlations beat Frobenius in both runs, and singular-spectrum/order-disagreement metrics explain order gaps better than raw symmetric overlap.
- Limitations: zero-shot limits are too small for accuracy conclusions; LoRA steps are intentionally tiny; the SPQ-like implementation uses dense replacement/proxy quantization, not SPQ's deployable memory accounting or int8 kernels.

## Artifacts

- `Pythia-70M SPQ smoke`: `pretrained_orthogonality_pythia70m_spq_lora_smoke_20260624_v2/report.md`, `metrics/strategy_performance.csv`, `metrics/spq_recipe_diagnostics.csv`, `figures/pretrained_goal_dashboard.png`
- `Qwen2.5-1.5B SPQ smoke`: `pretrained_orthogonality_qwen25_1p5b_spq_lora_smoke_20260624/report.md`, `metrics/strategy_performance.csv`, `metrics/spq_recipe_diagnostics.csv`, `figures/pretrained_goal_dashboard.png`
- Machine-readable summary: `spq_execution_summary.csv`
- Cross-run visualization: `figures/spq_fixed_vs_guided_summary.svg`
