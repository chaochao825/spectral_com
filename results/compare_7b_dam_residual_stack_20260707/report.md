# 7B Residual-Stack vs DAM-like Proxy Comparison

This report aggregates existing small-model runs plus two Qwen2-7B layer-subset runs. The DAM rows are proxy implementations from the paper equations, not an official reproduction.

## Strategy Table

| run | best <=0.258 strategy | best PPL delta | Q+L delta | Q+S+L delta | DAM-grid delta | selector mix note |
|---|---:|---:|---:|---:|---:|---|
| Pythia-70M | Sequential QSR | +4.459 | +10.071 | +18.192 | +35.188 |  |
| Pythia-160M | Q+S+L | -0.748 | +0.702 | -0.748 | +23.151 |  |
| Qwen2-7B attention-only | Q+L | -0.537 | -0.537 | +0.165 | +3.833 | selector chose Q+S for all 3 o_proj layers; Q+L had best PPL |
| Qwen2-7B attn+MLP | Q+S+L | -0.281 | +0.029 | -0.281 | +4.323 | selector chose Q+S for all 6 layers; Q+S+L PPL was slightly lower but not selected |

## Main Readout

- Qwen2-7B attention-only does not support residual stacking: Q+L is best, while Q+S+L is worse than Q+L.
- Qwen2-7B attention+MLP changes the sign: Q+S+L beats Q+L at the same 0.258 memory and also beats fixed SPQ-like, sequential QSR, and Hessian-guided SPQ in this subset.
- The gain is driven mostly by residual sparse/Wanda on MLP and attention output layers; the greedy selector chose Q+S everywhere, while Q+S+L was only slightly better at PPL, so the current selector is still imperfect.
- DAM-like proxy remains far worse in these runs. This should be reported as a proxy limitation, because no official implementation was available in the current repo/workflow.
- Zero-shot means are unchanged at 0.75 because this smoke uses a tiny backup subset; it is not sensitive enough to rank methods.

## Files

- Strategy CSV: `results/compare_7b_dam_residual_stack_20260707/strategy_comparison.csv`
- PPL delta heatmap: `results/compare_7b_dam_residual_stack_20260707/figures/ppl_delta_heatmap.png`
- Qwen2-7B copied figures: `figures/qwen7b_6mods_*.png`
