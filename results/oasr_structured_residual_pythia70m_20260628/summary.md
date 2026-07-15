# OASR Structured Residual Smoke Result

- Model: `EleutherAI/pythia-70m`
- Dataset request: `wikitext/wikitext-2-raw-v1` split `validation`; backup `wikitext_2_raw`; allow_fallback=True
- Target layers: `attention_o,mlp_up,mlp_down`; max_layers=4
- Quantizer methods: `rtn`; q_bits=4,3,2; group_size=128
- Block sizes: `16,32,64`; C:L splits: `0.25:0.75,0.5:0.5,0.75:0.25`; include_l_c_order=True
- Calibration/eval: calib_limit=8; eval_limit=8; activation_rows=128; sequence_length=128; batch_size=1
- Dense baseline PPL: 26.2220; NLL: 3.2666; tokens: 1016
- Target memory ratios: 0.196,0.220,0.258
- Conditional-overlap filter threshold: 0.3

## Strategy PPL

| Strategy | Family | Target memory | Actual memory | PPL | PPL delta | Mean score | Status |
|---|---|---:|---:|---:|---:|---:|---|
| dense_baseline | baseline | 1.000 | 1.000 | 26.2220 | +0.0000 | 0 | ok |
| q_only_target0.196 | q_only | 0.196 | 0.188 | 27.4833 | +1.2614 | 0.08012 | ok |
| q_l_target0.196 | q_l | 0.196 | 0.195 | 27.6658 | +1.4438 | 0.07741 | ok |
| q_c_target0.196 | q_c | 0.196 | 0.156 | 116.1148 | +89.8928 | 0.6295 | ok |
| q_c_l_target0.196 | q_c_l | 0.196 | 0.195 | 66.0001 | +39.7781 | 0.5391 | ok |
| q_l_c_target0.196 | q_l_c | 0.196 | 0.192 | 70.0637 | +43.8417 | 0.5426 | ok |
| selector_target0.196 | selector | 0.196 | 0.195 | 27.6658 | +1.4438 | 0.07741 | ok |
| q_only_target0.220 | q_only | 0.220 | 0.188 | 27.4833 | +1.2614 | 0.08012 | ok |
| q_l_target0.220 | q_l | 0.220 | 0.219 | 27.2464 | +1.0244 | 0.07359 | ok |
| q_c_target0.220 | q_c | 0.220 | 0.203 | 27.6250 | +1.4030 | 0.07908 | ok |
| q_c_l_target0.220 | q_c_l | 0.220 | 0.218 | 27.3875 | +1.1655 | 0.07476 | ok |
| q_l_c_target0.220 | q_l_c | 0.220 | 0.218 | 27.5966 | +1.3747 | 0.07474 | ok |
| selector_target0.220 | selector | 0.220 | 0.219 | 27.2464 | +1.0244 | 0.07359 | ok |
| q_only_target0.258 | q_only | 0.258 | 0.250 | 25.8638 | -0.3582 | 0.01494 | ok |
| q_l_target0.258 | q_l | 0.258 | 0.257 | 25.8054 | -0.4166 | 0.01426 | ok |
| q_c_target0.258 | q_c | 0.258 | 0.219 | 27.5644 | +1.3425 | 0.07757 | ok |
| q_c_l_target0.258 | q_c_l | 0.258 | 0.257 | 27.0520 | +0.8301 | 0.06899 | ok |
| q_l_c_target0.258 | q_l_c | 0.258 | 0.254 | 27.4051 | +1.1832 | 0.06989 | ok |
| selector_target0.258 | selector | 0.258 | 0.257 | 25.8054 | -0.4166 | 0.01426 | ok |

## Same-Budget Interpretation

- Target 0.196: Q+C+L PPL 66.0001 vs Q+L 27.6658; same-memory=True; Q+C+L better=False.
- Target 0.220: Q+C+L PPL 27.3875 vs Q+L 27.2464; same-memory=True; Q+C+L better=False.
- Target 0.258: Q+C+L PPL 27.0520 vs Q+L 25.8054; same-memory=True; Q+C+L better=False.

## Residual Structure Diagnostic

- Block-circulant projection beats random structured baseline in 208/208 C-bearing candidates.
- Selector picked Q+C+L in 0/12 selected layer-budget rows.

## Post-Hoc Analysis

- This is a fallback-text smoke run because the remote server could not fetch `wikitext/wikitext-2-raw-v1` through the configured HF mirror. Treat PPL as an end-to-end sanity metric for this prototype, not as a standard benchmark number.
- Same-budget evidence does not support the current OASR block-circulant residual. Q+C+L is worse than Q+L at all three target memory ratios, and the selector chooses Q+L for every selected layer-budget row.
- The negative result is not only caused by qbit mismatch. When controlling for the same q3 base, Q+C+L has higher activation error than Q+L in all 8 comparable layer-budget cases at target memory 0.220 and 0.258.
- C_res is structurally non-random but not competitive with low-rank under the same residual memory. It beats the random block-circulant baseline in 208/208 C-bearing candidates, but block-circulant projection error is worse than matched-memory low-rank projection error in 208/208 candidates.
- Signed conditional overlap is mostly negative or near zero, especially for rho(Q+C,L), yet this does not translate into activation or PPL gain. In this prototype, low/negative overlap is not sufficient as a final selector for C_res.
- Conservative conclusion: block-circulant C_res is not supported yet for these Pythia-70M layers and budgets. The next meaningful variants are residual whitening before C projection, conditional residual decomposition instead of sequential Q+C+L, and richer structured residuals such as Monarch/Butterfly.

## Files

- `candidate_metrics.csv`
- `selected_recipe.json`
- `figures/memory_ppl_frontier.png`
- `figures/candidate_activation_error_by_layer.png`
- `figures/conditional_overlap_heatmap.png`
- `figures/residual_structure_scatter.png`
