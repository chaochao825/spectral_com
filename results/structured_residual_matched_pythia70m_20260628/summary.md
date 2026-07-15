# Structured Residual Matched-Memory Offline Test

- Model: `EleutherAI/pythia-70m`
- Dataset request: `wikitext/wikitext-2-raw-v1` split `validation`; backup `wikitext_2_raw`; allow_fallback=True
- Target layers: `attention_o,mlp_up,mlp_down`; max_layers=4
- Fixed Q base: methods=`rtn,sinq_like`, bits=4, group_size=128
- Block sizes: `16,32,64`
- Structured methods: `naive_block_circulant,norm_sorted_block_circulant,activation_clustered_block_circulant,random_permuted_block_circulant,monarch_like_two_block`
- Calibration only: calib_limit=8; activation_rows=128; sequence_length=128; batch_size=1
- Accounting: explicit row/column permutation metadata is counted as rows+cols parameters for permuted variants. Matched low-rank uses the largest rank not exceeding the structured parameter count; ceil-rank diagnostics are also saved.

## Decision

- Structured residual wins in 0/120 matched-memory comparisons against the non-overbudget matched low-rank floor rank. Criterion failed, so PPL was not run.
- Conservative decision: stop the current block-circulant / permutation / Monarch-like structured residual line for now.

## Best Structured Case

- Method: `norm_sorted_block_circulant`; layer `L0:dense_h_to_4h`; q `rtn` q4; block_size=64
- Structured activation error: 0.00338403
- Matched low-rank activation error: 0.00332573
- Delta structured-lowrank: +5.82964e-05

## Win Counts By Method

| Method | Wins | Rows | Best delta act |
|---|---:|---:|---:|
| activation_clustered_block_circulant | 0 | 24 | +8.76798e-05 |
| monarch_like_two_block | 0 | 24 | +9.33865e-05 |
| naive_block_circulant | 0 | 24 | +8.1558e-05 |
| norm_sorted_block_circulant | 0 | 24 | +5.82964e-05 |
| random_permuted_block_circulant | 0 | 24 | +8.43517e-05 |

## Files

- `matched_residual_metrics.csv`
- `figures/structured_vs_lowrank_scatter.png`
- `figures/activation_error_by_method.png`
- `run_config.json`
