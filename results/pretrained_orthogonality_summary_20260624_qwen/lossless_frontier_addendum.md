# Lossless Frontier Addendum

This addendum answers whether a lossless Q+S+R stack can provide more nominal compression benefit than any single lossless method under the same benchmark-drop criterion.

## Search Setup

Run artifact:

- `results/pretrained_orthogonality_pythia70m_lossless_frontier_4mods_factorized672_20260627`

Configuration:

- Model: `EleutherAI/pythia-70m`
- Selected modules: 4 first-layer attention/MLP linear modules
- Selected parameters: 3,145,728
- PPL evaluation: 254 tokens
- Lossless criterion: PPL benchmark drop `< 1%`
- Zero-shot frontier: not evaluated in this expanded run; this is a PPL frontier smoke test.
- Candidate count: 706 total, including all 34 single-method candidates and the 672 lowest-nominal-memory Q+S+R candidates from a 2,688-candidate Q+S+R grid.
- Memory proxy: Q uses `bits/16`, S uses kept-weight fraction, and R uses factorized SVD parameter ratio for each selected layer shape. Q+S+R multiplies these applicable factors.

Search ranges:

| Method family | Search range |
| --- | --- |
| Q-only | bits = 8, 6, 4, 3; q = RTN, rotated RTN |
| S-only | keep = .995, .99, .98, .95, .9, .8; s = Wanda, magnitude |
| R-only | rank = .995, .99, .98, .95, .9, .8, .5; r = whitened SVD, SVD |
| Q+S+R | same bit/keep/rank ranges; qsr and rqs orders; all listed Q/S/R methods; evaluated as the 672 lowest-nominal-memory candidates from the full 2,688-candidate stack grid |

## Frontier Table

| Family | Selected config | PPL drop | Pass | Nominal memory ratio | Nominal saving |
| --- | --- | ---: | --- | ---: | ---: |
| Q-only | rotated RTN, 4-bit | 0.0000% | yes | 0.250 | 75.0% |
| S-only | Wanda, keep=0.8 | 0.0000% | yes | 0.800 | 20.0% |
| R-only | whitened SVD, rank=0.5 | 0.0000% | yes | 0.667 | 33.3% |
| Q+S+R | rqs, rotated RTN, Wanda, whitened SVD, 4-bit, keep=0.8, rank=0.5 | 0.0000% | yes | 0.133 | 86.7% |

Machine-readable files:

- `metrics/lossless_frontier_candidates.csv`
- `metrics/lossless_frontier_summary.csv`
- `figures/lossless_frontier_summary.png`

## Interpretation

On this Pythia-70M 4-module PPL smoke run, the Q+S+R stack exceeds the best single-method lossless benefit by the nominal memory proxy. The best single method is Q-only at 4-bit, with memory ratio 0.25. The best passing stack has memory ratio 0.133, giving 1.875x lower selected-layer nominal memory than the best single method under the same `<1%` PPL-drop rule.

This is a stronger answer than the earlier conservative 8-bit/0.995/0.995 triple-stack result. The earlier result proved that all three operations could be applied with near-zero loss, but its extra benefit over Q-only was marginal. This frontier run shows that a more aggressive stack can still satisfy the same PPL lossless criterion in a small-model smoke setting.

## Limitations

The memory ratio is a nominal selected-layer proxy. It does not include packed-kernel overheads, metadata, runtime, non-selected model weights, or actual deployment storage. The R-only row uses factorized SVD storage, so rank=0.5 maps to a memory ratio of 0.667 for these layer shapes rather than 0.5; the Q+S+R memory ratio similarly uses `0.25 * 0.8 * 0.667 = 0.133`. The run also uses a short PPL window, evaluates the lowest-memory Q+S+R slice rather than all 2,688 stack candidates, and does not yet include the requested zero-shot frontier, so the result should be treated as evidence for the next experiment stage rather than a paper-strength final claim.

The stricter fixed-config comparison in `fair_benchmark_addendum.md` does not show Q+S+R superiority under a no-result-selection protocol. That result should take priority when discussing benchmark fairness.
