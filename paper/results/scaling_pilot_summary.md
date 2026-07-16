# Verified three-job scalability smoke

This report is generated from the fail-closed scaling aggregate.  It contains three 
separate seed-17 observations at selected-weight artifact target 0.258.  It is not a 
multi-seed result, a cross-model leaderboard, a model-size trend, or a whole-model 
compression claim.

## Final-byte-equal conservative combination and Hessian geometry

| Model / scope | Tensors | QL=strict bytes | Strict-QL NLL | 8-window wins | rho_SL | rho_QS | rho_QL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pythia-70M / full MLP | 12 | 6,497,344 | +0.005417 | 3/8 | +0.0263 | -0.3340 | -0.5878 |
| OPT-125M / 5-depth MLP | 10 | 12,158,976 | +0.000931 | 5/8 | +0.0871 | -0.3670 | -0.6106 |
| Qwen3-0.6B / 5-depth MLP | 10 | 16,196,352 | +0.007253 | 1/8 | +0.0221 | -0.5211 | -0.7426 |

`Strict-QL NLL` is the signed held-out endpoint difference; negative favors the 
strict component-scaled Q+S+L endpoint.  Final byte equality is produced by tail 
padding: the strict natural files leave 31,360/640/1,024 bytes unused for 
Pythia/OPT/Qwen, so these are conservative candidates rather than 
budget-exhausted frontiers.  The window count is descriptive over the 
same eight fixed windows and is not a significance test.  Values with 
`abs(rho_SL) <= 0.1` satisfy the declared near-orthogonality diagnostic; negative 
Q-S or Q-L values are cancellation, not orthogonality.

## Within-model parameter-utilization controls

Every numeric difference below is `left - right`; negative NLL favors the left 
endpoint.  Byte differences charge the complete aligned selected-tensor research 
artifact.

| Model | Comparison | Left / right | Byte difference | NLL difference | Same bytes? |
|---|---|---|---:|---:|---|
| Pythia-70M | folded global scale | `Q_global_scale` / `Q` | +0 | -0.004097 | true |
| Pythia-70M | block scale | `Q_block_scale` / `Q` | +165,888 | -0.101614 | false |
| Pythia-70M | fixed-support OBS values | `Q+S_OBS` / `Q+S` | +0 | -0.025834 | true |
| Pythia-70M | strict scaled composition | `Q+S+L_QL_budget_component_scale` / `Q+L` | +0 | +0.005417 | true |
| OPT-125M | folded global scale | `Q_global_scale` / `Q` | +0 | -0.013126 | true |
| OPT-125M | block scale | `Q_block_scale` / `Q` | +311,872 | -0.069025 | false |
| OPT-125M | fixed-support OBS values | `Q+S_OBS` / `Q+S` | +0 | -0.008144 | true |
| OPT-125M | strict scaled composition | `Q+S+L_QL_budget_component_scale` / `Q+L` | +0 | +0.000931 | true |
| Qwen3-0.6B | folded global scale | `Q_global_scale` / `Q` | +0 | +0.007507 | true |
| Qwen3-0.6B | block scale | `Q_block_scale` / `Q` | +450,624 | -0.033438 | false |
| Qwen3-0.6B | fixed-support OBS values | `Q+S_OBS` / `Q+S` | +0 | -0.016009 | true |
| Qwen3-0.6B | strict scaled composition | `Q+S+L_QL_budget_component_scale` / `Q+L` | +0 | +0.007253 | true |

Zero-byte scale or OBS improvements are reported as direct recovery rather than 
infinite recovery-per-byte.  Positive-cost controls can be compared within a model 
using `scaling_pilot_endpoints.csv`, which reports added exact bits per selected 
parameter and held-out NLL recovery from Q.

## Theory-to-experiment boundary

- The signed rho values test local additivity in the declared PSD activation-Gram 
  geometry; they do not certify held-out accuracy.
- The strict equal-file-byte endpoint difference is the realized exchange test.  
  The experiment does not separately identify every term in the theoretical 
  `P_A + Gamma_A` decomposition.
- Six 13-point paths per model connect the local Taylor diagnostic to epsilon=1.  
  Fits use only epsilon <= 0.125; the remainder is labelled extrapolation.
- External methods remain in A0/A1/B/C training-dependence lanes.  Literature 
  numbers are not placed on these axes without the same checkpoint, tensor scope, 
  data, serializer, and exported-state accounting.
