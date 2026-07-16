# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/spco/base-2-bitnet/.hf_cache/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6`; selected tensors: 10 MLP linears.
- Data: `dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + FP16 low-rank factors, including the manifest, descriptors and 64-byte alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 4.025780, PPL = 56.023997, tokens = 1016.

## Codec endpoints near target 0.258

| strategy | logical value ratio | artifact/reference ratio | file bytes | strict match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.250814 | 0.250999 | 11844800 | no | 0.005848 | n/a | n/a | n/a | n/a | 7.744192 |
| Q_global_scale | 0.250814 | 0.250999 | 11844800 | no | 0.005762 | n/a | n/a | n/a | n/a | 6.912630 |
| Q_block_scale | 0.257422 | 0.257608 | 12156672 | yes | 0.003368 | 0.000002801 | n/a | n/a | n/a | 3.491039 |
| Q+S | 0.258000 | 0.258422 | 12195064 | yes | 0.004202 | 0.000001709 | n/a | 0.519537 | n/a | 6.390365 |
| Q+S_OBS | 0.258000 | 0.258422 | 12195064 | yes | 0.003345 | 0.000002600 | n/a | 0.856089 | n/a | 5.884153 |
| Q+L | 0.257324 | 0.257657 | 12158976 | yes | 0.002789 | 0.000003506 | n/a | n/a | 1.045874 | 4.272078 |
| Q+S+L_QL_budget | 0.257078 | 0.257657 | 12158976 | yes | 0.003090 | 0.000003285 | 0.084749 | 0.223225 | 0.716808 | 4.835379 |
| Q+S+L_QL_budget_component_scale | 0.257078 | 0.257657 | 12158976 | yes | 0.003072 | 0.000003307 | 0.087149 | 0.231840 | 0.712114 | 4.328213 |
| Q+S+L | 0.258000 | 0.258568 | 12201984 | yes | 0.003051 | 0.000002905 | 0.079510 | 0.247463 | 0.706126 | 4.716053 |
| Q+S_OBS+L | 0.258000 | 0.258568 | 12201984 | yes | 0.003018 | 0.000002940 | 0.020728 | 0.559708 | 0.418067 | 4.692476 |
| Q+S+L_component_scale | 0.258000 | 0.258568 | 12201984 | yes | 0.003034 | 0.000002923 | 0.081729 | 0.254982 | 0.702051 | 4.224219 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is capped per layer by the Q+L codec budget. Aggregate value-stream bit delta versus Q+L = `-92864`; serialized file-byte delta = `0`; PPL-delta improvement = `-0.056134`; normalized-Hessian-cost improvement = `-0.000282776`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|
| Q | 1.000 | yes | 0.129474 | 0.998619 |
| Q_block_scale | 0.125 | no | 0.060449 | 0.995701 |
| Q+S_OBS | 1.000 | yes | 0.099872 | 0.997057 |
| Q+L | 1.000 | yes | 0.073487 | 0.999744 |
| Q+S+L_QL_budget_component_scale | 1.000 | yes | 0.074417 | 0.999816 |
| Q+S+L_component_scale | 1.000 | yes | 0.072693 | 0.999753 |

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
