# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `EleutherAI/pythia-70m`; selected tensors: 12 MLP linears.
- Data: `dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + FP16 low-rank factors, including the manifest, descriptors and 64-byte alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 4.523897, PPL = 92.194141, tokens = 1016.

## Codec endpoints near target 0.258

| strategy | logical value ratio | artifact/reference ratio | file bytes | strict match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.251221 | 0.251662 | 6334784 | no | 0.009612 | n/a | n/a | n/a | n/a | 43.127800 |
| Q_global_scale | 0.251221 | 0.251662 | 6334784 | no | 0.009463 | n/a | n/a | n/a | n/a | 42.574490 |
| Q_block_scale | 0.257812 | 0.258252 | 6500672 | yes | 0.005975 | 0.000028759 | n/a | n/a | n/a | 30.052751 |
| Q+S | 0.257999 | 0.259056 | 6520898 | yes | 0.007042 | 0.000019758 | n/a | 0.527770 | n/a | 26.561664 |
| Q+S_OBS | 0.257999 | 0.259056 | 6520898 | yes | 0.006112 | 0.000026909 | n/a | 0.728298 | n/a | 23.533009 |
| Q+L | 0.257324 | 0.258120 | 6497344 | yes | 0.004760 | 0.000041431 | n/a | n/a | 1.009781 | 19.994095 |
| Q+S+L_QL_budget | 0.255799 | 0.258120 | 6497344 | yes | 0.005298 | 0.000049104 | 0.026128 | 0.211624 | 0.685846 | 21.094390 |
| Q+S+L_QL_budget_component_scale | 0.255799 | 0.258120 | 6497344 | yes | 0.005258 | 0.000049561 | 0.026344 | 0.212914 | 0.681180 | 20.603442 |
| Q+S+L | 0.257418 | 0.258522 | 6507456 | yes | 0.004746 | 0.000040920 | 0.024741 | 0.186030 | 0.826645 | 20.732756 |
| Q+S_OBS+L | 0.257418 | 0.258522 | 6507456 | yes | 0.004762 | 0.000040788 | 0.011852 | 0.282690 | 0.732068 | 20.276093 |
| Q+S+L_component_scale | 0.257418 | 0.258522 | 6507456 | yes | 0.004713 | 0.000041197 | 0.024907 | 0.186642 | 0.822715 | 20.333007 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is capped per layer by the Q+L codec budget. Aggregate value-stream bit delta versus Q+L = `-307008`; serialized file-byte delta = `0`; PPL-delta improvement = `-0.609348`; normalized-Hessian-cost improvement = `-0.000498063`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|
| Q | 1.000 | yes | 0.383760 | 0.999252 |
| Q_block_scale | 1.000 | yes | 0.282146 | 0.999741 |
| Q+S_OBS | 1.000 | yes | 0.227339 | 0.999394 |
| Q+L | 1.000 | yes | 0.196282 | 0.998953 |
| Q+S+L_QL_budget_component_scale | 1.000 | yes | 0.201698 | 0.999392 |
| Q+S+L_component_scale | 1.000 | yes | 0.199298 | 0.999808 |

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
