# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/wangmeiqi/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42`; selected tensors: 6 MLP linears.
- Data: `dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + FP16 low-rank factors, including the manifest, descriptors and alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 4.267632, PPL = 71.352501, tokens = 2032.

## Codec endpoints near target 0.258

| strategy | logical value ratio | artifact/reference ratio | file bytes | strict match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.251221 | 0.251672 | 3167552 | no | 0.010077 | n/a | n/a | n/a | n/a | 15.643586 |
| Q_global_scale | 0.251221 | 0.251672 | 3167552 | no | 0.009951 | n/a | n/a | n/a | n/a | 15.283671 |
| Q_block_scale | 0.257812 | 0.258262 | 3250496 | yes | 0.005767 | 0.000033529 | n/a | n/a | n/a | 9.403732 |
| Q+S | 0.257999 | 0.259060 | 3260546 | yes | 0.006803 | 0.000024767 | n/a | 0.639078 | n/a | 9.293155 |
| Q+S_OBS | 0.257999 | 0.259060 | 3260546 | yes | 0.005857 | 0.000031918 | n/a | 0.837513 | n/a | 7.156156 |
| Q+L | 0.257324 | 0.258130 | 3248832 | yes | 0.004633 | 0.000045733 | n/a | n/a | 1.080513 | 6.761523 |
| Q+S+L_QL_budget | 0.255799 | 0.258130 | 3248832 | yes | 0.005112 | 0.000055603 | 0.030086 | 0.343253 | 0.642033 | 7.419177 |
| Q+S+L_QL_budget_component_scale | 0.255799 | 0.258130 | 3248832 | yes | 0.005071 | 0.000056057 | 0.030354 | 0.344159 | 0.640037 | 7.319517 |
| Q+S+L | 0.257418 | 0.258531 | 3253888 | yes | 0.004607 | 0.000045258 | 0.029119 | 0.279602 | 0.806220 | 6.578220 |
| Q+S_OBS+L | 0.257418 | 0.258531 | 3253888 | yes | 0.004637 | 0.000045007 | 0.012449 | 0.413631 | 0.672652 | 6.555837 |
| Q+S+L_component_scale | 0.257418 | 0.258531 | 3253888 | yes | 0.004574 | 0.000045535 | 0.029321 | 0.279994 | 0.804377 | 6.475780 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is capped per layer by the Q+L codec budget. Aggregate value-stream bit delta versus Q+L = `-153504`; serialized file-byte delta = `0`; PPL-delta improvement = `-0.557993`; normalized-Hessian-cost improvement = `-0.000438176`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|
| Q | 1.000 | yes | 0.198231 | 0.998206 |
| Q_block_scale | 1.000 | yes | 0.123803 | 0.999460 |
| Q+S_OBS | 1.000 | yes | 0.095577 | 0.997013 |
| Q+L | 1.000 | yes | 0.090537 | 0.997829 |
| Q+S+L_QL_budget_component_scale | 1.000 | yes | 0.097655 | 0.999250 |
| Q+S+L_component_scale | 1.000 | yes | 0.086872 | 0.998385 |

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
