# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/zengqiuhao/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`; selected tensors: 10 MLP linears.
- Data: `dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + FP16 low-rank factors, including the manifest, descriptors and 64-byte alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 3.395705, PPL = 29.835670, tokens = 1016.

## Codec endpoints near target 0.258

| strategy | logical value ratio | artifact/reference ratio | file bytes | strict match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.250651 | 0.250792 | 15779648 | no | 0.022950 | n/a | n/a | n/a | n/a | 3.560281 |
| Q_global_scale | 0.250651 | 0.250792 | 15779648 | no | 0.022036 | n/a | n/a | n/a | n/a | 3.811929 |
| Q_block_scale | 0.257812 | 0.257954 | 16230272 | yes | 0.006622 | 0.000714033 | n/a | n/a | n/a | 2.462043 |
| Q+S | 0.257999 | 0.258334 | 16254218 | yes | 0.006692 | 0.000692885 | n/a | 1.417331 | n/a | 1.279711 |
| Q+S_OBS | 0.257999 | 0.258334 | 16254218 | yes | 0.003865 | 0.000813366 | n/a | 1.663259 | n/a | 0.785565 |
| Q+L | 0.257161 | 0.257415 | 16196352 | yes | 0.004139 | 0.000904885 | n/a | n/a | 1.639071 | 0.683324 |
| Q+S+L_QL_budget | 0.256969 | 0.257415 | 16196352 | yes | 0.004441 | 0.000917496 | 0.022045 | 0.532573 | 1.080783 | 0.917718 |
| Q+S+L_QL_budget_component_scale | 0.256969 | 0.257415 | 16196352 | yes | 0.004401 | 0.000919473 | 0.022057 | 0.526680 | 1.086285 | 0.905489 |
| Q+S+L | 0.257999 | 0.258446 | 16261248 | yes | 0.004151 | 0.000801177 | 0.015464 | 0.607247 | 1.031285 | 0.675767 |
| Q+S_OBS+L | 0.257999 | 0.258446 | 16261248 | yes | 0.004186 | 0.000799664 | 0.011577 | 0.899136 | 0.745436 | 0.675418 |
| Q+S+L_component_scale | 0.257999 | 0.258446 | 16261248 | yes | 0.004113 | 0.000802789 | 0.015468 | 0.602555 | 1.035970 | 0.659735 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is capped per layer by the Q+L codec budget. Aggregate value-stream bit delta versus Q+L = `-96960`; serialized file-byte delta = `0`; PPL-delta improvement = `-0.222165`; normalized-Hessian-cost improvement = `-0.000262325`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|
| Q | 1.000 | yes | 0.112730 | 0.999176 |
| Q_block_scale | 1.000 | yes | 0.079292 | 0.999981 |
| Q+S_OBS | 1.000 | yes | 0.025989 | 0.983374 |
| Q+L | 1.000 | yes | 0.022645 | 0.974553 |
| Q+S+L_QL_budget_component_scale | 1.000 | yes | 0.029898 | 0.994422 |
| Q+S+L_component_scale | 0.375 | yes | 0.021871 | 0.965663 |

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
