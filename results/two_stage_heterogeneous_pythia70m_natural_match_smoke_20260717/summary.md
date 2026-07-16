# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/wangmeiqi/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42`; selected tensors: 1 MLP linears.
- Data: `dataset:wikitext|dataset:wikitext|dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + packed 4/8-bit or FP16 low-rank factors and their scales, including the manifest, descriptors and 64-byte alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 4.594140, PPL = 98.903002, tokens = 62.

## Codec endpoints near target 0.300

| strategy | logical value ratio | artifact/reference ratio | file bytes | logical target match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.251953 | 0.252471 | 529664 | no | 0.005572 | n/a | n/a | n/a | n/a | -2.096750 |
| Q_global_scale | 0.251953 | 0.252471 | 529664 | no | 0.005524 | n/a | n/a | n/a | n/a | -2.074257 |
| Q+S | 0.299999 | 0.301103 | 631690 | yes | 0.002516 | 0.000002997 | n/a | 1.101663 | n/a | -2.159151 |
| Q+S_OBS | 0.299999 | 0.301103 | 631690 | yes | 0.001799 | 0.000003701 | n/a | 1.354452 | n/a | 2.474376 |
| Q+L | 0.298340 | 0.299207 | 627712 | yes | 0.000979 | 0.000004666 | n/a | n/a | 1.648840 | -4.804508 |
| Q+S_OBS_global | 0.292364 | 0.299207 | 627712 | no | 0.001292 | 0.000004991 | n/a | 1.283468 | n/a | 0.910280 |
| Q+L_global | 0.298340 | 0.299207 | 627712 | yes | 0.000979 | 0.000004666 | n/a | n/a | 1.648840 | -4.804508 |
| Q+S_OBS_or_L_global | 0.298340 | 0.299207 | 627712 | yes | 0.000979 | 0.000004666 | n/a | n/a | 1.648840 | -4.804508 |
| Q+S+L_QL_budget | 0.298340 | 0.299207 | 627712 | yes | 0.000979 | 0.000004666 | n/a | n/a | 1.648840 | -4.804508 |
| Q+S+L_QL_budget_component_scale | 0.298340 | 0.299207 | 627712 | yes | 0.000978 | 0.000004667 | n/a | n/a | 1.647948 | -4.278456 |
| Q+S+L | 0.299999 | 0.301495 | 632512 | yes | 0.001524 | 0.000003971 | -0.004190 | 1.048361 | 0.406299 | 0.319650 |
| Q+S_OBS+L | 0.299999 | 0.301495 | 632512 | yes | 0.001359 | 0.000004133 | 0.024391 | 1.186473 | 0.341022 | -0.203840 |
| Q+S+L_component_scale | 0.299999 | 0.301495 | 632512 | yes | 0.001521 | 0.000003973 | -0.004189 | 1.045821 | 0.406547 | 1.146063 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is selected from enumerated per-layer support/rank budget bands by an aggregate additive frontier, then checked against the complete natural Q+L file-byte cap. Aggregate value-stream bit delta versus Q+L = `0`; serialized file-byte delta = `0`; natural file-byte delta = `0`; PPL-delta improvement = `-0.526052`; normalized-Hessian-cost improvement = `0.000000943`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
