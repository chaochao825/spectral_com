# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/wangmeiqi/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42`; selected tensors: 6 MLP linears.
- Data: `dataset:wikitext` with content-disjoint calibration/evaluation source texts; fallback is disabled.
- Payload is codec-exact for selected tensor value streams: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + FP16 low-rank factors. Format-dependent model-container headers are shared/excluded.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 4.267626, PPL = 71.352079, tokens = 2032.

## Codec endpoints near target 0.258

| strategy | actual payload | strict match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.251221 | no | 0.010077 | n/a | n/a | n/a | n/a | 15.645911 |
| Q_global_scale | 0.251221 | no | 0.009951 | n/a | n/a | n/a | n/a | 15.285018 |
| Q_block_scale | 0.257812 | yes | 0.005767 | 0.000033529 | n/a | n/a | n/a | 9.403924 |
| Q+S | 0.257999 | yes | 0.006803 | 0.000024767 | n/a | 0.639077 | n/a | 9.293567 |
| Q+S_OBS | 0.257999 | yes | 0.005857 | 0.000031918 | n/a | 0.837512 | n/a | 7.160379 |
| Q+L | 0.257324 | yes | 0.004633 | 0.000045733 | n/a | n/a | 1.080367 | 6.761818 |
| Q+S+L_QL_budget | 0.257324 | yes | 0.004615 | 0.000045887 | 0.022990 | 0.271154 | 0.812915 | 6.549810 |
| Q+S+L_QL_budget_component_scale | 0.257324 | yes | 0.004581 | 0.000046168 | 0.023143 | 0.271639 | 0.811239 | 6.440811 |
| Q+S+L | 0.257418 | yes | 0.004607 | 0.000045258 | 0.029123 | 0.279599 | 0.806064 | 6.578466 |
| Q+S_OBS+L | 0.257418 | yes | 0.004637 | 0.000045007 | 0.012450 | 0.413623 | 0.672506 | 6.555827 |
| Q+S+L_component_scale | 0.257418 | yes | 0.004574 | 0.000045535 | 0.029318 | 0.279994 | 0.804375 | 6.476782 |

### Exact equal-bit Q+L control

`Q+S+L_QL_budget_component_scale` is capped per layer by the exact Q+L codec payload. Aggregate bit delta versus Q+L = `0`; PPL-delta improvement = `0.321007`; normalized-Hessian-cost improvement = `0.000051782`. A positive improvement means the combination wins without using more stored bits.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|
| Q | 1.000 | yes | 0.198259 | 0.998366 |
| Q_block_scale | 1.000 | yes | 0.123806 | 0.999510 |
| Q+S_OBS | 1.000 | yes | 0.095631 | 0.997315 |
| Q+L | 1.000 | yes | 0.090541 | 0.998024 |
| Q+S+L_QL_budget_component_scale | 1.000 | yes | 0.086424 | 0.998477 |
| Q+S+L_component_scale | 1.000 | yes | 0.086886 | 0.998537 |

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, and `comfort_sweep.csv` for the measured loss landscape.
