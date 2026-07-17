# Pretrained exact-rate Hessian repair probe

## Scope and design expectations

- Model: `/home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`; selected tensors: 1 MLP linears.
- Data: `dataset:wikitext|dataset:wikitext|dataset:wikitext` with independent train calibration, validation allocation selection and final-test splits; exact duplicate content is removed across roles and test is reserved until validation selection completes.
- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + FP16 scales + FP16 sparse values + fixed-width CSR support + packed 4/8-bit or FP16 low-rank factors with their stored scales, including the manifest, descriptors and 64-byte alignment. This is byte-audit evidence, not a production inference backend.
- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.
- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.

## Held-out baseline

NLL = 2.551900, PPL = 12.831458, tokens = 254.

## Codec endpoints near target 0.300

| strategy | logical value ratio | artifact/reference ratio | file bytes | logical target match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| Q | 0.250488 | 0.250513 | 11295488 | no | 0.010507 | n/a | n/a | n/a | n/a | 0.005184 |
| Q_global_scale | 0.250488 | 0.250513 | 11295488 | no | 0.010401 | n/a | n/a | n/a | n/a | 0.008152 |
| Q+S | 0.300000 | 0.300049 | 13529072 | yes | 0.007299 | 0.000000197 | n/a | 0.608237 | n/a | 0.005286 |
| Q+S_OBS | 0.300000 | 0.300049 | 13529072 | yes | 0.004996 | 0.000000338 | n/a | 1.049140 | n/a | 0.008210 |
| Q+L | 0.299714 | 0.299753 | 13515712 | yes | 0.003336 | 0.000000443 | n/a | n/a | 1.365081 | -0.001742 |
| Q+S_OBS_global | 0.297481 | 0.299753 | 13515712 | yes | 0.003098 | 0.000000479 | n/a | 0.939147 | n/a | -0.020189 |
| Q+L_global | 0.299509 | 0.299753 | 13515712 | yes | 0.002119 | 0.000000520 | n/a | n/a | 1.274302 | -0.018488 |
| Q+S_OBS_or_L_global | 0.299509 | 0.299753 | 13515712 | yes | 0.002119 | 0.000000520 | n/a | n/a | 1.274302 | -0.018488 |
| Q+S+L_QL_budget | 0.299509 | 0.299753 | 13515712 | yes | 0.002119 | 0.000000520 | n/a | n/a | 1.274302 | -0.018488 |
| Q+S+L_QL_budget_component_scale | 0.299509 | 0.299753 | 13515712 | yes | 0.002115 | 0.000000520 | n/a | n/a | 1.273095 | -0.010758 |
| Q+S+L | 0.300000 | 0.300065 | 13529792 | yes | 0.004856 | 0.000000347 | 0.003124 | 0.474775 | 0.600725 | 0.009789 |
| Q+S_OBS+L | 0.300000 | 0.300065 | 13529792 | yes | 0.004526 | 0.000000367 | 0.067168 | 0.852266 | 0.319435 | 0.026657 |
| Q+S+L_component_scale | 0.300000 | 0.300065 | 13529792 | yes | 0.004832 | 0.000000348 | 0.003123 | 0.474058 | 0.599463 | 0.006224 |

### Q+L fixed-rate control

`Q+S+L_QL_budget_component_scale` is selected from enumerated per-layer support/rank budget bands by an aggregate additive frontier, then checked against the complete natural Q+L file-byte cap. Aggregate value-stream bit delta versus Q+L = `-73728`; serialized file-byte delta = `0`; natural file-byte delta = `-9216`; PPL-delta improvement = `0.009016`; normalized-Hessian-cost improvement = `0.001220926`. A positive improvement means the combination wins without using more measured storage only when the reported file-byte delta is non-positive.

`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.

`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.

## Comfort-zone / loss-landscape check

Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.

| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |
|---|---:|:---:|---:|---:|

## Theory–experiment contract

For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.

See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.
