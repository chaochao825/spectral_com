# Publishing Note

This directory is a compact, publishable audit record for the Qwen2.5-3B
one-tensor interaction-aware allocator sentinel.

The canonical run completed on server 210 from target commit `ea77d126`:

```text
/home/wangmeiqi/codex_results/large_model_interaction_aware_v4_ea77d12_20260717
```

The 14 raw `.hrc` codec artifacts, totaling `216,422,240` bytes, are
intentionally not committed. They remain under the canonical run's
`jobs/s0_qwen_two_stage_sentinel__seed17__rate0p300/artifacts` directory.
`artifact_manifest.json` and `artifact_payloads.csv` preserve filenames,
SHA-256 digests, natural and padded byte counts, stream composition, and
round-trip checks. Paths under `artifacts/` are provenance references, not
files available in this Git checkout.

The evidence gate passed:

- suite status: `completed_valid`
- exact-natural no-joint search: `true`
- calibration/selection/test identical-text overlap counts: `0/0/0`
- allocator source: `global_exact_canonical_layout_pareto_frontier`
- final source: `validation_nll_rerank_of_exact_proxy_top_k`
- candidate quantizer order: `symmetric_mse_clip`, then `symmetric_rtn`

The result is negative for same-layer interaction:

- QSL natural bytes: `13,506,496`
- exact-natural no-joint bytes: `13,506,496`
- padded physical bytes for each endpoint: `13,515,712`
- QSL final-test NLL: `2.5504579318789986`
- no-joint final-test NLL: `2.5504579318789986`
- QSL NLL gain over no-joint: `0.0`
- same-layer joint-value claim: `false`

Both endpoints are the same W4/group-128 symmetric-RTN base plus rank-72
FP16 low-rank repair, with no active sparse component. This establishes
3B-model execution feasibility for the two-stage allocator, not a benefit
from joint S+L repair.

The run uses one layer-0 `gate_proj`, calibration/selection/evaluation limits
of `4/2/2`, and sequence length 128. It is a scalability smoke, not full-MLP
or population-level evidence.

The versioned resource trace reports `child_max_rss_gib=10.5188`. Its GPU
samples are whole-device `nvidia-smi` readings and may include other processes;
`peak_gpu_memory_mib` must not be interpreted as process-attributable peak
VRAM or as a compression-method runtime claim.
