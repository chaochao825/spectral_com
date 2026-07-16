# large_scale_hessian_pilot_20260714

This is an orchestration/evidence summary, not a claim that planned jobs ran.

- Config SHA-256: `ed7710e3343444b879c3c06771847708641e00e5c1c5760f6a2fcfdf7c56aab5`
- Jobs: 3
- Valid completed: 3
- Planned: 0
- Running (fail-closed): 0
- Failed (fail-closed): 0
- Invalid (fail-closed): 0

## Stage matrix

| Stage | Model | Availability | Evidence role | Seeds x rates | Tensor scope |
|---|---|---|---|---:|---|
| pythia70m_full_mlp_pilot | EleutherAI/pythia-70m (70M) | required | scalability_smoke; seed aggregation=false | 1 x 1 | full_mlp_weights (12 tensors) |
| opt125m_depth_mlp_pilot | facebook/opt-125m (125M) | optional | scalability_smoke; seed aggregation=false | 1 x 1 | five_depth_mlp_weights (10 tensors) |
| qwen3_06b_depth_mlp_pilot | Qwen/Qwen3-0.6B (0.6B) | optional | scalability_smoke; seed aggregation=false | 1 x 1 | five_depth_mlp_weights (10 tensors) |

`suite_manifest.json` records commands, hashes, actual token counts, runtime/GPU provenance and physical artifact scope for completed jobs. Optional jobs are never executed unless selected explicitly or `--include-optional` is passed.

Important: the current runner uses the same sequential data windows for every seed. Repeated seed jobs are scalability/reproducibility smoke checks only and must not be averaged or used as independent multi-seed evidence. A confirmatory label remains disabled until the numerical runner consumes the preregistered protocol manifest.
