# large_model_interaction_aware_v4_20260717

This is an orchestration/evidence summary, not a claim that planned jobs ran.

- Config SHA-256: `330127e39f8ee21af6758f3df1cc18ecb01d2db929ca678c8e3e690e4b61e317`
- Jobs: 9
- Valid completed: 1
- Planned: 8
- Running (fail-closed): 0
- Failed (fail-closed): 0
- Invalid (fail-closed): 0

## Stage matrix

| Stage | Model | Availability | Evidence role | Seeds x rates | Tensor scope |
|---|---|---|---|---:|---|
| s0_qwen_two_stage_sentinel | Qwen/Qwen2.5-3B-Instruct (3B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | layer0_gate_two_stage (1 tensors) |
| s0_llama_two_stage_sentinel | meta-llama/Llama-2-7b-hf (7B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | layer0_gate_two_stage (1 tensors) |
| s0_mistral_two_stage_sentinel | mistralai/Mistral-7B-v0.1 (7B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | layer0_gate_two_stage (1 tensors) |
| s1_qwen_three_depth_interactions | Qwen/Qwen2.5-3B-Instruct (3B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | three_depth_gate_up (6 tensors) |
| s2_llama_three_depth_interactions | meta-llama/Llama-2-7b-hf (7B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | three_depth_gate_up (6 tensors) |
| s2_mistral_three_depth_interactions | mistralai/Mistral-7B-v0.1 (7B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | three_depth_gate_up (6 tensors) |
| s3_qwen_full_mlp_feasibility | Qwen/Qwen2.5-3B-Instruct (3B) | required | scalability_smoke; seed aggregation=false | 1 x 1 | all_mlp_projections (108 tensors) |
| s4_llama_full_mlp_feasibility | meta-llama/Llama-2-7b-hf (7B) | optional | scalability_smoke; seed aggregation=false | 1 x 1 | all_mlp_projections (96 tensors) |
| s4_mistral_full_mlp_feasibility | mistralai/Mistral-7B-v0.1 (7B) | optional | scalability_smoke; seed aggregation=false | 1 x 1 | all_mlp_projections (96 tensors) |

`suite_manifest.json` records commands, hashes, actual token counts, runtime/GPU provenance and physical artifact scope for completed jobs. Optional jobs are never executed unless selected explicitly or `--include-optional` is passed.

Important: the current runner uses the same sequential data windows for every seed. Repeated seed jobs are scalability/reproducibility smoke checks only and must not be averaged or used as independent multi-seed evidence. A confirmatory label remains disabled until the numerical runner consumes the preregistered protocol manifest.
