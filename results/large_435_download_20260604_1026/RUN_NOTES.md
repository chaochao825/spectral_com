# Large 435 Run Notes

These artifacts are pretrained-only feasibility validation results from server 35.

- `qwen15_moe`: Qwen1.5-MoE-A2.7B, layers `0,12,23`, sites `resid_post,attn_out,mlp_out`, `4 x 128` tokens. The planned `64 x 128` run was attempted but was too slow on the shared server load, so this directory is a smaller feasibility artifact rather than the full planned sweep.
- `qwen57_a14b`: Qwen2-57B-A14B-Instruct, layers `0,14,27`, sites `resid_post,attn_out,mlp_out`, `16 x 128` tokens.
- `llama70_smoke`: Llama-2-70B-chat-hf gated smoke, layers `0,40,79`, sites `resid_post,attn_out,mlp_out`, `4 x 64` tokens.

All three result sets use `sample_space_reservoir` eigenspectrum estimation. Delta CSVs are marked unavailable because no random-init variants were run for these large models.
