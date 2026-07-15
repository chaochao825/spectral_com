# Result Archive

This directory is versioned because the CSV, JSON, Markdown, PNG, PDF, and small tensor payloads are part of the research record. It does not contain model weights or datasets.

## Main groups

| Group | Directories | Purpose |
|---|---|---|
| Activation-spectrum MVP | `mvp_real` | Real-text pretrained/random activation spectra and token dynamics. |
| Large-model feasibility | `large_435_download_20260604_1026` | Qwen1.5-MoE, Qwen2-57B-A14B, and Llama-2-70B pretrained-only runs from server 35. |
| Structured Qwen2.5 | `structured_qwen25_*` | Multi-phase weight approximation, activation reconstruction, replacement, adapter, rotation, and quantization smokes. |
| CUDA benchmark | `cuda_benchmark_20260610` | Structured operation timing and plotting artifacts. |
| Orthogonality MVP | `compression_orthogonality_mvp_*` | Synthetic/methodology iterations for Hessian cosine, additivity, and loss landscapes. Later versions supersede earlier ones but all are retained for provenance. |
| Pretrained orthogonality | `pretrained_orthogonality_*` | Pythia and Qwen scale/strength sweeps, summaries, fair-budget variants, and limited downstream checks. |
| Structured residual tests | `oasr_structured_residual_*`, `structured_residual_matched_*` | OASR and matched residual decompositions. |
| Residual-stack validation | `residual_stack_validate_*` | Pythia and Qwen2-7B quantization/sparse/low-rank residual-stack experiments. |
| Cross-method summary | `compare_7b_dam_residual_stack_20260707` | Matched-budget comparison tables, figures, and the July 2026 interpretation report. |

## Qualification notes

- `large_435_download_20260604_1026/RUN_NOTES.md` is the source of truth for large-model sample sizes.
- Empty large-model delta CSVs are intentional for pretrained-only runs.
- Directories with `smoke` in their name test feasibility and plumbing; they are not full benchmark results.
- Multiple versioned orthogonality directories preserve the development history. Use `compression_orthogonality_mvp_20260623_v7` for the latest MVP audit.
- The July Qwen2-7B report is a layer-subset result with a small evaluation set. Negative PPL deltas there are encouraging screening signals, not population-level improvements.
- The two Qwen2-7B run directories have different dense PPL baselines. Only within-run deltas are interpretable; the absolute PPL values are not a matched cross-row comparison.
- Historical reports may contain absolute server paths from the run environment. Those paths are provenance, not repository setup requirements.
- Historical environment capture is incomplete. `environments/validation_210_20260716.yaml` documents only the current sync-time tests and synthetic smoke, not the original experiment environments.

See `docs/methods_and_results.md` for the current evidence summary and interpretation limits.
