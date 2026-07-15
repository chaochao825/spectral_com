# Current Methods and Results

Updated: 2026-07-16

## 1. Activation spectral pipeline

For a selected model, layer, and hook site, token activations are streamed into a centered covariance estimator. The implementation avoids retaining the full activation tensor and accumulates covariance in float64 using Welford/Chan updates. The output spectrum is summarized with:

- normalized eigenvalues and explained variance at fixed ranks;
- participation ratio and entropy effective rank;
- spectral entropy and condition number;
- anisotropy and sample outlier scores;
- power-law slope, confidence interval, fit range, and fit quality;
- token-lag PCA autocorrelation and DMD summaries when enabled.

The main sites are `resid_post`, `attn_out`, and `mlp_out`. Comparison figures normalize each spectrum by its eigenvalue sum and use shared axes. Pretrained/random delta tables are emitted only when both variants exist in a matched model/layer/site group.

The large-model loader accepts a local model path and supports `local_files_only`, `low_cpu_mem_usage`, explicit `torch_dtype`, and `device_map=auto`. With automatic device placement, inputs are sent to the embedding device rather than moving the dispatched model again.

## 2. Structured compression pipeline

The structured track separates several questions that should not be conflated:

1. **Weight geometry:** singular values, energy ranks, effective/stable rank, channel outliers, and residual concentration.
2. **Equal-budget approximation:** low-rank, block-circulant, and Monarch-like candidates under the same nominal parameter or memory budget.
3. **Functional error:** activation reconstruction, perplexity, and limited zero-shot checks after replacing selected linear layers.
4. **Residual composition:** quantize first, then model the remaining error as sparse and low-rank components.
5. **Interaction geometry:** parameter cosine, Hessian-weighted cosine, empirical additivity error, order gap, and loss-landscape slices.
6. **Adaptation/quantization:** structured adapters, LoRA-like baselines, channel rotations, and direct/structured quantization error.

The current residual-stack hypothesis is:

```text
R_q = W - Q(W)
R_q ~= S_res + L_res
W_hat = Q(W) + S_res + L_res
```

Candidate selection is evaluated separately from candidate construction. Current evidence shows that activation/Hessian proxies can reject poor candidates, but are not yet reliable enough to be the final perplexity selector.

## 3. Result summary

### Real-data activation MVP

`results/mvp_real/report.md` reports mean fitted alpha `1.826`, mean participation ratio `42.99`, and mean effective rank `82.92` over its collected rows. In that run, attention-output effective rank averaged `31.66`, compared with `108.4` for FFN output and `108.8` for the residual stream. The pretrained/random alpha difference was small (`-0.03623`), which explains why separate auto-scaled log-log figures looked nearly identical.

These values describe one MVP sample and are not universal model properties.

### Large-model feasibility on server 35

`results/large_435_download_20260604_1026` contains pretrained-only results for:

| Model | Layers | Sites | Sample qualification |
|---|---|---|---|
| Qwen1.5-MoE-A2.7B | 0, 12, 23 | residual, attention, MLP | `4 x 128`; reduced feasibility artifact because the planned larger run was too slow under shared load. |
| Qwen2-57B-A14B-Instruct | 0, 14, 27 | residual, attention, MLP | `16 x 128`; planned validation setting completed. |
| Llama-2-70B-chat-hf | 0, 40, 79 | residual, attention, MLP | `4 x 64`; gated smoke completed. |

All three use sample-space reservoir eigenspectrum estimation. Pretrained/random delta tables are marked unavailable by design because random-init variants were not run at this scale.

### Structured Qwen2.5-1.5B smoke

The archived report gives a dense baseline perplexity of `7.649`. For the tested `down_proj` replacement, the best compressed row had perplexity `9.875`; low-rank was the best tested weight-error structure. The best reported adapter smoke row was a structured adapter with perplexity `6.715`, but this is not a controlled broad benchmark and should not be read as a general superiority claim.

### Residual-stack evidence

The July Qwen2-7B comparison used a matched nominal memory ratio of at most `0.258` on selected layers:

| Setting | Dense PPL | Best tested same-budget method | Selected deltas |
|---|---:|---|---|
| 3 attention `o_proj` modules | 67.1155 | `Q+L` | `Q+L -0.537`, `Q+S+L +0.165` |
| 6 attention+MLP modules | 45.5759 | `Q+S+L` | `Q+L +0.029`, `Q+S+L -0.281` |

The two rows came from separate run directories and their dense baselines differ substantially. The archived metadata is insufficient to prove that evaluation examples, revisions, seeds, and all scoring settings were identical. Therefore, compare each method only with the dense baseline in the same row; do not compare the two absolute PPL values or their deltas across rows as if they were one matched experiment.

The six-module result supports continued testing of residual-space composition. It does not establish a general advantage because it covers one model family, a small layer subset, and a tiny evaluation set. The DAM rows in this repository are formula-based proxies rather than an official implementation and cannot be used to reject the published method.

## 4. Reproduction map

| Task | Entry point |
|---|---|
| Activation MVP | `scripts/run_mvp.sh` |
| Large local-path models | `scripts/run_large_435.sh` |
| KV-cache spectra | `scripts/run_kv_spectra.sh` |
| DMD | `scripts/run_dmd.sh` |
| Pythia checkpoints | `scripts/run_pythia_checkpoints.sh` |
| Structured Qwen2.5 pipeline | `scripts/run_structured_qwen25.sh` |
| Orthogonality/residual-stack experiments | `scripts/run_pretrained_llm_orthogonality.py` |
| OASR structured residual | `scripts/run_oasr_structured_residual.py` |
| Matched residual analysis | `scripts/run_structured_residual_matched.py` |

## 5. Remaining evidence gaps

- Repeat the Qwen2-7B residual-stack result over more layers, larger calibration/evaluation sets, and multiple seeds.
- Add a second 7B model family before claiming architecture-level generality.
- Evaluate selector quality against held-out perplexity rather than relying on proxy alignment alone.
- Keep pretrained/random controls for activation claims and pretrained-only language for feasibility runs.
- Extend the activation pipeline to vision without pooling away spatial structure; the required measurements are specified in `docs/vision_residual_rank_locality.md`.
- Emit a machine-readable manifest for every new run with package/CUDA versions, model and tokenizer revisions, dataset revision or local hash, seed, command, and git commit. These fields cannot be reconstructed reliably for every historical result in this archive.
