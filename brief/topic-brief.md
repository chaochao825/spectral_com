# Topic Brief

## Topic

Exact-serialized-rate heterogeneous post-training compression for pretrained language models, with Hessian geometry used to diagnose and allocate quantization, sparse, low-rank, and scale repairs.

## Research Question

At a fixed physical artifact size, can several individually small compression perturbations remain inside their low-loss "comfort zones" and outperform a single repair, what Hessian and byte-accounting conditions make that plausible, and which conditions are sufficient for a realized same-byte improvement?

## Evidence Boundary as of 2026-07-16

- **Verified bounded endpoint:** one Pythia-70M run covering six selected MLP linear tensors, one seed, and 2,032 held-out NLL tokens. The `Q+L` and legacy strict-scaled `Q+S+L` files are both 3,248,832 final bytes. Their perplexity deltas are respectively +6.761523 and +7.319517, so that tested combination is worse by 0.557993 perplexity. Its natural encoding is 3,233,152 bytes and 15,680 bytes are tail padding; this rejects the conservative per-layer-guard candidate, not every budget-exhausted Q/S/L allocation.
- **Verified three-job scalability smoke:** three separate seed-17, selected-weight-target-0.258 jobs cover Pythia-70M full MLP (12 tensors), OPT-125M five-depth `fc1/fc2` MLP (10 tensors), and Qwen3-0.6B five-depth `up/down` MLP (10 tensors). Each uses 1,016 held-out tokens and eight fixed windows. At identical within-job Q+L/strict-scaled-Q+S+L file bytes, strict-minus-Q+L NLL is respectively +0.005417, +0.000931, and +0.007253; `rho_H(S,L)` is +0.0263, +0.0871, and +0.0221. These are three unpooled observations, not a cross-model average, model-size trend, or significance test.
- **Verified diagnostic evidence:** the bounded run and each scalability-smoke job contain 13-point interpolation landscapes, fixed-window paired NLL diagnostics, Hessian self/cross terms, round-trip codec checks, and exact file hashes. In the three smoke scopes, fixed-support OBS improves Q+S at zero added bytes and block scales improve Q at positive byte cost; folded global scale helps two scopes and hurts one. These diagnostics do not establish population-level significance or model-scale generality.
- **Implemented, not yet empirically verified:** a nested allocator now considers Q, Q+S_OBS, Q+L, and enumerated Q/S/L rank-support states, prunes them by additive local byte/cost dominance, and checks complete multi-layer natural-file feasibility with the real serializer. “Exact” refers to that final feasibility check; the enumerated/pruned state space is not a global optimum over every possible support or rank.
- **Verified theoretical support:** the same-byte comparison is decomposed into payload give-up, realized repair gain, and interaction cost, retaining the first-order term and discrete serialized feasibility. The resulting condition is sufficient, not necessary. Local radial proxy agreement does not establish cross-candidate ranking accuracy.
- **Preregistered, not executed:** the eight-seed WikiText-2 calibration/validation/test split manifest. It is a data/split contract, not an eight-seed accuracy result.
- **Partial live state, not committed evidence:** server 210 is reachable. The 15-job `large_model_method_ablation_20260716` directory has 2 `completed_valid` rows and 13 planned rows; its completed Pythia/OPT manifests report CUDA unavailable/CPU execution. Separate live suites have 1/24 completed Pythia confirmatory-frontier job and 2/28 completed Qwen2.5-3B layer-0 gate-only SVD sentinels. All three directories are untracked, so none can be labelled a completed matrix, 3B/7B frontier, full-MLP result, or confirmatory evidence. The older 69-job expanded matrix also remains unexecuted, and independent-seed claims still require the preregistered protocol.
- **Literature-reported only:** external method performance and training requirements. OBS, Hessian-constrained quantization, GPTQ/OBC/SparseGPT/SpQR, SLiM, Harma et al., OBR, Fisher-guided variable-length formats, and ProjQ prevent “first QSL,” “first Hessian+rate,” or “first interaction/orthogonality” claims. QuantSparse, CacheQuant, Q-VDiT/S2Q-VDiT, TeaCache, sparse/structured video attention, and sparse-low-rank attention additionally bound the multimodal extension. The 59-row repository matrix supports protocol classification, not an apples-to-apples measured leaderboard.
- **Proposed multimodal stack, not measured:** read-only visual-model diagnostics motivate a gated decomposition into sink/global low rank, local/cyclic structure, dynamic sparse routes, and dense fallback. The proposed stack combines that attention gate with exact-byte weight Q/S/L/OBS, W/A PTQ, cache, and temporal reuse under joint quality, storage, runtime-state, peak-memory, and end-to-end-latency accounting.

## Scope

### In Scope

- Decoder-only pretrained language models evaluated after compression.
- Native components already implemented in the project: quantization (`Q`), block/component scales, sparse residuals (`S`), OBS value repair, low-rank residuals (`L`), and strict/relaxed combined endpoints.
- Exact serialized artifact bytes, including codes, scales, sparse values and support, low-rank factors, descriptors, headers, padding, and alignment.
- Hessian-weighted perturbation norms and cross terms, marginal held-out loss recovery per added physical byte, and endpoint NLL/perplexity.
- Separate comparison tracks for A0 data-free PTQ, A1 calibrated no-backward PTQ, B local backward-assisted correction, and C global task-recovery/QAT methods.
- A scope-separated D lane for video/image diffusion quantization, cache, sparse/structured attention, and sparse-low-rank attention, plus a proposed multimodal validation protocol.
- Negative results and scope limitations as first-class evidence.

### Out of Scope Until New Evidence Exists

- Production-kernel throughput, energy, or deployment-format claims.
- A universal claim that quantization, sparse, and low-rank perturbations are mutually orthogonal.
- A claim that combined compression already beats the best single component at strict equal rate.
- Full-model, multi-seed, model-scale-trend, or cross-family ranking conclusions inferred from the bounded run or the three scalability-smoke jobs.
- Treating the selected-weight artifact ratio 0.258 as a whole-model compression ratio; embeddings, attention, output heads, and other excluded state remain outside the charged smoke scopes.
- Numeric ranking of external methods before model, data, tensor scope, training budget, artifact bytes, and evaluation protocol are aligned.
- Treating reported kernel, attention-only, cache-only, PTQ-only, or end-to-end speedups as multiplicative components of an unmeasured stack.

## Audience

ICML reviewers and researchers working on language-model post-training quantization, pruning, low-rank correction, rate-distortion allocation, and second-order compression diagnostics.

## Constraints

- Venue: ICML 2026 style and double-blind submission conventions.
- Main-text target: exactly 8 pages, excluding references and supplementary material.
- Evidence rule: every empirical statement maps to an artifact or is explicitly labelled preregistered, planned, or literature-reported.
- Comparison rule: training-dependent methods remain in their own lane and are not presented as frozen no-backward PTQ controls.
- Rate rule: nominal bits per weight and logical value-stream sizes are descriptive only; direct comparisons use physical serialized bytes over the same tensor scope.
- Statistical rule: the present fixed-window intervals and three separate seed-17 observations are descriptive and are never pooled; multi-seed claims require the preregistered protocol to be consumed by the numerical runner and executed.

## Key Terms

exact serialized rate; Hessian geometry; perturbation orthogonality; loss landscape; comfort-zone allocation; parameter efficiency; OBS repair; block scale; sparse residual; low-rank residual; post-training compression
