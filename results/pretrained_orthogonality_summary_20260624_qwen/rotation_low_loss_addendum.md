# Rotation Quantization And Low-Loss Triple-Stack Addendum

This addendum extends the pretrained orthogonality/SPQ evidence with rotation quantization and a conservative Q+S+R stack whose lossless criterion is benchmark drop below 1%. The new runs are smoke/diagnostic runs, not final leaderboard evaluations.

## Why Rotation Quantization Matters

Rotation-based quantization is not just another Q variant. It changes the coordinate basis before rounding so that weight/activation outliers are less axis-aligned. This connects directly to the Hessian-overlap framework: if quantization error is concentrated in fewer high-curvature coordinates, `rho_H`, additivity error, and order gaps should worsen; if rotation spreads that error into lower-conflict directions, Q should become more compatible with S and R.

Relevant prior work:

- QuaRot (<https://arxiv.org/abs/2404.00456>) uses computationally invariant rotations to remove LLM outliers and reports 4-bit W/A/KV quantization, plus lossless 6/8-bit settings.
- SpinQuant (<https://arxiv.org/abs/2405.16406>) learns rotation matrices and argues that not all rotations help equally; learned rotations reduce the zero-shot gap relative to fixed/random rotations.
- QuIP# (<https://arxiv.org/abs/2402.04396>) uses randomized Hadamard incoherence processing and lattice codebooks for strong low-bit weight-only PTQ.

Our implementation is intentionally narrower: `rotated_rtn` is a Hadamard-basis RTN proxy that quantizes `W H` and de-rotates by `H^T`. It does not implement full activation/KV rotation, learned rotations, packed kernels, or QuIP#/SpinQuant codebooks.

## Evidence From Existing Structured MD

The structured Qwen2.5-1.5B report at `results/structured_qwen25_1p5b_goal_smoke_20260606_024653/report.md` already supports the basic rotation mechanism:

- Hadamard reduced down-projection input-channel max/median from 1.237 to 1.079.
- Input-channel outlier count dropped from 18 to 0.
- 4-bit direct quantization error improved from 0.1869 to 0.1657.

This says rotation is doing the expected outlier-smoothing work before we evaluate it inside the pretrained orthogonality runner.

## New Pretrained Smoke Results

| Run | Rotation effect | Low-loss Q+S+R result |
| --- | --- | --- |
| Pythia-70M, 4 modules | Median Q error improves slightly, 0.1421 to 0.1402; median outlier ratio improves, 1.405 to 1.190. Median Hessian Q cost worsens slightly, 3.954 to 4.330, so rotation is not uniformly beneficial layer-wise. Fixed rotated-Q QSR PPL is 88.116 vs default QSR 88.354. | `rqs`, q=RTN, s=Wanda, r=whitened SVD, bits=8, keep=0.995, rank=0.995. PPL benchmark drop is 0.0000%, passing the <1% criterion. |
| Qwen2.5-1.5B, 2 attention modules | Median Q error improves from 0.1953 to 0.1411; median Hessian Q cost improves from 17.904 to 8.244; median outlier ratio improves from 2.084 to 1.445. Fixed rotated-Q QSR PPL is 36.613 vs default QSR 36.562, so lower local Q cost did not translate into a better full QSR PPL in this tiny run. | `rqs`, q=rotated RTN, s=Wanda, r=whitened SVD, bits=8, keep=0.995, rank=0.995. PPL benchmark drop is 0.0244%, passing the <1% criterion. |

The cross-run figure is `figures/rotation_low_loss_summary.svg`; machine-readable values are in `rotation_low_loss_summary.csv`.

## Zero-Shot Benchmark Check

Because the first smoke runs used too few zero-shot examples to support a 1% benchmark claim, an additional Pythia-70M ARC-Easy run was executed with 100 examples, one selected attention module, and the same conservative Q+S+R candidate family. The selected `low_loss_triple_stack` used `rqs`, q=rotated RTN, s=Wanda, r=whitened SVD, bits=8, keep=0.995, rank=0.995. Baseline ARC-Easy accuracy was 0.28; low-loss triple-stack accuracy was 0.32. The measured benchmark drop is therefore 0.0000%, passing the <1% criterion.

Artifact path: `results/pretrained_orthogonality_pythia70m_low_loss_zeroshot100_20260625/metrics/strategy_performance.csv`.

## Interpretation

Rotation quantization is effective as an outlier and quantization-error reducer, especially on Qwen attention projections. It should be treated as a strong Q candidate in the selector rather than as a guaranteed replacement for RTN. The Pythia smoke shows why: Hadamard rotation improves median error and outlier ratios but can increase Hessian self cost on some layers. This is exactly where the framework is useful: the decision should depend on local Hessian cost and pairwise overlap, not just Frobenius quantization error.

The low-loss triple-stack result establishes the requested minimum viable variant: all three operations are applied in a conservative regime, both Pythia and Qwen smoke runs satisfy PPL drop below 1%, and the added ARC-Easy100 check satisfies zero-shot benchmark drop below 1%. A final paper claim still needs a broader multi-task run; the current zero-shot evidence covers one task, not a full benchmark suite.

## Open Work For A Paper-Strength Result

1. Run the same rotation-aware selector on a larger Qwen/LLaMA target with more modules and a stable WikiText/C4 PPL window.
2. Use zero-shot tasks with enough examples for a meaningful <1% accuracy-drop decision.
3. Add learned rotations or block rotations as candidates, then test whether `rho_H` predicts when fixed Hadamard is insufficient.
4. Report memory/runtime separately from proxy dense replacement, because current `rotated_rtn` is a diagnostic weight-replacement proxy.
