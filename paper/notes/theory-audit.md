# Theory derivation audit

This note records the assumptions behind `sections/theory.tex` and
`appendices/theory_proofs.tex`. It is an audit aid, not an additional result.

## Curvature objects

- `J_0 = nabla^2 L(w_0)` is the task Hessian and may be indefinite.
- `H >= 0` is the declared geometry: empirical Fisher, GGN, activation Gram,
  or a task Hessian damped enough to be PSD.
- Norms, correlations, Cauchy--Schwarz, and the coherence bound use only `H`.
- When `H != J_0`, the quadratic mismatch is explicitly included in
  `epsilon_H`; the local path remainder is separate conceptually but grouped in
  the same uncertainty term in the main paper.

## Result-to-theory boundary

- The coherence proposition proves approximate additivity of self costs; it
  does not prove small loss or same-rate superiority.
- Negative `rho_H` means cancellation, not orthogonality.
- The comfort-zone KKT result is a continuous relaxation. Rank, sparse-group,
  descriptor, and alignment thresholds require a discrete exact-file-byte
  endpoint comparison.
- The new nested allocator verifies complete natural-file feasibility exactly,
  but it optimizes only an enumerated and Pareto-pruned rank/support state set.
  It is not an exhaustive global optimum over supports, ranks, orders, or
  continuous factors.
- The exchange identity is exact for a fixed realized, order-specific
  weight-space decomposition. The Hessian estimate of its interaction is only
  local.
- A zero-byte repair is reported as direct recovery, never infinite
  recovery/byte, and must have no decoder side state.

## Current evidence ceiling

The detailed exploratory result remains one seed on six selected Pythia-70M
MLP tensors.  Three additional seed-17 scalability-smoke jobs are also
verified: 12 full-MLP Pythia tensors and ten depth-stratified MLP tensors each
from OPT-125M and Qwen3-0.6B.  In all three smoke jobs S--L is inside the
declared near-orthogonality band, Q--S/Q--L are negative cancellation, and the
strict scaled QSL endpoint has higher NLL than same-byte Q+L.  These are three
unpooled observations with unequal scopes, not multi-seed or model-scale
confirmation.  The legacy strict files are byte-equal after tail padding, but
their natural QSL encodings underfill the Q+L caps; they reject the tested
per-layer-guard allocations rather than every budget-exhausted QSL allocation.
The nested allocator and the 15-job 3B/7B method-factor matrix are implemented
but unexecuted because server 210 was unreachable on 2026-07-16.  No theorem or
wording in the theory files claims full-model, production-format, global
optimality, or universal advantage.

## Reviewer checks to preserve

1. Disclose the exact PSD proxy and damping, if any.
2. Report all pairwise signed correlations and absolute correlations.
3. Keep the gradient term and endpoint loss; do not infer from the quadratic
   term alone.
4. Report natural bytes, file bytes, descriptors, indices/row pointers,
   factors, alignment, and padding.
5. Recompute geometry after the final serialization round trip and fixed stage
   order.
6. Treat calibration-to-held-out transfer and source/model scaling as empirical
   questions.
7. Keep protocol-v2 comfort-path fitting on validation windows and reserve test
   windows for endpoint evaluation.
8. Record the whitened-SVD fitting floor separately from the PSD covariance
   used for endpoint scoring.
