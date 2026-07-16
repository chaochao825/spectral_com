# ICML 2026 Outline Contract

## Venue and Length Contract

- Format: ICML 2026, double blind.
- Main text: 8.00 pages exactly, excluding references and supplementary material.
- Reader: an adjacent-area ML researcher who understands PTQ but not this codec or Hessian notation.
- Claim policy: `verified`, `preregistered`, `planned`, and `literature-reported` are never collapsed into a single evidence tier.
- Current numeric boundary: (i) the original Pythia-70M six-selected-tensor, one-seed endpoint and (ii) three separate seed-17, selected-weight-target-0.258 scalability-smoke jobs: Pythia-70M full MLP, OPT-125M five-depth MLP, and Qwen3-0.6B five-depth up/down MLP. The smoke jobs are never pooled and are not whole-model, multi-seed, multi-rate, or model-scale-trend evidence.

## Section Tree and Page Budget

| # | Section | Pages | Intent | Claims | Citation quota | Visual quota |
|---:|---|---:|---|---|---:|---:|
| 1 | Introduction | 0.70 | Define the exact-rate failure mode, state the bounded negative result and repeated scalability-smoke diagnostic, and motivate comfort-zone allocation. | C0, C1, C2, C13 | 6-8 | 1 teaser |
| 2 | Related Work | 0.70 | Organize work by A0/A1/B/C training dependence and by quantization, pruning, low-rank, and hybrid payload. | C9, C10 | 12-16 | 1 compact taxonomy table |
| 3 | Problem Formulation and Exact Rate | 0.85 | Define tensor scope, research codec bytes, perturbations, NLL, Hessian metric, and rate-distortion objective. | C0, C1, C11 | 3-5 | 1 byte-accounting schematic |
| 4 | Method: Comfort-Zone Repair Allocation | 1.35 | Specify candidate generation, OBS/scale/low-rank repair, exact-byte marginal gain, and strict allocation. | C0, C5, C6 | 4-6 | 1 method pipeline |
| 5 | Theory and Diagnostic Predictions | 1.15 | Derive self/cross-term conditions, distinguish cancellation from orthogonality, and include first-order endpoint limits. | C3, C4, C6, C13 | 3-5 | 1 geometry/loss-landscape figure |
| 6 | Experimental Protocol | 0.75 | Fix models, tensor coverage, splits, seeds, rates, metrics, baseline lanes, and provenance requirements; distinguish completed smoke from confirmatory design. | C7, C8, C9, C13 | 5-7 | 1 setup table |
| 7 | Results and Analysis | 2.10 | Lead with verified physical-rate evidence and three unpooled smoke observations; add multi-seed/model frontiers only after execution. | C1-C5, C8, C12, C13 | 4-6 | 2 tables + 3 result panels |
| 8 | Limitations and Conclusion | 0.40 | State evidence gaps, deployment limits, and the bounded/smoke takeaway without upgrading planned claims. | C2, C7, C8, C11, C12, C13 | 0-2 | 0 |
|  | **Total** | **8.00** |  |  | **37-55** | **5 figures/panels + 3 tables** |

## Argument Order

1. Nominal bits and value-only streams are insufficient rate controls when heterogeneous payloads have indices, factors, descriptors, and alignment.
2. The current physical-byte control reverses the earlier direction: strict Q+L beats strict scaled Q+S+L in the bounded run.
3. Three separate seed-17/target-0.258 selected-weight smoke scopes repeat the positive strict-scaled-QSL-minus-QL NLL sign and near-zero S/L Hessian correlation, without pooling or implying a model-size trend.
4. The reversal and repeated smoke sign motivate a rate allocator, not a claim that heterogeneous repair is ineffective in principle.
5. Hessian self terms define local component cost; Hessian cross terms distinguish complementarity, redundancy, and cancellation.
6. Every candidate component is charged by exact new bytes and accepted only when its endpoint held-out gain remains positive.
7. Multi-seed, multi-rate, full-scope, and larger-model experiments decide whether that mechanism improves the frontier; until then, the performance claim is planned.

## Theory Obligations

- Define `Delta L(d) = g^T d + 1/2 d^T H d` and state when the first-order term can or cannot be neglected.
- Expand `Delta L(d_a+d_b)` and define the Hessian correlation `rho_H(a,b)`.
- Give sufficient, not necessary, conditions for a combination to improve over the best single repair at a fixed serialized-byte budget.
- Include discrete-rate terms for sparse support, rank increments, descriptors, and alignment; do not optimize a continuous relaxation and report it as a physical endpoint.
- Distinguish repair-to-repair near-orthogonality from both repairs cancelling the same quantization error.
- State a trust-region or comfort-zone condition and test its diagnostics through interpolation paths.
- Avoid a theorem claiming global optimum or universal advantage unless its assumptions and proof actually support it.

## Results Slots and Upgrade Rules

| Slot | Initial status | Required evidence before factual prose |
|---|---|---|
| Exact physical same-rate endpoint table | verified | Existing manifest, endpoint CSV, on-disk hashes, and paired summary |
| Hessian geometry and 13-point landscape | verified | Existing strategy/landscape artifacts; caption must say diagnostic and one seed |
| Three-job scalability-smoke table and panels | verified | Three completed-valid suite jobs, fail-closed aggregate manifest, exact selected-weight bytes, and explicit one-seed/one-rate/unpooled captions |
| Eight-seed Pythia-70M result | preregistered | Eight completed source-snapshotted runs and aggregate script output |
| Pythia-160M/410M/1B scaling | planned | Protocol-matched independent-seed artifacts for each model and disclosed rates; the Pythia-70M smoke is not a substitute |
| Qwen-family transfer | planned | Architecture-compatible tensor scope and protocol-matched independent-seed/rate artifacts; the Qwen3-0.6B depth smoke is not a transfer conclusion |
| External A0/A1 direct table | planned | Official or audited implementations under identical rate/scope/evaluation |
| Backward-assisted B and global-recovery C tables | literature-reported/planned | Separate compute-qualified tables; no mixing with frozen PTQ |
| Accuracy-rate Pareto frontier | planned | Multiple exact byte targets, endpoint metrics, and uncertainty across seeds |
| Encoder/decode resource table | planned | Wall time, peak memory, output bytes, and implementation/provenance disclosure |

## Figure and Table Contract

- Figure 1: problem/byte-accounting teaser; structural, not a result claim.
- Figure 2: method pipeline from candidate repairs through exact-byte allocation and endpoint validation.
- Figure 3: verified Hessian cross-term geometry and 13-point loss landscapes; source paths embedded in generation metadata.
- Figure 4: verified three-job efficiency/geometry and loss-path panels; captions state three separate seed-17/target-0.258 selected-weight scopes and prohibit cross-model reading.
- Figure 5: planned multi-seed exact-rate Pareto/model-scale panels; absent from the paper until confirmatory result files exist.
- Table 1: training-lane and payload-accounting taxonomy from primary sources.
- Table 2: verified bounded endpoints, with tensor and seed scope in the caption.
- Table 3: verified three-job scalability-smoke endpoints, with within-model comparisons only and no pooled statistic.
- A confirmatory large-scale/external numeric table remains absent until matching artifacts exist; literature-reported protocol rows may appear only as a clearly separated taxonomy or supplement.

## Supplement Contract

The supplement may contain codec layout, full method matrix, complete rate sweeps, per-layer geometry, per-seed tables, provenance hashes, negative controls, and external reproduction notes. Moving content to the supplement does not relax evidence labels.
