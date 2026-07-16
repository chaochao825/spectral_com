# Publishing Note

This directory is a compact, publishable audit record for the Pythia-70M
one-tensor smoke run.

The raw `.hrc` codec artifacts are intentionally not committed. They remain in
the canonical run directory on server 210:

```text
/home/wangmeiqi/codex_worktrees/com_compression-two-stage-20260717/results/two_stage_heterogeneous_pythia70m_natural_match_smoke_20260717/artifacts
```

`artifact_manifest.json` and `artifact_payloads.csv` preserve the artifact
filenames, SHA-256 digests, natural and padded byte counts, stream composition,
and exact-rate checks. Paths under `artifacts/` are therefore provenance
references, not files available in this Git checkout.

The result is a feasibility and evidence-gate check, not a compression claim:

- QSL natural bytes: `627,712`
- exact-natural no-joint bytes: `627,712`
- QSL final-test NLL: `4.544342`
- no-joint final-test NLL: `4.544342`
- same-layer joint-value claim: `false`

Both endpoints selected the same 4-bit row-scale RTN plus FP16 low-rank repair,
with no active sparse component.

The archived `summary.md` wording was clarified for the heterogeneous
4/8/16-bit low-rank factor codec. Numeric CSV/JSON evidence was not changed.
