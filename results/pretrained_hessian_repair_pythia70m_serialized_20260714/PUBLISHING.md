# Publication note

The original server-side run produced 12 deterministic `.hrc` research-codec
payloads totaling 48,200,900 bytes. They are intentionally omitted from this
GitHub publication clone.

`artifact_manifest.json` retains each payload path, SHA-256 digest, physical
file bytes, natural file bytes, logical payload bits, alignment, and
round-trip status. The CSV/JSON/Markdown summaries and bounded execution logs
remain versioned.

Tests that reopen and hash the raw codec files run strictly when those payloads
are present. In a publication clone without `.hrc` files, those artifact-only
tests are reported as skipped rather than treating intentional retention policy
as an algorithm failure.
