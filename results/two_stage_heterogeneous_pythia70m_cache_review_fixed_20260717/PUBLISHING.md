# Publishing Note

This directory is the compact audit record for the Pythia-70M one-tensor
regression after the cache review fixes. It is a scalability and evidence-gate
check, not a full-model compression claim.

The 14 raw `.hrc` codec artifacts (10,084,436 bytes total) are intentionally not
committed. They remain in the canonical run directory on server 210:

```text
/home/wangmeiqi/codex_results/cache_review_fixed_pythia70m_smoke_20260717/artifacts
```

`artifact_manifest.json` and `artifact_payloads.csv` retain each artifact name,
SHA-256 digest, natural and padded byte count, stream composition, and exact-rate
check. Paths under `artifacts/` are provenance references rather than files in
this Git checkout.

The relevant final-test result is:

- QSL natural bytes: `627,712`
- exact-natural no-joint bytes: `627,712`
- QSL final-test NLL: `4.5663907451`
- no-joint final-test NLL: `4.5663907451`
- same-layer joint-value claim: `false`

The repaired priming policy covers every target and global budget-multiplier
band before smaller ranks slice the largest cached superset decomposition. This
reduced SVD solver calls from the original 66 to 44. Relative to the earlier
target-only priming run, 161 FP16 Q+L endpoint elements changed, with maximum
absolute difference `3.0517578e-05`; therefore this directory is archived as a
distinct numerical-policy result instead of overwriting the earlier evidence.

Key committed-file hashes before Git line-ending normalization:

- `run_config.json`: `714996595f3c0e84e9d36007eac835af7a262d9997b05d71ec4294b394bbd011`
- `strategy_endpoints.csv`: `26f9c60b29326eab734877519d9284490a06a056c893e991221a924dabf981be`
- `artifact_manifest.json`: `5964819ffd62089843557fec39d80a0dde46a49ff4c5c645a14abfd6a316a4df`
