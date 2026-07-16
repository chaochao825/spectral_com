# Paper Figure Sources

All experimental figures are regenerated from committed result tables; no paper plot contains hand-entered measurement values.

- `exact_rate_tradeoffs_plot.py` reads the serialized endpoint and paired-window comparison CSV files.
- `loss_landscape_plot.py` reads the 13-point path sweep.
- `hessian_geometry_plot.py` reads the conservative endpoint Hessian correlations; its natural file underfills Q+L and is tail-padded to the same final bytes.
- `scaling_pilot_plots.py` first verifies the scaling-report manifest and CSV
  hashes, then plots exact-bit parameter utilization, conservative strict
  endpoints tail-padded to equal final bytes (with natural underfill disclosed),
  differences, signed Hessian geometry, and 13-point path slices for the three
  separate scalability-smoke jobs.

Each script emits vector PDF/SVG and a 300-dpi PNG preview, plus a manifest
containing input SHA256 values and an explicit evidence scope.  The original
figures cover Pythia-70M, six selected MLP linear tensors, and one seed.  The
scaling-smoke figures add three unpooled seed-17 observations with unequal
tensor scopes.  Neither set may be described as whole-model evidence,
multi-seed confirmation, or a cross-model magnitude ranking.
