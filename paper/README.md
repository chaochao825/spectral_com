# ICML 2026 paper

The manuscript uses the unmodified official ICML 2026 style files committed in
this directory. Experimental prose is bounded by the evidence contracts in
brief, plan, issues, and notes/design.

## Regenerate bound results

~~~bash
python results/build_verified_tables.py
python results/build_scaling_pilot.py
python figures/exact_rate_tradeoffs_plot.py
python figures/loss_landscape_plot.py
python figures/hessian_geometry_plot.py
python figures/scaling_pilot_plots.py
~~~

Each figure emits vector PDF/SVG, a PNG preview, and an input-hash manifest.
The exploratory plots cover one Pythia-70M seed and six selected MLP tensors.
The separate scaling-smoke plots cover three seed-17 jobs at one selected-weight
artifact rate: all 12 Pythia-70M MLP projections and ten depth-stratified MLP
projections each from OPT-125M and Qwen3-0.6B.  These observations are not
pooled, do not form a model-size trend, and are not whole-model compression
results.

## Compile

From the paper directory:

~~~bash
tectonic main.tex --keep-logs --keep-intermediates
~~~

The local development build was made with the official Tectonic 0.16.9 binary.
Use the blind-review icml2026 package option. Do not switch to accepted until
the paper is accepted.

Tectonic uses XeTeX, for which the legacy `times` package may otherwise fall
back to Latin Modern. The repository therefore bundles TeX Gyre Termes, Heros,
and Cursor under `fonts/`; these are Times/Helvetica/Courier-compatible fonts
distributed under the included GUST Font License. pdfLaTeX builds continue to
use the official style's native font setup.

## Evidence rule

- **verified**: committed, hash-checkable result artifacts.
- **preregistered**: fixed protocol only; no implied model result.
- **planned**: unexecuted design.
- **literature-reported**: primary-source metadata, not our measurement.

The manuscript must never merge these labels into one leaderboard.
