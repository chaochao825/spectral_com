# LLM Spectral Dynamics Report

This report is generated from the current experiment outputs. Interpret conclusions as preliminary until the full configured model and dataset sweep has completed.

## Long-tail spectra
- Mean fitted alpha: 1.152
- Mean participation ratio: 21.95
- Mean effective rank: 45.08

## Pretrained vs random-init
- Comparison unavailable because one variant is missing.

## Attention vs FFN
- Attention effective rank mean: 33.69; FFN effective rank mean: 66.63.
- Residual stream effective rank mean: 34.91.

## KV cache
- KV-cache spectral and compression results are written separately when `scripts/run_kv_spectra.sh` or KV sites are enabled.

## Token-time dynamics
- Mean absolute PC autocorrelation across reported lags: 0.1946. DMD summaries are included in `results/metrics/dynamic_metrics.csv`.

## Loss and metric associations
- Mean token negative log likelihood over analyzed batches: 3.126. Use the metrics table to correlate alpha, rank metrics, and loss at model/layer/site granularity.
