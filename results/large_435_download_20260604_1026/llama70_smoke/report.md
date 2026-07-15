# LLM Spectral Dynamics Report

This report is generated from the current experiment outputs. Interpret conclusions as preliminary until the full configured model and dataset sweep has completed.

## Long-tail spectra
- Mean fitted alpha: 1.06
- Mean participation ratio: 30.88
- Mean effective rank: 57.05

## Pretrained vs random-init
- Comparison unavailable because one variant is missing.

## Attention vs FFN
- Attention effective rank mean: 39.17; FFN effective rank mean: 55.62.
- Residual stream effective rank mean: 76.38.

## KV cache
- KV-cache spectral and compression results are written separately when `scripts/run_kv_spectra.sh` or KV sites are enabled.

## Token-time dynamics
- Mean absolute PC autocorrelation across reported lags: 0.2406. DMD summaries are included in `results/metrics/dynamic_metrics.csv`.

## Loss and metric associations
- Mean token negative log likelihood over analyzed batches: 3.98. Use the metrics table to correlate alpha, rank metrics, and loss at model/layer/site granularity.
