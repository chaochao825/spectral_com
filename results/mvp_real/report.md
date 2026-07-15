# LLM Spectral Dynamics Report

This report is generated from the current experiment outputs. Interpret conclusions as preliminary until the full configured model and dataset sweep has completed.

## Long-tail spectra
- Mean fitted alpha: 1.826
- Mean participation ratio: 42.99
- Mean effective rank: 82.92

## Pretrained vs random-init
- Pretrained mean alpha: 1.808; random-init mean alpha: 1.844; delta: -0.03623.

## Attention vs FFN
- Attention effective rank mean: 31.66; FFN effective rank mean: 108.4.
- Residual stream effective rank mean: 108.8.

## KV cache
- KV-cache spectral and compression results are written separately when `scripts/run_kv_spectra.sh` or KV sites are enabled.

## Token-time dynamics
- Mean absolute PC autocorrelation across reported lags: 0.1042. DMD summaries for this archived run are in `results/mvp_real/metrics/dynamic_metrics.csv`.

## Loss and metric associations
- Mean token negative log likelihood over analyzed batches: 6.542. Use the metrics table to correlate alpha, rank metrics, and loss at model/layer/site granularity.
