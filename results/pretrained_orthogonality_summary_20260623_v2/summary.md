# Pretrained Orthogonality Run Summary

All corrected runs use `EleutherAI/pythia-70m` and local ARC-Easy/HellaSwag backup text for PPL/calibration because WikiText-2 download timed out on the remote server. Per-run provenance is in `metrics/text_source_metadata.csv`.
The overlap metric is a layer-local Hessian/Gauss-Newton proxy based on activation covariance `X^T X`; additivity rows test linearized perturbation sums `W + Delta_i + Delta_j`, while order-gap rows test executable composed compression orders.

| Run | Modules | Compression | rho additivity | rho PPL | rho zero-shot | Taylor vs loss | Frobenius vs loss | Hessian PPL | Fixed default PPL | SLiM-proxy PPL |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default_4bit_6mods_v2 | 6 | 4-bit, keep0.5, rank0.5 | 0.1249 | 0.2941 | 0.2877 | -0.0526 | 0.1290 | 97.05 | 105.4 | 102.3 |
| mid_3bit_12mods_v2 | 12 | 3-bit, keep0.5, rank0.5 | 0.1449 | 0.1748 | -0.0503 | 0.7246 | 0.5001 | 353.4 | 398.1 | 455.2 |
| strong_2bit_12mods_v2 | 12 | 2-bit, keep0.4, rank0.4 | 0.4842 | 0.2165 | 0.1164 | 0.5367 | 0.4857 | 3.607e+07 | 1.042e+08 | 5.899e+07 |

Interpretation: the pretrained evidence is mixed. Hessian-guided layer-wise selection beats fixed default and the SLiM-like proxy in all three settings by PPL, but `rho_H` only moderately predicts linearized additivity in the strong setting and weakly predicts PPL/zero-shot degradation. Taylor/cross-term prediction is strongest in the mid setting and beats Frobenius there; in the strong setting it is only slightly above Frobenius, and in the default setting it fails.
