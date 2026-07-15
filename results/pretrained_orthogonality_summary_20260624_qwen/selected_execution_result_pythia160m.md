# Selected Execution Result: Pythia-160M Fair SPQ-Prior Run

This run executes the selected analysis path from `selected_execution_analysis.md`. The preferred 210/Qwen run was not launched because 210 was busy at check time: GPU 0/2/3 were near fully occupied and GPU 1 had nontrivial utilization. The documented fallback was used instead: Pythia-160M on the 236 server.

## Run

Artifact:

- `results/pretrained_orthogonality_pythia160m_selected_fair_spq_20260627`

Remote execution:

- Server: 236
- Conda env: `myenv`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Model: `EleutherAI/pythia-160m`
- Modules: first/middle/last `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`; 12 modules total
- Split protocol: disjoint calibration / PPL evaluation / LoRA recovery windows
- Calibration texts: 256
- PPL evaluation texts: 512
- LoRA recovery texts: 512
- PPL tokens: 8,128
- Zero-shot: ARC-Easy and HellaSwag, 100 examples each
- Compression budget: 4-bit, keep=0.8, rank=0.5
- LoRA recovery: rank 4, 5 steps

Command shape:

```bash
export PYTHONPATH=/home/wangmeiqi/llm_spectral_dynamics/src
export CUDA_VISIBLE_DEVICES=0
python scripts/run_pretrained_llm_orthogonality.py \
  --model EleutherAI/pythia-160m \
  --local-files-only \
  --device cuda \
  --svd-device cuda \
  --output-dir results/pretrained_orthogonality_pythia160m_selected_fair_spq_20260627 \
  --module-types query_key_value,dense,dense_h_to_4h,dense_4h_to_h \
  --layer-positions first,middle,last \
  --max-modules 12 \
  --bits 4 \
  --keep-fraction 0.8 \
  --rank-fraction 0.5 \
  --q-method rtn \
  --s-method wanda \
  --r-method whitened_svd \
  --calib-limit 32 \
  --eval-limit 64 \
  --disjoint-text-splits \
  --texts-per-batch-window 8 \
  --sequence-length 128 \
  --batch-size 1 \
  --text-source zero_shot_backup \
  --zero-shot-tasks arc_easy,hellaswag \
  --zero-shot-strategy-limit 0 \
  --fair-benchmark-zero-shot-limit 100 \
  --include-fair-benchmark \
  --include-fair-extended-recipes \
  --spq-s-method wanda \
  --spq-r-method whitened_svd \
  --spq-guided-q-methods rtn,rotated_rtn \
  --spq-guided-s-methods magnitude,wanda \
  --spq-guided-r-methods svd,whitened_svd \
  --spq-lora-steps 5 \
  --spq-lora-rank 4 \
  --spq-lora-train-limit 64
```

## Fair Benchmark Summary

Baseline PPL is 54.6310 and baseline mean zero-shot accuracy is 0.375.

| Strategy | Memory | Predicted Hessian cost | PPL delta | Mean zero-shot delta | Judgment |
| --- | ---: | ---: | ---: | ---: | --- |
| `q_only_rotated_4bit` | 0.250 | 5624.7020 | +7.99% | -0.025 | strong single-method reference |
| `s_only_wanda_keep0p8` | 0.800 | 11.9349 | +2.19% | -0.035 | best PPL reference, high memory |
| `r_only_whitened_rank0p5` | 0.667 | 168.1213 | +19.61% | -0.065 | low-rank is still costly |
| `qsr_rotated_wanda_whitened` | 0.133 | 5795.2216 | +32.29% | -0.055 | aggressive fixed stack |
| `rqs_rotated_wanda_whitened` | 0.133 | 5877.9638 | +28.51% | -0.050 | best fixed QSR/RQS PPL |
| `hessian_guided_qsr_budget` | 0.133 | 5757.3837 | +28.63% | -0.070 | nearly tied on PPL, worse zero-shot |
| `slim_like_srq_proxy` | 0.133 | 7355.5632 | +31.78% | -0.050 | fixed proxy, not official SLiM |
| `spq_like_rsq_no_lora` | 0.196 | 7152.9278 | +13.03% | -0.055 | fixed SPQ-like |
| `hessian_guided_spq_no_lora` | 0.196 | 5710.5654 | +9.05% | -0.055 | PPL improves, zero-shot tied |
| `spq_like_rsq_lora` | 0.196 | 7152.9278 | +11.58% | -0.035 | fixed SPQ-like + tiny LoRA |
| `hessian_guided_spq_lora` | 0.196 | 5710.5654 | +8.06% | -0.065 | best PPL, worse zero-shot |

Focused visualization:

- `results/pretrained_orthogonality_pythia160m_selected_fair_spq_20260627/figures/focused_spq_qsr_comparison.png`

## What Changed Relative To Pythia-70M

### Hessian-Guided QSR

Pythia-160M is more favorable to Hessian-guided QSR than the Pythia-70M split run on PPL, but it still does not establish a strong method claim.

- PPL: `hessian_guided_qsr_budget` is +28.63%, essentially tied with fixed RQS at +28.51% and better than fixed QSR at +32.29%.
- Zero-shot: `hessian_guided_qsr_budget` is -0.070, worse than fixed QSR (-0.055), fixed RQS (-0.050), and SLiM-like proxy (-0.050).

Conclusion: Hessian-guided QSR is no longer clearly worse on PPL in this larger Pythia run, but it is not Pareto-superior and should not be the main method claim.

### Hessian-Guided SPQ

The SPQ-prior path is again the strongest result.

- No LoRA: PPL delta improves from +13.03% to +9.05%, a 3.98 percentage-point reduction. Mean zero-shot delta is unchanged at -0.055.
- With equal tiny LoRA: PPL delta improves from +11.58% to +8.06%, a 3.52 percentage-point reduction. Mean zero-shot delta worsens from -0.035 to -0.065.

Conclusion: the run supports the selected execution strategy. Hessian guidance can improve PPL inside a sensible SPQ-like layer prior at matched memory and matched recovery budget. It is still not Pareto-dominant because the LoRA row worsens zero-shot and the no-LoRA row only ties zero-shot.

## Diagnostic Correlations

| Diagnostic | Spearman rho | Interpretation |
| --- | ---: | --- |
| `abs(rho_H)` vs `abs(additivity_error)` | 0.3452 | moderate support for overlap explaining additivity |
| `abs(rho_H)` vs PPL degradation | -0.3514 | raw `rho_H` does not predict real PPL degradation here |
| Taylor/cross-term vs loss degradation | 0.5732 | useful, but not dominant |
| Frobenius delta sum vs loss degradation | 0.7254 | stronger than the Hessian/Taylor predictor in this run |
| activation reconstruction vs loss degradation | 0.5786 | slightly above Taylor/cross-term |
| trace-only cost vs loss degradation | 0.5655 | comparable to Taylor/cross-term |
| order-gap symmetric overlap | 0.2530 | weak-to-moderate order-gap signal |
| order-gap singular entropy shift | 0.1939 | weak signal |
| order-gap final weight disagreement | 0.2939 | strongest reported order-gap correlate in this run |

This reinforces the revised method diagnosis: `rho_H` and Taylor/cross-term features are useful diagnostics, but they should not be the only selector objective. Frobenius and activation-reconstruction terms must be included in the next selector.

## Selection Behavior

For Hessian-guided-SPQ:

- Attention modules mostly choose `R->Q` with `whitened_svd`; most Q choices are `rotated_rtn`.
- The large `L6:query_key_value` attention module still chooses plain `rtn`.
- MLP modules mostly choose `Q->S` or `S->Q` with Wanda; late `dense_4h_to_h` chooses `S->Q`.

This is consistent with the SPQ-prior interpretation: the layer family prior does most of the safety work, while Hessian guidance refines order/method choices.

## Paper Claim Update

Supported after this run:

- The selected SPQ-prior execution path is reasonable.
- Hessian-guided-SPQ gives repeatable PPL improvement over fixed SPQ-like under matched memory, now on Pythia-70M and Pythia-160M split-data runs.
- The framework remains useful for diagnosing additivity and order, but raw `rho_H` is not a robust scalar predictor of real PPL degradation.

Not supported:

- Hessian-guided QSR as a standalone competitive compressor.
- Pareto-dominance over fixed SPQ-like recipes.
- A claim that Hessian/Taylor predictors beat simple baselines in every run.

Next action:

1. Retry the selected Qwen2.5-1.5B run when 210 has an idle GPU and the runtime environment is ready.
2. Implement selector ablations: fixed method + guided order; guided method + fixed order; guided method + guided order.
3. Add a multi-objective selector using Hessian/Taylor, Frobenius, activation reconstruction, trace-only sensitivity, and a zero-shot proxy.
