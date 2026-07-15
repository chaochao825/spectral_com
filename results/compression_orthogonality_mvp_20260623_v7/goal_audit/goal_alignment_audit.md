# Goal Alignment Audit

Source result directory: `E:\Codex_work\ssh_experiment\llm_spectral_dynamics\results\compression_orthogonality_mvp_20260623_v7`.

## Verdict

The current MVP is consistent with the original goal at toy-model scale. It supports the central diagnostic claim more than the claim of a new compression pipeline: Hessian-weighted cross terms predict additivity error and real degradation, order non-commutativity is observable, and the metric can guide layer-wise choices under the same q/s/r budget.

The main caveat is scope: this is a small character-language model with diagonal empirical-Fisher/Hessian proxy, not yet a pretrained LLM with GPTQ/AWQ/SparseGPT-grade baselines.

## Success Criteria

| Criterion from original goal | Status | Evidence |
|---|---:|---|
| q/s/r Hessian cosine heatmap on at least one small model | Achieved | `hessian_cosine_heatmap.png`; 3 linear layers (`fc1`, `fc2`, `head`) with q/s/r pair matrix. |
| Higher `rho_H` pairs have larger additivity error | Achieved | Spearman(|rho_H|, |A_ij|) = `0.7582417582417583` over n=27; high row `fc2/qs` has |rho_H|=0.4898, |A_ij|=6.1102; low row `fc1/qr` has |rho_H|=0.0009, |A_ij|=0.3040. |
| Report real PPL/accuracy degradation correlations | Achieved | Spearman(|rho_H|, PPL degradation) = `0.5586080586080587`; Spearman(|rho_H|, accuracy degradation) = `0.4899281919710891`. Taylor loss prediction = `0.9420024420024421`, above Frobenius baseline `0.5927960927960929`. |
| Compare R->Q/S vs Q/S->R and explain order gap with singular spectrum + Hessian overlap | Partially achieved | Largest gap `head: rq vs qr` has abs loss gap=0.0344. R-first conditional overlap correlates with abs loss gap `0.5795378640469799`; singular entropy/top1/stable-rank shifts correlate `0.7982717345852616`. Symmetric max overlap is weak `0.14711055835105272`, so the order explanation should be phrased directionally. |
| Layer-wise method/order selection beats naive fixed-order baseline under same q/s/r settings | Achieved | Hessian-guided layer-wise PPL `1.0838` vs fixed `Q->S->R` PPL `1.1523`; accuracy degradation `0.0124` vs `0.0332`. |

## Interpretation Against Expected Claim

- Consistent: the evidence goes beyond a pretty landscape. It reports `rho_H`, additivity error, order gap, and actual PPL/accuracy degradation correlations.
- Consistent: the strongest pair is `fc2/qs` at `bits2/keep0.45`; the selected loss-landscape anchors now match that exact additivity row.
- Consistent but limited: the framework explains complement/conflict better for additivity and real loss degradation than for a symmetric order-gap overlap metric.
- Not yet sufficient for paper-scale evidence: needs at least one pretrained small LLM or transformer classifier, multiple seeds, and comparisons against stronger compression baselines.

## Generated Visualizations

- `figures/goal_alignment_dashboard.png/pdf`
- `figures/goal_alignment_layerwise.png/pdf`
- Existing supporting figures: `hessian_cosine_heatmap.png`, `loss_landscape_fc2_qs_contour.png`, `loss_landscape_fc2_qs_surface.png`
