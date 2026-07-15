# Compression Orthogonality MVP

- Baseline validation loss: 0.0005; PPL: 1.0005; accuracy: 1.0000; examples: 6544.
- Highest |rho_H| additivity row: layer=fc2, pair=qs, |rho_H|=0.4898, |A_ij|=6.1102.
- Lowest |rho_H| additivity row: layer=fc1, pair=qr, |rho_H|=0.0009, |A_ij|=0.3040.
- Largest order gap: layer=head, rq vs qr, loss gap=0.0344, max directional conditional |rho_H|=0.1009.
- Layer-wise Hessian selection PPL: 1.0838; fixed-order PPL: 1.1523; same q/s/r settings are used for both.

## Correlations

- additivity: Spearman(abs_rho_h, abs_additivity_error) = 0.7582 over n=27.
- additivity: Spearman(rho_h, additivity_error) = 0.5354 over n=27.
- real_ppl: Spearman(abs_rho_h, ppl_degradation_pair) = 0.5586 over n=27.
- real_accuracy: Spearman(abs_rho_h, accuracy_degradation_pair) = 0.4899 over n=27.
- taylor: Spearman(taylor_predicted_loss_delta, loss_degradation_pair) = 0.9420 over n=27.
- frobenius_baseline: Spearman(frobenius_delta_sum, loss_degradation_pair) = 0.5928 over n=27.
- param_cos_baseline: Spearman(abs_parameter_cosine, abs_additivity_error) = 0.7143 over n=27.
- order_gap: Spearman(max_abs_conditional_hessian_overlap, abs_loss_gap) = 0.1471 over n=12.
- order_gap_ppl: Spearman(max_abs_conditional_hessian_overlap, abs_ppl_gap) = 0.1471 over n=12.
- order_gap_accuracy: Spearman(max_abs_conditional_hessian_overlap, abs_accuracy_gap) = 0.4751 over n=12.
- order_gap_mean_overlap: Spearman(mean_abs_conditional_hessian_overlap, abs_loss_gap) = 0.0280 over n=12.
- spectrum_order_rank90: Spearman(abs_first_rank_90_delta, abs_loss_gap) = -0.2560 over n=12.
- spectrum_order_entropy: Spearman(abs_first_spectral_entropy_delta, abs_loss_gap) = 0.7983 over n=12.
- spectrum_order_top1: Spearman(abs_first_top1_energy_delta, abs_loss_gap) = 0.7983 over n=12.
- spectrum_order_stable_rank: Spearman(abs_first_stable_rank_delta, abs_loss_gap) = 0.7983 over n=12.
- order_disagreement: Spearman(final_weight_disagreement, abs_loss_gap) = 0.3846 over n=12.

## Artifacts

- `metrics/hessian_cosine.csv` and `figures/hessian_cosine_heatmap.png`
- `metrics/additivity.csv`
- `metrics/order_gap.csv`
- `metrics/layerwise_selection.csv` and `metrics/layerwise_performance.csv`
- `metrics/correlations.csv`
