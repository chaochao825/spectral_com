# Pretrained LLM Orthogonality Conclusion With Qwen Validation

This summary updates the model-scale conclusion with the 210-server Qwen2.5-1.5B run. It is intentionally framed as an evidence audit for the original goal: the claim is not that a landscape can be drawn, but whether Hessian cross-terms explain complementarity, conflict, and non-commutative compression order.

Scope notes:
- `rho_H` is computed with a layer-local Hessian/Gauss-Newton proxy from activation covariance `X^T X`; it is not the exact full-model Hessian.
- Additivity rows use linearized perturbation sums `W + Delta_i + Delta_j`. Executable order gaps use actual sequential application and are reported separately.
- The older cross-run summary rows use 1016 PPL tokens and 8 zero-shot examples/task for Pythia, while Qwen2.5-1.5B uses 508 PPL tokens and 4 zero-shot examples/task; those cross-family numbers should be read qualitatively rather than as a strict leaderboard. The newer fair benchmark addendum is separate and uses 100 zero-shot examples/task for ARC-Easy and HellaSwag.
- These summarized runs were generated before the zero-shot-backup text loader was corrected to interleave tasks. Their metadata records candidate backup availability for `arc_easy,hellaswag`, but PPL/calibration text selection may be dominated by the first listed task. The runner now records actual task counts for future reruns.
- External GPTQ/AWQ/SparseGPT packages were unavailable in the tested environments. Native coverage is RTN, magnitude, Wanda-style activation-aware pruning, vanilla SVD, and activation-whitened SVD proxy. `slim_like_srq_proxy` is a fixed recipe proxy, not the official SLiM implementation.

## 210 Model Search

The successful 210 experiment stage found these relevant text-model candidates. This is an operator note, not a reproducible scan artifact from the metrics bundle:
- `/home/wangmeiqi/ZHuan/model/Qwen2.5-0.5B`
- `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B (used for this validation)`
- `/home/wangmeiqi/ZHuan/model/Qwen3-0.6B`
- `/home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct`
- `/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf`
- `Additional vision-language candidates exist, e.g. Qwen2.5-VL/LLaVA/Cambrian, but the current script is text-only.`

A later read-only re-search attempt in this turn was blocked before command execution by the 210 login shell's `nvm`/`PREFIX` initialization error, so the final list above is based on the already completed 210 experiment-stage discovery and the Qwen run artifact path. The same note is saved as `model_search_210_operator_note.md`.

## Experiment Configurations

| Run | Model | Params | Server/source | Modules | Bits | Keep | Rank | Q | S | R | PPL tokens | Zero-shot examples | Text source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70M_default_4bit_6mods | EleutherAI/pythia-70m | 70.4M | 236 GPU server, Hugging Face model id | 6 | 4 | 0.5000 | 0.5000 | rtn | wanda | whitened_svd | 1,016 | arc_easy:8; hellaswag:8 | zero_shot_backup:arc_easy,hellaswag |
| 70M_mid_3bit_12mods | EleutherAI/pythia-70m | 70.4M | 236 GPU server, Hugging Face model id | 12 | 3 | 0.5000 | 0.5000 | rtn | wanda | whitened_svd | 1,016 | arc_easy:8; hellaswag:8 | zero_shot_backup:arc_easy,hellaswag |
| 70M_strong_2bit_12mods | EleutherAI/pythia-70m | 70.4M | 236 GPU server, Hugging Face model id | 12 | 2 | 0.4000 | 0.4000 | rtn | wanda | whitened_svd | 1,016 | arc_easy:8; hellaswag:8 | zero_shot_backup:arc_easy,hellaswag |
| 160M_mid_3bit_12mods | EleutherAI/pythia-160m | 162.3M | 236 GPU server, Hugging Face model id | 12 | 3 | 0.5000 | 0.5000 | rtn | wanda | whitened_svd | 1,016 | arc_easy:8; hellaswag:8 | zero_shot_backup:arc_easy,hellaswag |
| Qwen2.5_1.5B_mid_3bit_12mods | /home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B | ~1.5B | 210 A800 server, local path /home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B | 12 | 3 | 0.5000 | 0.5000 | rtn | wanda | whitened_svd | 508 | arc_easy:4; hellaswag:4 | zero_shot_backup:arc_easy,hellaswag |

### Selected Layer Names

- `70M_default_4bit_6mods`: gpt_neox.layers.0.mlp.dense_4h_to_h, gpt_neox.layers.0.mlp.dense_h_to_4h, gpt_neox.layers.3.mlp.dense_4h_to_h, gpt_neox.layers.3.mlp.dense_h_to_4h, gpt_neox.layers.5.mlp.dense_4h_to_h, gpt_neox.layers.5.mlp.dense_h_to_4h
- `70M_mid_3bit_12mods`: gpt_neox.layers.0.attention.dense, gpt_neox.layers.0.attention.query_key_value, gpt_neox.layers.0.mlp.dense_4h_to_h, gpt_neox.layers.0.mlp.dense_h_to_4h, gpt_neox.layers.3.attention.dense, gpt_neox.layers.3.attention.query_key_value, gpt_neox.layers.3.mlp.dense_4h_to_h, gpt_neox.layers.3.mlp.dense_h_to_4h, gpt_neox.layers.5.attention.dense, gpt_neox.layers.5.attention.query_key_value, gpt_neox.layers.5.mlp.dense_4h_to_h, gpt_neox.layers.5.mlp.dense_h_to_4h
- `70M_strong_2bit_12mods`: gpt_neox.layers.0.attention.dense, gpt_neox.layers.0.attention.query_key_value, gpt_neox.layers.0.mlp.dense_4h_to_h, gpt_neox.layers.0.mlp.dense_h_to_4h, gpt_neox.layers.3.attention.dense, gpt_neox.layers.3.attention.query_key_value, gpt_neox.layers.3.mlp.dense_4h_to_h, gpt_neox.layers.3.mlp.dense_h_to_4h, gpt_neox.layers.5.attention.dense, gpt_neox.layers.5.attention.query_key_value, gpt_neox.layers.5.mlp.dense_4h_to_h, gpt_neox.layers.5.mlp.dense_h_to_4h
- `160M_mid_3bit_12mods`: gpt_neox.layers.0.attention.dense, gpt_neox.layers.0.attention.query_key_value, gpt_neox.layers.0.mlp.dense_4h_to_h, gpt_neox.layers.0.mlp.dense_h_to_4h, gpt_neox.layers.6.attention.dense, gpt_neox.layers.6.attention.query_key_value, gpt_neox.layers.6.mlp.dense_4h_to_h, gpt_neox.layers.6.mlp.dense_h_to_4h, gpt_neox.layers.11.attention.dense, gpt_neox.layers.11.attention.query_key_value, gpt_neox.layers.11.mlp.dense_4h_to_h, gpt_neox.layers.11.mlp.dense_h_to_4h
- `Qwen2.5_1.5B_mid_3bit_12mods`: model.layers.0.mlp.down_proj, model.layers.0.mlp.up_proj, model.layers.0.self_attn.o_proj, model.layers.0.self_attn.q_proj, model.layers.14.mlp.down_proj, model.layers.14.mlp.up_proj, model.layers.14.self_attn.o_proj, model.layers.14.self_attn.q_proj, model.layers.27.mlp.down_proj, model.layers.27.mlp.up_proj, model.layers.27.self_attn.o_proj, model.layers.27.self_attn.q_proj

## Cross-Run Result Summary

| Run | rho add. | rho PPL | rho zero-shot | Taylor-loss | Frob-loss | Trace-loss | Spectrum-order | Base PPL | Hessian PPL | Naive QSR PPL | Default QSR PPL | SLiM-proxy PPL | Hessian-Naive | Hessian-Fixed | Hessian-SLiM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70M_default_4bit_6mods | 0.1249 | 0.2941 | 0.2877 | -0.0526 | 0.1290 | -0.1517 | -0.4685 | 68.5878 | 97.0526 | 5874.5 | 105.4 | 102.3 | -5777.5 | -8.3290 | -5.2595 |
| 70M_mid_3bit_12mods | 0.1449 | 0.1748 | -0.0503 | 0.7246 | 0.5001 | 0.4592 | 0.1409 | 68.5878 | 353.4 | 1.020e+05 | 398.1 | 455.2 | -1.017e+05 | -44.7821 | -101.8 |
| 70M_strong_2bit_12mods | 0.4842 | 0.2165 | 0.1164 | 0.5367 | 0.4857 | 0.2896 | 0.4261 | 68.5878 | 3.607e+07 | 1.048e+08 | 1.042e+08 | 5.899e+07 | -6.868e+07 | -6.818e+07 | -2.292e+07 |
| 160M_mid_3bit_12mods | 0.0654 | 0.2932 | 0.0400 | 0.4610 | 0.1012 | 0.1441 | 0.3635 | 45.9731 | 112.5 | 6510.1 | 135.7 | 113.5 | -6397.6 | -23.1552 | -0.9880 |
| Qwen2.5_1.5B_mid_3bit_12mods | 0.3876 | 0.0875 | 0.1286 | 0.4798 | 0.1869 | 0.4103 | 0.5261 | 12.3748 | 17.2540 | 29.3551 | 18.4215 | 17.1197 | -12.1011 | -1.1675 | 0.1342 |

## MVP Criteria Check

1. Hessian cosine heatmaps exist for q/s/r on all pretrained runs, including Qwen2.5-1.5B at `pretrained_orthogonality_qwen25_1p5b_mid_20260624/figures/hessian_cosine_heatmap.png`.
2. High `rho_H` predicting additivity is supported in the stress/Qwen settings but not uniformly. Pythia-70M strong gives Spearman 0.4842; Qwen2.5-1.5B gives 0.3876; Pythia-160M mid is weak at 0.0654.
3. Order non-commutativity is observable. Qwen's largest order-gap row is `L27:down_proj` with `rq vs qr` and absolute loss gap 0.1874. Across settings, singular-spectrum shifts explain order gaps more consistently than symmetric Hessian overlap: Qwen spectrum-order 0.5261; Pythia-160M spectrum-order 0.3635.
4. Hessian-guided layer-wise selection beats both true naive QSR and default fixed QSR in the older strategy rows, including Qwen where PPL is 17.2540 vs naive QSR 29.3551 and default QSR 18.4215. It does not always beat the SLiM-like fixed recipe proxy: Qwen is 17.2540 vs SLiM-proxy 17.1197. The newer fair benchmark is stricter: Hessian-guided Q+S+R has lower predicted Hessian cost but worse split-window PPL than both fixed rotated QSR/RQS stacks, while Hessian-guided-SPQ improves PPL over fixed SPQ-like under matched SPQ prior and matched LoRA recovery budget but has lower mean zero-shot accuracy.

## Method Effectiveness Analysis

The strongest evidence for the framework is the Taylor/cross-term diagnostic rather than raw `rho_H` alone. In matched Pythia mid settings, Taylor-vs-loss is 0.7246 for 70M and 0.4610 for 160M, both above the Frobenius baseline (0.5001 and 0.1012). In the larger Qwen run, Taylor-vs-loss remains useful at 0.4798, above Frobenius 0.1869, and also above/near trace-only 0.4103.

Raw `rho_H` is a partial predictor. It tracks linearized additivity under stronger perturbation and the Qwen family check (0.4842 and 0.3876), but it is weak on Pythia-160M mid (0.0654) and only weakly connected to real PPL/zero-shot degradation in Qwen (0.0875, 0.1286). The conclusion should therefore be: Hessian cross-terms are useful as a diagnostic feature and selector input, not a universal scalar predictor by themselves.

The layer-wise selector is promising as a diagnostic but not competitive as a standalone recipe. In the older strategy rows it improves over true naive Q->S->R and the default fixed Q->S->R baseline under the same compression budget. Against a stronger SLiM-like fixed recipe, it wins on the Pythia runs, narrowly on 160M (112.5 vs 113.5), but loses slightly on Qwen (17.2540 vs 17.1197). In the stricter split-data Pythia-70M fair benchmark, however, `hessian_guided_qsr_budget` reaches the lowest predicted Hessian cost among same-budget QSR stacks but has worse PPL (+64.10%) than fixed `qsr_rotated_wanda_whitened` (+57.64%) and fixed `rqs_rotated_wanda_whitened` (+59.54%). It only beats the weaker `slim_like_srq_proxy` in that specific split run (+67.45%). The stronger result is recipe-conditioned and PPL-only: with the SPQ layer-type prior, Hessian-guided method/order selection improves no-LoRA SPQ-like PPL delta from +12.31% to +11.01%, and equal rank4/5-step LoRA recovery from +13.53% to +9.74% on disjoint recovery/evaluation text windows, but mean zero-shot delta is lower than fixed SPQ-like in both cases. That means the paper claim should be phrased as evidence that Hessian guidance can improve the PPL side of fixed ensemble recipes when constrained by a sensible prior, not as a standalone all-purpose or Pareto-dominant Q+S+R recipe.

The updated method diagnosis and revision plan is in `method_revision_after_fair_benchmark.md`. Its main conclusion is that local Hessian/cross-term features should become inputs to a constrained, multi-objective selector with SPQ/SLiM-like layer priors, activation reconstruction, trace/Frobenius sensitivity, rotation/outlier penalties, and held-out zero-shot proxies. The current single-objective Hessian-cost selector should not be used as the paper's final algorithmic claim.

The selected execution path is in `selected_execution_analysis.md`. It prioritizes split-data fair SPQ-prior validation on a larger local model, then selector ablations, then a multi-objective selector, while explicitly deprioritizing unconstrained QSR sweeps, result-selected lossless frontier claims, and landscape-only figures as main evidence.

The selected execution fallback has been run on Pythia-160M because the 210 Qwen server was busy at launch time. The result is summarized in `selected_execution_result_pythia160m.md`: Hessian-guided-SPQ again improves PPL over fixed SPQ-like at matched memory, but it is not Pareto-dominant because zero-shot accuracy is tied without LoRA and worse with LoRA.

The revised orthogonality-as-filter experiment has now been run on the same Pythia-160M fair protocol. `orthofilter_spq_refine` keeps the SPQ-like prior, rejects positive conditional-Hessian conflicts before ranking, then scores candidates with Hessian, activation reconstruction, worst-token risk, and zero-shot choice-text proxy. At matched 0.196 memory it improves zero-shot tradeoff but does not beat Hessian-guided SPQ on PPL. The residual-enabled variant improves both PPL and zero-shot but raises additive memory to 0.258, so it is promising compensation evidence rather than a same-budget win. Details are in `orthofilter_refinement_result_pythia160m.md`.

## Comparison With Existing Work

Existing compression work such as SLiM-like fixed recipes, QSLR-style quantization/low-rank combinations, LoSparse-style low-rank+sparse decomposition, and LQ-LoRA-style low-rank/quantized adaptation already covers much of the algorithmic combination space. The distinct contribution supported here is not 'Q+S+R works', but a measurement layer that asks which pairs conflict or complement in a layer and why the order changes the outcome.

Compared with those fixed or algorithm-specific recipes, this framework exposes: pairwise Hessian overlap heatmaps for Q/S/R perturbations; linearized additivity error tied to cross-terms; executable order gaps tied partly to singular-spectrum shifts; and a layer-wise choice rule. The current evidence is enough to motivate the framework, but not enough to claim dominance over official SLiM, GPTQ/AWQ, or SparseGPT implementations because those packages were not run here.

## SPQ Addendum

SPQ (<https://github.com/JiaminYao/SPQ_LLM_Compression/>) should be treated as a strong recent ensemble-compression baseline rather than as redundant prior work. It uses a fixed recipe: SVD on attention projections, activation-based structured pruning on MLP layers, global INT8 linear quantization, and LoRA recovery. This is close in spirit to a Q/S/R compression recipe, but it does not measure Hessian cross-terms, additivity error, or order gaps. The detailed comparison is in `spq_comparison_analysis.md`.

The requested SPQ-like smoke comparison has now been executed. The native runner includes `spq_like_rsq_no_lora`, `spq_like_rsq_lora`, `hessian_guided_spq_no_lora`, and `hessian_guided_spq_lora`. On Pythia-70M, guided SPQ reduced PPL from 1194.7 to 259.7 without LoRA and from 1013.4 to 235.6 with equal tiny LoRA recovery. On Qwen2.5-1.5B, guided SPQ reduced PPL from 64.7770 to 50.4522 without LoRA and from 62.5219 to 49.7825 with equal recovery. These are smoke/diagnostic results, not a deployable official SPQ reproduction; the memory budget is matched by nominal bits/keep/rank and the recovery budget by LoRA rank/steps/lr.

## Rotation And Low-Loss Addendum

Rotation quantization has been added as a native Hadamard-basis `rotated_rtn` proxy. The new rotation smoke runs show that it reduces Qwen attention median quantization error from 0.1953 to 0.1411 and median Hessian Q cost from 17.904 to 8.244, while Pythia remains mixed. A conservative `low_loss_triple_stack` also now evaluates all three operations Q+S+R under a benchmark-drop threshold; both new smoke runs pass the PPL-drop <1% criterion. An added Pythia ARC-Easy100 benchmark check gives baseline accuracy 0.28 vs low-loss triple-stack 0.32, i.e. 0.0000% benchmark drop. Details are in `rotation_low_loss_addendum.md`.

## Lossless Frontier Addendum

The matched single-method frontier has been added in `lossless_frontier_addendum.md`. On a Pythia-70M 4-module PPL smoke run, Q-only, S-only, R-only, and a low-memory Q+S+R slice were searched under the same `<1%` PPL-drop rule. The best single method is Q-only rotated RTN at 4-bit with nominal memory ratio 0.25. The best passing Q+S+R stack is `rqs`, q=rotated RTN, s=Wanda, r=whitened SVD, bits=4, keep=0.8, rank=0.5, with factorized-rank nominal memory ratio 0.133 and 0.0000% PPL benchmark drop. This should be treated as hypothesis-generating search evidence, not a fair lossless-dominance claim; the lossless stacking claim needs pre-registered split-data validation with zero-shot and larger PPL windows.

## Fair Benchmark Addendum

The fair benchmark in `fair_benchmark_addendum.md` is the stricter result for paper-style comparison. It evaluates predeclared Q-only, S-only, R-only, fixed Q+S+R, SLiM-like proxy, SPQ-like, and Hessian-guided variants on a split text protocol: 32 calibration texts, 64 PPL-evaluation texts, and 64 LoRA-recovery texts, plus ARC-Easy/HellaSwag 100-example zero-shot. Under this protocol, the aggressive Q+S+R stacks do not beat the higher-memory single-method quality references: `s_only_magnitude_keep0p8` has memory 0.800 and +0.05% PPL, while Q+S+R rows have memory 0.133. At memory 0.133, fixed `qsr_rotated_wanda_whitened` has +57.64% PPL and beats Hessian-guided-QSR's +64.10%; the SLiM proxy is worse at +67.45%. At SPQ-like memory 0.196, Hessian-guided-SPQ improves PPL versus fixed SPQ-like both without LoRA (+11.01% vs +12.31%) and with equal tiny LoRA recovery (+9.74% vs +13.53%), but it is not Pareto-dominant because zero-shot mean is lower. Therefore the frontier result should be framed as search evidence, and the strongest fair claim is recipe-conditioned SPQ PPL improvement rather than broad Q+S+R dominance.

## Figures

- `figures/correlation_by_config_model_qwen.svg`
- `figures/strategy_ppl_by_config_model_qwen.svg`
- `figures/mid_config_model_family_comparison_qwen.svg`
- `figures/additivity_scatter_representative_qwen.svg`
- `figures/spq_fixed_vs_guided_summary.svg`
- `figures/rotation_low_loss_summary.svg`
- `../pretrained_orthogonality_pythia70m_lossless_frontier_4mods_factorized672_20260627/figures/lossless_frontier_summary.png`
- `../pretrained_orthogonality_pythia70m_fair_benchmark_4mods_arc_hella100_20260627/figures/fair_benchmark_summary.png`
- `../pretrained_orthogonality_pythia70m_fair_benchmark_guided_4mods_arc_hella100_20260627/figures/fair_benchmark_guided_competitiveness.png`
- `../pretrained_orthogonality_pythia70m_fair_benchmark_extended_4mods_arc_hella100_lora5_20260627/figures/fair_benchmark_extended_competitiveness.png`
- `../pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3/figures/fair_benchmark_extended_competitiveness.png`
- `figures/selector_failure_diagnostic.png`

## Files

- `summary.csv`: machine-readable cross-run metrics.
- `experiment_configurations.csv`: exact setup details, selected layers, data source fingerprints, and evaluation budgets.
- `strategy_rows.csv`: raw strategy-level PPL/accuracy rows.
- `method_status_all_runs.csv`: available/native/proxy/unavailable method status for Q/GPTQ/AWQ, S/SparseGPT, and R/SVD-LLM-style methods.
- `model_search_210_operator_note.md`: operator note for the 210 model paths used to choose the Qwen validation target.
- `spq_comparison_analysis.md`: SPQ method/baseline comparison and recommended integration experiment.
- `spq_execution_summary.md` and `spq_execution_summary.csv`: implemented SPQ-like baselines, Pythia/Qwen smoke metrics, and artifact links.
- `rotation_low_loss_addendum.md` and `rotation_low_loss_summary.csv`: rotation quantization analysis and <1% low-loss Q+S+R smoke evidence.
- `lossless_frontier_addendum.md`: matched Q-only/S-only/R-only/Q+S+R lossless frontier table and interpretation.
- `fair_benchmark_addendum.md`: fixed-recipe and calibration-only Hessian-guided PPL/zero-shot benchmark comparison without result-based row selection, including SLiM-like and SPQ-like rows.
- `method_revision_after_fair_benchmark.md`: failure analysis and revised selector design after the split-data fair benchmark.
- `orthofilter_refinement_result_pythia160m.md`: conditional-Hessian filter, activation/worst-token/choice-text proxy, and residual SPQ-prior refinement result on Pythia-160M.
- `selected_execution_analysis.md`: selected execution path, model/recipe priorities, commands, evidence package, and stop criteria derived from the original goal text.
- `selected_execution_result_pythia160m.md`: executed Pythia-160M fallback result, fair benchmark table, diagnostic correlations, and updated claim assessment.
