# SPQ Comparison Addendum

This addendum analyzes SPQ (SVD-Pruning-Quantization) from the public code at <https://github.com/JiaminYao/SPQ_LLM_Compression/> and the arXiv paper <https://arxiv.org/abs/2602.18420>. It compares SPQ with the current Hessian-overlap compression framework and with the baselines reported by SPQ.

## SPQ Implementation Summary

SPQ is a fixed ensemble compression recipe:

1. **R / SVD on attention projections.** `methods/svd.py` selects LLaMA/OPT attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj` / `out_proj`). It computes exact SVD, chooses the minimum rank preserving a variance threshold, and replaces a linear layer by two low-rank linear factors when the factorization reduces parameter count.
2. **S / structured activation pruning on MLP.** `methods/pruning.py` collects forward activation statistics on target layers. The default path uses MLP layers, computes per-layer log-inverse pruning ratios, then prunes hidden neurons consistently through `up_proj`/`gate_proj` rows and `down_proj` columns.
3. **Q / INT8 linear quantization on all linear layers.** `methods/linear_quant.py` replaces linear layers with tensor-wise, channel-wise, or hybrid int8 wrappers. Hybrid choices are name-based, percentile-based, or mean/std based.
4. **LoRA recovery.** `methods/train.py` wraps target compressed modules with trainable LoRA adapters and fine-tunes briefly on WikiText-103.

The default order in `main.py` is therefore `R(attention) -> S(MLP) -> Q(all linear) -> LoRA recovery`. This is not an order-search method; it is a hand-designed layer-type assignment.

## SPQ Reported Baselines And Effects

SPQ compares against several levels of baseline:

| Baseline group | What it tests | Reported SPQ relationship |
|---|---|---|
| Single methods | SVD-only, pruning-only, quantization-only | SPQ reports better perplexity at matched or stronger compression, arguing that the three operations are complementary. |
| Pairwise combinations | SVD+Q, pruning+Q, SVD+pruning | SPQ uses these to show additional memory savings without significant PPL damage relative to quant-only in the reported setting. |
| External strong baselines | ASVD, SparseGPT, GPTQ | On LLaMA-2-7B, SPQ reports 75% memory reduction, WikiText-2 PPL 4.91, C4 PPL 7.11, and average zero-shot accuracy 0.60; GPTQ reports 73% memory reduction, WikiText-2 PPL 5.48, C4 PPL 6.66, average zero-shot accuracy 0.62. |
| Cross-model scaling | LLaMA-3.2-1B/3B, LLaMA-2-7B, OPT-1.3B/2.7B/6.7B, Vicuna-7B, Mistral-7B | SPQ reports 62-74% memory reduction; PPL improves on several larger models and slightly worsens on LLaMA-3.2-1B and Mistral-7B. |
| Throughput | GPTQ-8bit and GPTQ-4bit | SPQ reports higher generation throughput at comparable compression levels. |

The main empirical advantage is pragmatic: it obtains a strong memory/PPL trade-off by matching each operation to a layer family that empirically tolerates it: low-rank for attention, activation pruning for MLP, and int8 quantization globally.

## Code-Level Strengths

- **Simple modular recipe.** The code is small and readable: SVD, pruning, quantization, and LoRA recovery are separate modules.
- **Layer-type specialization.** Unlike a uniform recipe, SPQ avoids applying SVD or pruning everywhere by default. This is compatible with our own observation that fixed Q/S/R orders and layer choices matter.
- **Recovery phase.** LoRA fine-tuning gives SPQ an advantage over strictly post-training, no-update baselines in quality recovery.
- **Deployment motivation.** It measures memory, PPL, and throughput instead of only reporting loss-landscape or proxy statistics.

## Code-Level Limitations

- **No Hessian/cross-term criterion.** SPQ does not measure whether SVD, pruning, and quantization are complementary in a given layer. It assumes a fixed layer-type assignment and validates it empirically.
- **No order-gap analysis.** The implementation applies `R -> S -> Q -> LoRA` and does not test `Q -> S -> R`, `S -> R -> Q`, or layer-wise order choices.
- **Quantization memory accounting is theoretical.** `QuantLinearTensorWise` and `QuantLinearChannelWise` keep `weight_fp32` as a parameter while also storing `quantized_weight`. `get_mem_size_gb()` counts only the int8 buffer and bias for those modules, so reported weight memory can understate actual PyTorch runtime/state_dict memory unless the fp32 copy is removed for deployment.
- **Quantization forward path dequantizes to float.** The wrapper computes `quantized_weight.float() * scale` before `linear`, so it is not an integer-kernel implementation. Throughput gains likely come mostly from structural SVD/pruning and smaller matrix shapes, not from int8 kernel acceleration.
- **PPL implementation is a short fixed-token proxy.** `calculate_perplexity()` truncates to a 512-token padded sample rather than a standard sliding-window WikiText-2 evaluation. This makes direct PPL comparison with lm-eval/GPTQ papers risky.
- **SVD is exact and potentially expensive.** `torch.linalg.svd` over large projection matrices is simple but memory-heavy; it may not scale smoothly to larger models without randomized or block SVD.
- **LoRA makes method comparisons non-equivalent.** SPQ includes a recovery-training phase. It should be compared either to baselines with comparable recovery/fine-tuning budgets or reported separately as "compression + recovery."

## Comparison With Our Hessian-Overlap Framework

| Axis | SPQ | Current framework |
|---|---|---|
| Main contribution | A strong fixed ensemble recipe: attention SVD + MLP pruning + global int8 + LoRA recovery. | A diagnostic/selection framework: Hessian overlap, additivity error, order gap, and layer-wise choice. |
| Compression operations | SVD, activation pruning, int8 quantization. | RTN Q, magnitude/Wanda S, vanilla/whitened SVD R; can be extended to SPQ variants. |
| Complementarity evidence | Empirical pairwise and final recipe results. | Direct `rho_H`, additivity error, Taylor/cross-term prediction, and order-gap measurements. |
| Order handling | Fixed `R -> S -> Q` pipeline. | Explicit executable order comparisons such as `R->Q/S` vs `Q/S->R`. |
| Layer choice | Hand-coded by layer type. | Layer-wise selection from estimated Hessian costs and method/order choices. |
| Recovery training | Yes, LoRA. | No recovery training in current summarized runs. |
| Strongest evidence so far | Memory/PPL/throughput on larger 1B-7B models. | Cross-term/Taylor diagnostics and order/spectrum analysis; older layer-wise rows beat naive/default fixed QSR, while the stricter split-data result supports only constrained SPQ-like PPL improvement rather than standalone QSR competitiveness. |

The two works are therefore complementary rather than redundant. SPQ is evidence that a layer-aware fixed Q/S/R-style recipe can work well at scale. Our framework asks why that recipe works, when it fails, and whether a layer-wise or order-wise criterion can predict better alternatives.

## Does Our Existing Method Have Advantage?

Current evidence supports these advantages:

- **Interpretability.** We can explain pair conflict via `rho_H`, additivity error, and Taylor/cross-term terms. SPQ mostly relies on empirical sweeps.
- **Order sensitivity.** Our order-gap tables expose non-commutativity. SPQ does not test order.
- **Selector potential under constraints.** Older Pythia and Qwen summary rows show Hessian-guided layer-wise choice beating true naive/default fixed QSR, but the stricter split-data run does not support standalone QSR competitiveness. The more defensible advantage is PPL improvement inside an SPQ-like layer prior.

Current evidence does not support these stronger claims:

- We cannot claim better deployment compression than SPQ because our runs are small/medium diagnostic runs, not 7B deployment runs with memory/throughput benchmarking.
- We cannot claim superiority over GPTQ/AWQ/SparseGPT official implementations because those were not run in our environment.
- We cannot claim `rho_H` alone is a universal degradation predictor; Qwen shows good additivity correlation but weak PPL/zero-shot correlation.

## Improvement Opportunities

1. **Add SPQ as a fixed-recipe baseline.** Implement `spq_like_rsq`: attention-only SVD, MLP-only activation pruning, int8/RTN quantization, optional LoRA recovery. Compare against our current `slim_like_srq_proxy`.
2. **Use Hessian criteria to choose SPQ deviations.** Keep SPQ as the default prior, but override per layer when Hessian overlap predicts conflict, e.g. skip SVD in high `rho_H(R,Q)` attention layers or swap order when order-gap proxy is high.
3. **Measure SPQ-style operations in the same diagnostic table.** For each layer, compute `rho_H` and additivity for SPQ's actual `R(attn)`, `S(mlp)`, `Q(all)` perturbations rather than generic q/s/r on every selected module.
4. **Separate "compression only" and "compression + recovery."** Evaluate no-LoRA SPQ-like, LoRA-only recovery, and Hessian-guided+LoRA variants to make comparisons fair.
5. **Fix quantized deployment accounting.** For a deployable SPQ baseline, remove or exclude fp32 shadow weights and use an actual int8/packed backend or explicitly label memory as theoretical weight memory.
6. **Use standardized eval.** Reuse the same PPL and zero-shot evaluator across SPQ, GPTQ/AWQ/SparseGPT, and our method: WikiText-2/C4 sliding-window PPL plus ARC/HellaSwag/PIQA/WinoGrande/TruthfulQA/GSM8K where feasible.

## Executed SPQ-Like Smoke Experiment

The recommended integration experiment has been implemented in the native pretrained-LLM runner:

1. `spq_like_rsq_no_lora`: attention modules use `R->Q`; MLP modules use `S->Q`; other selected linear modules use `Q`.
2. `spq_like_rsq_lora`: the same fixed recipe with LoRA recovery.
3. `hessian_guided_spq_no_lora`: the same nominal bits/keep/rank budget and SPQ layer-type prior, but Hessian-guided order and S/R method choices.
4. `hessian_guided_spq_lora`: the guided variant with the same LoRA rank, step, and learning-rate recovery budget as fixed SPQ-like.

Low-budget smoke results are recorded in `spq_execution_summary.md`, `spq_execution_summary.csv`, and `figures/spq_fixed_vs_guided_summary.svg`. On Pythia-70M, guided SPQ improved PPL from 1194.7 to 259.7 without LoRA and from 1013.4 to 235.6 with equal LoRA recovery. On Qwen2.5-1.5B, guided SPQ improved PPL from 64.7770 to 50.4522 without LoRA and from 62.5219 to 49.7825 with equal recovery.

This strengthens the paper claim in a limited but concrete sense: Hessian cross-term/order diagnostics can modify a strong fixed ensemble recipe and improve its PPL under matched nominal compression and recovery budgets. The result should still be described as smoke/diagnostic evidence. It does not yet reproduce official SPQ memory accounting, INT8 kernels, standardized WikiText/C4 PPL, full zero-shot suites, or saturated LoRA recovery.

## Split-Data Fair Update

The stricter split-data benchmark changes the claim strength. In `pretrained_orthogonality_pythia70m_fair_benchmark_extended_split_4mods_arc_hella100_lora5_20260627_v3`, calibration, PPL evaluation, and LoRA recovery use disjoint text windows, and zero-shot uses ARC-Easy/HellaSwag with 100 examples per task.

Under this protocol, Hessian-guided SPQ remains favorable on PPL but not on Pareto quality. At the same SPQ layer prior and nominal memory ratio 0.196, no-LoRA PPL delta improves from +12.31% to +11.01%, and equal rank4/5-step LoRA PPL delta improves from +13.53% to +9.74%. Mean zero-shot delta is lower than fixed SPQ-like in both cases: +0.010 vs +0.030 without LoRA, and +0.015 vs +0.035 with LoRA.

The correct interpretation is therefore narrower than the smoke result: Hessian guidance can improve the PPL side of an SPQ-like recipe under matched memory and recovery budgets, but the current selector is not Pareto-dominant and should be revised into a constrained multi-objective selector. The detailed failure analysis and revised selector design are in `method_revision_after_fair_benchmark.md`.

The next higher-confidence experiment is to repeat this on a larger LLaMA/Qwen checkpoint with more calibration/eval text, official GPTQ/AWQ/SparseGPT comparisons where available, and a deployable packed-memory accounting path.
