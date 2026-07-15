# Selected Execution Analysis

This file selects the reasonable execution path from the original Compression Orthogonality Landscape plan. The purpose is to avoid spending compute on attractive but low-evidence analyses, and to focus on experiments that directly test the paper claim: whether Hessian cross-terms explain complementarity, conflict, and order non-commutativity.

## Decision

The next execution should prioritize a **split-data, fair, SPQ-prior analysis on a larger local model**, plus the same diagnostic correlations already required by the original goal.

Do not prioritize another unconstrained Hessian-guided QSR run as the main story. The current fair split-data result already shows that unconstrained Hessian-guided QSR is not a reliable competitive recipe: it has the lowest predicted Hessian cost among same-memory QSR rows, but worse PPL than fixed rotated QSR/RQS. The strongest remaining path is therefore constrained selection inside a sensible layer prior.

## Selected Main Analysis

| Item | Choice |
| --- | --- |
| Main model | Qwen2.5-1.5B local checkpoint on 210, then Pythia-160M/410M on 236 if Qwen is blocked |
| Main recipe family | SPQ-like layer prior: attention R+Q, MLP S+Q, global Q |
| Selector comparison | fixed SPQ-like vs Hessian-guided-SPQ at the same memory and same LoRA recovery budget |
| Diagnostic metrics | `rho_H`, additivity error, order gap, singular-spectrum order explanation, PPL delta, zero-shot delta |
| Fairness rule | disjoint calibration / PPL-evaluation / LoRA-recovery text windows; no selection by final PPL or accuracy |
| Baselines kept in the table | Q-only, S-only, R-only, fixed QSR/RQS, SLiM-like proxy, fixed SPQ-like, Hessian-guided-SPQ |
| Deprioritized analyses | pretty 2D landscapes, result-selected lossless frontier claims, unconstrained QSR as final method |

This is the best fit to the pasted plan because it tests all four MVP requirements while respecting the negative fair benchmark result.

Verified 210 prerequisites on 2026-06-27:

- Model path exists: `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B`.
- Runner path exists: `/home/wangmeiqi/codex_llm_spectral_dynamics/scripts/run_pretrained_llm_orthogonality.py`.
- Recommended conda env: `Qwen3` has `torch 2.6.0` and `transformers 4.51.0`; `base-2-bitnet` also has torch/transformers and can be a fallback.
- The 210 login shell still prints an `nvm`/`PREFIX` warning. Include `unset PREFIX` in the remote command. The warning appears before the command body but does not block short read-only checks.
- When using `ssh-run.ps1`, pass both `-UseWorkDir` and `-WorkDir`; `-WorkDir` alone does not change directories.

## Why This Is The Reasonable Path

### 1. It directly tests the original contribution

The original text says the paper is not about inventing the N-th hybrid compression pipeline. The selected analysis therefore keeps the claim on the diagnostic layer:

- Do high `rho_H` pairs show larger additivity error?
- Do order gaps correlate with conditional Hessian overlap or singular-spectrum shift?
- Can Hessian features improve a recipe when the layer prior is held fixed?
- Does the improvement survive real PPL and zero-shot evaluation?

### 2. It avoids the known failure mode

The latest split-data Pythia-70M run shows this same-memory QSR mis-ranking:

| Strategy | Predicted Hessian cost | PPL delta | Mean zero-shot delta |
| --- | ---: | ---: | ---: |
| fixed QSR | 21.2457 | +57.64% | -0.045 |
| fixed RQS | 14.3136 | +59.54% | -0.020 |
| Hessian-guided QSR | 12.3022 | +64.10% | -0.025 |

The selector minimized local Hessian cost but did not minimize held-out PPL. Running more unconstrained QSR before changing the selector objective is unlikely to strengthen the paper.

### 3. It tests the strongest positive signal

Under the SPQ prior, Hessian guidance improved PPL at the same memory ratio 0.196:

| Strategy | Recovery | PPL delta | Mean zero-shot delta |
| --- | --- | ---: | ---: |
| fixed SPQ-like | none | +12.31% | +0.030 |
| Hessian-guided-SPQ | none | +11.01% | +0.010 |
| fixed SPQ-like | LoRA rank4, 5 steps | +13.53% | +0.035 |
| Hessian-guided-SPQ | LoRA rank4, 5 steps | +9.74% | +0.015 |

This is not Pareto-dominant, but it is the best current evidence that Hessian-guided selection can improve a strong fixed ensemble recipe.

## Execution Tiers

### Tier 1: Larger-Model Fair SPQ-Prior Validation

Run this first. It is the most defensible experiment for the current paper claim.

Required outputs:

- `metrics/hessian_cosine.csv`
- `metrics/additivity.csv`
- `metrics/order_gap.csv`
- `metrics/correlations.csv`
- `metrics/fair_benchmark.csv`
- `metrics/fair_benchmark_zero_shot.csv`
- `metrics/fair_benchmark_selection.csv`
- `figures/hessian_cosine_heatmap.png`
- `figures/largest_order_gap_singular_spectrum.png`
- `figures/fair_benchmark_summary.png`

Primary judgment:

- If Hessian-guided-SPQ improves PPL and does not reduce mean zero-shot accuracy relative to fixed SPQ-like, it becomes a candidate method claim.
- If it improves only PPL but hurts zero-shot again, keep the claim as recipe-conditioned diagnostic value, not Pareto dominance.
- If it fails both PPL and zero-shot, the selector must be revised before more scale-up.

Recommended 210 Qwen command shape:

Local wrapper form:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Administrator\.codex\skills\ssh-dev\scripts\ssh-run.ps1" `
  -Profile 210 `
  -UseWorkDir `
  -WorkDir /home/wangmeiqi/codex_llm_spectral_dynamics `
  -CondaEnv Qwen3 `
  -Command "unset PREFIX; python scripts/run_pretrained_llm_orthogonality.py <args below>"
```

Remote command body:

```bash
python scripts/run_pretrained_llm_orthogonality.py \
  --model /home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B \
  --local-files-only \
  --output-dir results/pretrained_orthogonality_qwen25_1p5b_selected_fair_spq_YYYYMMDD \
  --module-types q_proj,o_proj,up_proj,down_proj \
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

If 210 is blocked by shell initialization or queue pressure, use the same analysis on 236 with `EleutherAI/pythia-160m` or `EleutherAI/pythia-410m`, depending on local model availability and GPU memory.

### Tier 2: Selector-Ablation Analysis

Run after Tier 1 or implement if the next paper claim needs a stronger algorithmic section.

The current runner compares fixed SPQ-like against fully Hessian-guided-SPQ. It does not cleanly separate order selection from method selection. The next useful ablation is:

| Ablation | Purpose |
| --- | --- |
| fixed method + fixed order | SPQ-like baseline |
| fixed method + guided order | isolate order-gap usefulness |
| guided method + fixed order | isolate method-choice usefulness |
| guided method + guided order | current Hessian-guided-SPQ |
| guided method/order + LoRA | test recovery interaction |

This is higher value than adding more random QSR orders because it answers which part of Hessian guidance helps.

### Tier 3: Multi-Objective Selector

Only implement after Tier 1 confirms that the SPQ-prior path is worth improving.

The current selector objective is too narrow:

```text
minimize local Hessian cost only
```

The revised objective should combine:

- local Taylor/cross-term cost;
- activation reconstruction error;
- trace/Frobenius sensitivity;
- predicted order gap;
- rotation/outlier penalty for unrotated RTN in high-impact layers;
- memory-budget penalty;
- held-out calibration proxy for zero-shot when feasible.

This tier is necessary before claiming Pareto dominance.

## Analyses To Deprioritize

| Analysis | Reason to defer |
| --- | --- |
| 2D loss landscape as main result | Useful as a figure, but the text explicitly says visualization alone cannot prove framework effectiveness. |
| Lossless frontier selected by final PPL | Current evidence is search diagnostic; it risks benchmark-selection bias. |
| More unconstrained QSR sweeps | The current failure is objective mismatch, not lack of QSR candidates. |
| Official GPTQ/AWQ/SparseGPT integration before Tier 1 | Important for final comparison, but it does not answer whether the Hessian selector itself is valid. |
| HVP/full Hessian mode on small models before LLM fair split | Useful for theory, but it does not repair the current LLM selector weakness. |

## Minimum Evidence Package For The Paper

For any selected run to be usable in the paper, report this package together:

1. Hessian cosine heatmap over Q/S/R perturbations.
2. `rho_H` vs additivity error correlation.
3. Taylor/cross-term vs real loss/PPL degradation correlation, compared against Frobenius, parameter cosine, activation reconstruction, and trace-only baselines.
4. R-first vs Q/S-first order-gap table.
5. Singular-spectrum plot for the largest order gap.
6. Fair benchmark table with signed PPL delta, per-task zero-shot delta, mean zero-shot delta, memory ratio, and recovery budget.
7. Explicit conclusion: diagnostic-only, PPL-only recipe improvement, or Pareto improvement.

## Stop Criteria

Stop scaling this path if either condition repeats on the larger model:

- Hessian-guided-SPQ improves PPL but lowers zero-shot again, and the paper needs a Pareto method claim.
- Hessian/cross-term correlations remain weaker than Frobenius and trace-only after larger evaluation windows.

If either condition holds, the reasonable next step is not another benchmark run. It is selector redesign and ablation.
