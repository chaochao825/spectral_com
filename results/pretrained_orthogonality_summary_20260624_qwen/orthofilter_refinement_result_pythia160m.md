# Orthofilter SPQ Refinement Result on Pythia-160M

This addendum tests the revised direction: use Hessian orthogonality as a filter/constraint rather than a final selector, add conditional orthogonality, and evaluate residual compensation inside an SPQ-like layer prior.

## Configuration

- Model: `EleutherAI/pythia-160m`
- Server: 236, GPU0, conda `myenv`
- Target modules: 12 Pythia linears from first/middle/last layers: attention `dense`, `query_key_value`, MLP `dense_h_to_4h`, `dense_4h_to_h`
- Text protocol: disjoint split with 32 calibration batches, 64 PPL evaluation batches, 64 LoRA recovery batches
- Zero-shot: ARC-Easy and HellaSwag, 100 examples per task in the fair benchmark
- Compression budget for non-residual SPQ rows: 4-bit Q, keep=0.8 S, rank=0.5 R
- LoRA recovery budget: rank 4, alpha 32, 5 steps, lr 5e-5
- New selector:
  - candidate space follows SPQ layer prior: attention chooses `rq/qr`; MLP chooses `sq/qs`
  - optional residual candidates: attention can choose `q+r_res`; MLP can choose `q+s_res`
  - filter: reject candidates with positive conditional Hessian overlap above 0.25 unless all candidates fail
  - ranking after filter: Hessian cost + activation reconstruction + worst-token risk + zero-shot choice-text P95 risk + memory penalty

Artifacts:

- Remote: `/home/wangmeiqi/llm_spectral_dynamics/results/pretrained_orthogonality_pythia160m_orthofilter_spq_20260627`
- Local download: `E:\Codex_work\ssh_experiment\downloads\pretrained_orthogonality_pythia160m_orthofilter_spq_20260627`
- Focused figure: `figures/orthofilter_spq_summary.png`
- Summary copy: `results/pretrained_orthogonality_summary_20260624_qwen/figures/orthofilter_spq_summary.png`

## Fair Benchmark Result

Baseline PPL is 54.6310 and baseline mean zero-shot accuracy is 0.375.

| Method | Memory | PPL delta | Zero-shot delta | Hessian cost | Recovery |
|---|---:|---:|---:|---:|---|
| fixed SPQ-like | 0.196 | +13.03% | -0.055 | 7152.9 | none |
| Hessian-guided SPQ | 0.196 | +9.05% | -0.055 | 5710.6 | none |
| Orthofilter SPQ | 0.196 | +11.16% | -0.050 | 5820.8 | none |
| Orthofilter SPQ + residual candidates | 0.258 | +8.66% | -0.045 | 5793.1 | none |
| fixed SPQ-like + LoRA | 0.196 | +11.58% | -0.035 | 7152.9 | rank4 / 5 steps |
| Hessian-guided SPQ + LoRA | 0.196 | +8.06% | -0.065 | 5710.6 | rank4 / 5 steps |
| Orthofilter SPQ + LoRA | 0.196 | +9.76% | -0.050 | 5820.8 | rank4 / 5 steps |
| Orthofilter SPQ + residual + LoRA | 0.258 | +7.76% | -0.040 | 5793.1 | rank4 / 5 steps |

## Selection Behavior

All selected orthofilter candidates passed the conditional-rho filter; no selected row used fallback. This confirms the new row is not just a renamed Hessian-cost selector.

Non-residual orthofilter selection over 12 layers:

- `rq`: 5 layers
- `qr`: 1 layer
- `sq`: 3 layers
- `qs`: 3 layers

Residual-enabled orthofilter selection over 12 layers:

- `rq`: 4 layers
- `q+r_res`: 1 layer
- `qr`: 1 layer
- `sq`: 3 layers
- `qs`: 3 layers

The residual result is therefore not a blanket residual rewrite. The selector used residual low-rank only on `L0:query_key_value`, while retaining sequential SPQ-like choices elsewhere.

## Interpretation

The revised idea is partially supported, but not yet a same-budget win.

At the same nominal memory 0.196, Orthofilter SPQ improves zero-shot relative to Hessian-guided SPQ but gives worse PPL. With LoRA, the tradeoff is clearer: Orthofilter SPQ has +9.76% PPL and -0.050 zero-shot, while Hessian-guided SPQ has better PPL at +8.06% but worse zero-shot at -0.065. This supports the diagnosis that adding activation and worst-token/zero-shot-proxy terms changes the tradeoff away from PPL-only optimization, but it is not Pareto-dominant.

Residual compensation is the most promising direction in this run: the residual-enabled row improves both PPL and zero-shot versus the non-residual orthofilter and Hessian-guided SPQ. However, its additive memory accounting rises from 0.196 to 0.258. It should be reported as evidence that residual decomposition can buy useful compensation, not as a same-budget victory.

The correct method claim after this run is:

> Hessian orthogonality is more credible as a candidate filter and diagnostic than as a final scalar selector. Inside an SPQ-like prior, adding conditional filtering and activation/worst-token/choice-text proxy terms improves the zero-shot side of the tradeoff, while residual low-rank compensation gives the best quality but currently spends extra memory.

The next experiment should budget residual components explicitly: reduce residual rank or bits until the residual row matches 0.196 memory, then compare again against fixed SPQ-like, Hessian-guided SPQ, and Orthofilter SPQ under the same LoRA recovery budget.
