# OASR 与结构化残差压缩实验整理

日期：2026-07-02

## 目标

当前实验围绕一个最小 OASR 假设：

```text
W_hat = Q_T(W) + C_res + L_res
```

其中：

- `Q_T(W)`：低比特量化基座，当前实现包含 RTN、SINQ-like、rotated RTN。
- `C_res`：结构化满秩残差，主要是 block-circulant 及其 channel permutation 变体。
- `L_res`：低秩残差，用同等残差 memory 与 `C_res` 公平比较。

核心问题不是追求 SOTA，而是判断 `C_res` 是否在同等 memory budget 下有独立价值，以及 conditional Hessian / activation overlap 是否能解释它的有效性。

## 已实现方法

### 1. OASR smoke

脚本：

```text
scripts/run_oasr_structured_residual.py
```

候选：

- `Q only`
- `Q + L`
- `Q + C`
- `Q + C + L`
- `Q + L + C`
- simple selector

指标：

- memory ratio
- weight error
- activation error
- worst-token p95 error
- Hessian-cost norm proxy
- conditional overlap: `rho(Q,C)`, `rho(Q,L)`, `rho(Q+C,L)`, `rho(C,L)`
- residual effective/stable rank
- block-circulant projection error
- matched low-rank projection error
- PPL smoke

### 2. Matched-memory structured residual probe

脚本：

```text
scripts/run_structured_residual_matched.py
```

固定 `Q` base：

- RTN q4
- SINQ-like q4

比较 residual `E = W - Q(W)` 的近似：

- low-rank residual
- naive block-circulant
- norm-sorted block-circulant
- activation-clustered block-circulant
- random-permuted block-circulant
- Monarch-like two-block proxy

判定标准：

```text
E_act(C_structured) < E_act(L_matched)
```

如果 offline activation reconstruction 不赢，则不跑 PPL。

## 关键结果

### OASR smoke

结果目录：

```text
results/oasr_structured_residual_pythia70m_20260628
```

同预算 PPL smoke：

| target memory | Q+L PPL | Q+C+L PPL | 结论 |
|---:|---:|---:|---|
| 0.196 | 27.6658 | 66.0001 | Q+C+L 更差 |
| 0.220 | 27.2464 | 27.3875 | Q+C+L 更差 |
| 0.258 | 25.8054 | 27.0520 | Q+C+L 更差 |

诊断：

- block-circulant projection 相比 random structured baseline 是非随机有效的：208/208 胜。
- 但它在同等 residual memory 下没有超过 low-rank projection。
- selector 在 12 个 layer-budget row 中 0 次选择 `Q+C+L`。
- signed conditional overlap 多数为负或接近 0，但没有转化为 activation/PPL gain。

### Structured residual matched-memory probe

结果目录：

```text
results/structured_residual_matched_pythia70m_20260628
```

结果：

| 指标 | 数值 |
|---|---:|
| matched-memory comparisons | 120 |
| structured wins vs floor low-rank | 0 |
| structured wins vs ceil low-rank diagnostic | 0 |
| structured wins on weight error | 0 |

最接近的一例：

| 字段 | 值 |
|---|---|
| q base | RTN q4 |
| layer | `L0:dense_h_to_4h` |
| method | norm-sorted block-circulant |
| block size | 64 |
| structured activation error | 0.00338403 |
| matched low-rank activation error | 0.00332573 |
| delta | +5.82964e-05 |

## 当前判断

当前失败不太像单纯实现 bug：

- partial block padding bias 已修复为 observed-entry averaging。
- permutation-aware projection 的 unpermute 逻辑已有测试覆盖。
- permutation metadata 已计入 structured params。
- low-rank baseline 同时记录 non-overbudget floor rank 与 ceil-rank diagnostic。
- 即使用 ceil-rank low-rank，structured residual 仍然 0/120 胜。
- structured residual 连 Frobenius weight residual 也 0/120 胜。

更可能的问题是方法结构本身：

- block-circulant 每个 block 只保留 cyclic diagonal 均值，约束太强。
- channel permutation 只能重排 residual 能量，不能改变 residual 的低秩/频谱结构。
- 当前 residual 更接近可由 low-rank 捕获的方向，而不是 circulant 子空间。
- 这个版本的 Monarch-like 是 two-pass block-circulant proxy，不是真正 learned Monarch/Butterfly product。

因此当前保守结论是：

> block-circulant / simple channel permutation / proxy Monarch-like structured residual 方向暂时停止；除非后续引入 learned structured residual、activation-aware residual decomposition、或更强的 Monarch/Butterfly parameterization，否则不值得继续跑 PPL 或 zero-shot。

## 结果文件

- `results/oasr_structured_residual_pythia70m_20260628/summary.md`
- `results/oasr_structured_residual_pythia70m_20260628/candidate_metrics.csv`
- `results/oasr_structured_residual_pythia70m_20260628/strategy_performance.csv`
- `results/structured_residual_matched_pythia70m_20260628/summary.md`
- `results/structured_residual_matched_pythia70m_20260628/matched_residual_metrics.csv`

## 注意

保存的 PPL smoke 使用了 fallback 文本，因为远端无法通过 HF mirror 拉取 `wikitext/wikitext-2-raw-v1`。因此 PPL 只作为端到端 smoke，不作为标准 benchmark 数值。offline activation/weight reconstruction 的负结论不依赖 PPL。
