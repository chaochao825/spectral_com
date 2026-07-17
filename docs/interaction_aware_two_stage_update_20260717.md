# 交互感知异构压缩：两阶段方法与当前结果

更新时间：2026-07-17

## 1. 当前研究问题

当前方法不再把目标限定为“提出一种 Q+S+L 组合”，而是研究：

> 在真实序列化字节、输出敏感几何和独立验证集约束下，如何自动选择量化、稀疏、低秩、分组尺度以及 dense fallback，并判断同层组件交互是否真的有价值。

每个被选线性层的候选可写为：

```text
W_hat = Q(W; bitwidth, quantizer, group_size) + S + L
```

候选池同时包含：

- 3/4-bit 等 base quantization 位宽；
- row scale 与 column-group scale；
- symmetric RTN 与 MSE clip；
- sparse support 及固定 support 上的 OBS value refit；
- 不同 rank 的 covariance-whitened low-rank repair；
- 4/8/16-bit low-rank factors；
- pure-S、pure-L、逐层二选一 no-joint 和允许同层 S+L 的 QSL。

## 2. 两阶段分配

### 阶段 A：代理筛选

在完整 multi-layer canonical artifact 上计算真实 natural serialized bytes。动态规划在共享 Q+L cap 下按 Hessian/activation-covariance cost 保留 top-K，而不是用各层 nominal bits 相加代替容器字节。

### 阶段 B：validation NLL 重排

代理 top-K 分别替换模型，在独立 validation split 上计算 NLL。validation winner 才成为最终 endpoint。train 用于 covariance/calibration，validation 只用于分配，test 在选择完成前不访问。

当前输出包括：

- `allocation_validation_rerank.csv`
- `allocation_validation_window_nll.csv`
- `endpoint_window_nll.csv`
- `strategy_endpoints.csv`
- `artifact_manifest.json`
- `artifact_payloads.csv`

## 3. 同层交互的严格门禁

QSL 只有同时满足以下条件，才能支持“同层联合有价值”：

1. no-joint 使用相同候选基础，只禁止任一层同时启用 S 和 L；
2. validation 选定 QSL 后，no-joint 重新在该 QSL 的 exact natural file bytes 上分配；
3. 两个最终 artifact 的 natural bytes 完全相等，尾部 padding 相等不算；
4. final test 上 QSL NLL 严格低于 no-joint。

若精确自然字节匹配不可达，或 QSL 只退化为 pure-S/pure-L，结果仍可描述，但不能作为同层交互证据。

全 MLP 候选较多时，exact-natural 动态规划可能触及硬状态上限。此时核心 QSL 与共享上限下的 cap-best no-joint 结果仍可输出，但 exact-natural 反事实被标记为 `state_limit_exceeded`，联合价值结论强制为不可判定；不得用近似字节匹配或碰巧相等的已输出文件替代该门禁。

## 4. 已完成真实 smoke

运行目录：

```text
/home/wangmeiqi/codex_worktrees/com_compression-two-stage-20260717/
results/two_stage_heterogeneous_pythia70m_natural_match_smoke_20260717
```

设置：Pythia-70M、一个 MLP tensor、train/validation/test 独立 split、`2 x 32` validation/test token windows、proxy top-2、异构量化与低秩位宽候选。

关键结果：

| 项目 | 结果 |
|---|---:|
| QSL natural bytes | 627,712 |
| exact-matched no-joint natural bytes | 627,712 |
| QSL final-test NLL | 4.544342 |
| no-joint final-test NLL | 4.544342 |
| QSL test NLL gain | 0.000000 |
| joint-value claim | false |

QSL 和 no-joint 最终都选择了同一个 4-bit row-scale RTN + FP16 low-rank 配置。该结果证明 exact-natural counterfactual 和 test gate 能正常工作，但不证明同层 S+L 有收益。

pure-S endpoint 选择了 4-bit、group-64、MSE-clip base quantizer，说明异构候选分配确实生效，而非始终退化为默认量化器。

### Qwen2.5-3B sentinel

运行目录：

```text
/home/wangmeiqi/codex_results/
large_model_interaction_aware_v4_ea77d12_20260717
```

设置：Qwen2.5-3B-Instruct、layer-0 `gate_proj` 一个 tensor、`calib-limit/selection-limit/eval-limit=4/2/2`、`sequence_length=128`、3/4-bit、row/group-128、MSE-clip/RTN、4/16-bit low-rank factors，以及 proxy top-2 validation NLL 重排。

关键结果：

| 项目 | 结果 |
|---|---:|
| QSL natural bytes | 13,506,496 |
| exact-matched no-joint natural bytes | 13,506,496 |
| padded physical file bytes | 13,515,712 |
| QSL final-test NLL | 2.550458 |
| no-joint final-test NLL | 2.550458 |
| QSL test NLL gain | 0.000000 |
| joint-value claim | false |

两个 endpoint 都选择了同一个 4-bit、group-128、symmetric RTN base quantizer 和 rank-72 FP16 low-rank repair，没有启用 sparse component。exact-natural counterfactual 搜索成功，train/validation/test 的文本重叠计数均为 0，canonical proxy `selection_source` 与 validation `final_selection_source` 也分别保留。

该结果把两阶段异构分配扩展到了 3B 模型，但仍然只是单 tensor、两个 validation/test window 的 scalability smoke。它证明工程路径可行，不证明同层 S+L 有价值，也不支持模型级质量或压缩结论。

## 5. 当前工程状态

- codec 支持分组量化 scale、不同 base bitwidth/quantizer 和量化低秩 factors；
- endpoint CSV 记录每层实际 `q_bits`、quantizer、group size 和 low-rank factor bits；
- 91 项 codec/Hessian/suite 关键回归通过；目标发布树全量回归为 318 passed、3 skipped，skip 仅对应按发布政策省略的历史 `.hrc` payload；
- 新增 `configs/large_model_interaction_aware_v4_20260717.json`；
- v4 包含 Qwen2.5-3B、Llama-2-7B、Mistral-7B 的单层 sentinel、三深度 6-tensor 和全 MLP feasibility 阶段；
- 三个单层 sentinel 使用相同的缩减但仍异构的提交门禁网格：3/4-bit、row/group、RTN/MSE、4/16-bit factors、proxy top-2 validation rerank 和 exact-natural no-joint；更完整的 factor/rank 网格保留在三深度阶段；
- 旧配置 Qwen sentinel 在 8 小时门限后失败，已原样移入输出根目录的 `trash/`；修复后的 commit `ea77d126` sentinel 在 210 的物理 GPU 0 完成并通过 suite 独立检查；
- 修复后结果的 QSL/no-joint exact-natural 匹配成功，但两者退化为同一个 pure-L endpoint，因此联合价值门禁给出明确 negative，而不是不可判定或正结论。

### 可扩展性优化

候选范围不因性能优化而删除。runner 对以下严格相同的计算做复用：

- 相同 candidate 与相同 Hessian metric 的 endpoint cost；
- 相同 residual、support、`rcond` 的 OBS refit；
- 相同 Q residual 的 pure-L rank frontier。

pure-L frontier 先计算该 residual 在各 base target 以及明确存在的 endpoint global budget multiplier band 下所需的最大 rank；不会把非 endpoint target 与 multiplier 做笛卡尔积。较小 rank 始终从当前最大 superset 分解中截取。对 randomized SVD 而言，这会把“每个 rank 独立随机分解”改为“共享的 nested randomized decomposition”，因此属于显式数值求解策略，不伪装成无语义影响的缓存。candidate-cost cache 使用弱 metric 身份键，OBS cache 先校验并绑定实际 prepared covariance，codec 回收时同步移除弱缓存条目。每层 `factorization_cache_*`、`q_residual_prime_*` 和 `performance_cache` 计数写入 `run_config.json`。

Pythia-70M 同配置端到端回归中，SVD solver 调用从原始 66 降至 44；获得 24 次 factorization、308 次 candidate-cost 和 7 次 OBS cache hit。修复前只按 target priming 时，pure-L/QSL/no-joint 的 test NLL 为 `4.5443420410`；覆盖 `1.25x` global budget band 并统一使用当前最大 superset 后，三者仍严格匹配 `627712` natural bytes，test NLL 同为 `4.5663907451`，联合价值结论仍为 false。两次 Q+L artifact 只有 161 个 FP16 endpoint 元素不同，最大绝对差为 `3.0517578e-05`；因此这是明确披露的求解策略变化，不把该回归描述为逐值不变。

## 6. 结果解释原则

- Hessian cost 用于候选筛选，不等同于 task loss；
- validation NLL 是模型选择证据，final test 只用于一次最终评估；
- 同一 physical padded bytes 不代表同一自然资源；
- 单层 no-joint 必然退化为 pure-S 或 pure-L，真正的异构 no-joint 优势主要在多层分配中出现；
- QSL 若没有选择任何同层 S+L 状态，即使优于其他方法，也不能归因于联合交互；
- selected-weight artifact 不是完整 checkpoint 压缩结果，必须明确 scope。

## 7. 下一步判据

三深度和全 MLP 阶段重点检查：

1. QSL 是否在至少一层同时启用 S 和 L；
2. matched no-joint 是否能在相同 natural bytes 下保持强竞争力；
3. validation 选择是否稳定跨 Qwen/Llama/Mistral；
4. QSL 的 final-test 改善是否超过 paired-window 不确定性；
5. 异构量化和 repair 分配是否呈现可解释的深度/模块规律；
6. 静态字节收益能否转化为峰值显存、运行时流量和真实延迟收益。
