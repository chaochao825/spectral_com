# Hessian 修复方法合理性、同类工作对比与大模型扩展计划（2026-07-16）

## 1. 结论先行

当前方法最稳健的贡献不是“首次把量化、稀疏和低秩组合起来”，也不是证明三者在同码率下必然优于单一修复。更准确的定位是：

> 在一个声明清楚的研究型权重编解码器下，对冻结模型的 Q/S/L 异构端点进行可执行、可往返验证的物理字节审计；同时用带符号的局部 PSD 几何诊断组件相互作用，并用同自然字节上限的真实 NLL/PPL 端点完成最终检验。

现有已完成结果支持以下窄结论：

1. 固定支持上的 OBS 重估在三个已验证的小模型作用域中都以相同文件字节改善了朴素稀疏值；
2. S/L 在当前作用域内接近 Hessian 正交，但这并不保证同字节 Q+S+L 优于 Q+L；
3. 旧的 strict-QSL 文件虽通过尾部填充与 Q+L 达到相同最终文件大小，但其自然编码比上限少 15,680 字节，因此只检验了一个逐层保守分配，不是预算耗尽的全局最优分配；
4. 新代码已经补上嵌套候选、全文件序列化可行性检查和方法因子消融；210 上已有少量未跟踪的 partial/sentinel 输出，但它们尚未形成受控 aggregate 或提交证据，仍不能把“已实现”或“部分跑通”写成“已验证前沿”。

## 2. 数学对象与合理性

对选中的线性层权重 `W`，先产生量化基座 `Q`，再用稀疏残差 `S` 和低秩残差 `L=UV^T` 修复：

```text
W_hat = Q + S + L,
Delta = W_hat - W.
```

由校准激活估计输入二阶矩 `C = E[x x^T]`。局部二阶代理为

```text
Phi(Delta) = tr(Delta C Delta^T).
```

把 `Delta = Delta_Q + S + L` 展开后，除了三个自项外，还包含带符号交叉项，例如

```text
2 tr(S C L^T),
rho_C(S,L) = tr(S C L^T) /
             sqrt(tr(S C S^T) tr(L C L^T)).
```

这一设计合理的部分是：

- `C` 把纯 Frobenius 权重误差转成输入分布加权误差，符合局部层输出重构的直觉；
- 保留交叉项符号能够区分“近似可加”“互相抵消”和“互相放大”；
- OBS 在固定支持下重新求解稀疏值，不增加支持/数值字段的数量，因而是有清楚零增量字节语义的修复；
- 白化 SVD 在 `C` 几何中拟合低秩项，和最终局部评分使用同一类输入统计；
- 真正的研究判断由独立解码后的端点 NLL/PPL完成，而不是由局部代理直接代替。

但二阶代理不是完整语言模型损失。更完整的局部展开还包括一阶梯度项、被忽略的跨层 Hessian 块、代理误差和有限步长路径曲率。因而：

- `rho_C(S,L)≈0` 只表示这个局部代理中的 S/L 交叉项较小；
- `rho_C<0` 表示抵消方向，不等同于正交；
- 一条径向插值路径上代理与 NLL 高相关，不足以证明代理能够在多个离散候选之间正确排序；
- Q/L 的负交叉项部分来自对 Q 残差进行投影/拟合的构造，不能单独当作新机制证据。

## 3. 旧 strict 分配为什么不够

旧实现先在每一层用对应 Q+L 的层内字节作为上限，并额外减去一段 64 字节保护量，再选择 QSL 状态。完整文件最终可以用尾部填充对齐到 Q+L 的大小，但这没有把未使用的自然字节重新分给其他层。

在已提交的 Pythia-70M 物理编解码结果中：

| 端点 | 自然编码字节 | 最终文件字节 | 解释 |
|---|---:|---:|---|
| Q+L | 3,248,832 | 3,248,832 | 比较上限 |
| 旧 strict QSL | 3,233,152 | 3,248,832 | 尾部填充 15,680 字节 |

因此旧实验能够说明“这个逐层保护、组件缩放候选没有赢”，却不能说明“所有同字节 QSL 分配都不可能赢”。论文和 README 中继续保留旧数值，但不再把它描述成全局预算耗尽的分配结果。

## 4. 新的嵌套全文件分配器

新接口 `--rate-allocation global_exact` 对每层声明以下嵌套状态：

- Q；
- Q+S_OBS；
- Q+L；
- 枚举的 Q+S+L 状态，包括不同 rank 和 support refinement；
- 必要时回退到旧 `local_guard` 候选，保证失败语义明确。

流程是：

1. 对局部最优 rank 额外枚举 `1.25/1.5/2.0` 倍修复预算带；倍率只作用于 `Q+L` 超出 base-Q 的修复 allowance，不会放大量化基座本身，因此允许某层在候选集中跨层借用适度字节；
2. 对每层候选按“自然字节、局部代理代价”做 Pareto 剪枝；
3. 用加性代理组合跨层状态；
4. 对候选完整组合调用真实多层编解码器，检查自然文件大小是否不超过该作业内 Q+L 上限；
5. 只在自然编码优化结束后做可选尾部填充，使最终比较文件字节相等；
6. 在输出中记录自然上限、剩余字节、枚举/剪枝状态数、是否触发 64 字节桶粗化、回退原因和完整序列化可行性。

“exact”只指最终文件可行性由真实序列化器判定。跨层组合前使用的单层自然字节加性剪枝仍是启发式；该算法也没有穷举所有支持、秩、顺序和连续数值，所以不能称为全局最优或完整 Pareto 前沿。正确表述是“枚举修复预算带候选集上的加性剪枝分配，并经完整文件序列化器精确验算”。

组件缩放在未缩放 QSL allocation 选定后才拟合。因此 `Q+S+L_QL_budget_component_scale` 是“固定 allocation 的后选择缩放消融”，不是在所有枚举状态上重新搜索得到的最优 scaled frontier；主分配器的选择报告和缩放后的端点报告必须分开解释。

## 5. 本轮加入的方法因子

| 因子 | 主设置 | 对照设置 | 要回答的问题 |
|---|---|---|---|
| 稀疏支持分数 | Wanda/输入加权 | magnitude | 激活统计是否真正改变支持选择和端点损失 |
| 固定支持值重估 | OBS | naive residual values | 改善来自支持还是同支持内重估 |
| 残差顺序 | S→L | L→S | 顺序依赖是否解释组合不稳定性 |
| 低秩拟合 | whitened SVD | ordinary SVD | `C` 几何是否比 Frobenius 几何更有效 |
| 低秩求解器 | seeded randomized (`q=rank+4`, `niter=2`) | full SVD（小矩阵/显式 CLI 控制） | 大投影上的可运行性与近似误差边界 |
| 协方差结构 | full | diagonal / identity | 非对角输入相关性贡献多大 |
| 协方差阻尼 | 0 | `1e-3 * mean(diag(C))` | 结果是否依赖数值正则化 |
| 白化特征值下限 | `1e-5 * lambda_max` | `1e-4 * lambda_max` | 低秩拟合对病态方向是否敏感 |
| 码率分配 | global_exact | local_guard | 旧的逐层保护是否造成预算浪费或错误排序 |

低秩“拟合度量”和端点“评分度量”现在分别记录。白化特征值下限只正则化低秩求解器；它不再被含混地描述为所有组件共用的同一个 `C`。3B/7B 配置显式使用按 job/layer/rank 隔离 seed 的 `torch.svd_lowrank`，并记录 oversampling、power-iteration、solver 集合和调用计数；这提高可运行性，但也意味着低秩项是近似求解，不能与 full SVD 数值精度混同。协方差 PSD 审计修复了正定矩阵最小特征值被错误报告为零的问题，并把 PSD audit 收敛为每张量一次谱验证；whitened fit 仍保留一次单独、可复用的特征分解。

## 6. 与最接近工作的对比

| 工作 | 与本项目重合处 | 关键差异 | 本项目不能声称什么 |
|---|---|---|---|
| OBS | 二阶信息、固定结构后的最优补偿 | 本项目把固定支持 OBS 作为 Q/S/L 编解码端点中的一个值重估组件 | 不能声称首次二阶稀疏补偿 |
| Choi et al., *Towards the Limit of Network Quantization* | Hessian 加权量化失真、熵约束/码率思想 | 该工作已明确连接 Hessian、ECSQ/Huffman 和压缩约束；本项目强调异构字段的完整研究文件与往返验算 | 不能声称首次 Hessian+rate 联合设计 |
| GPTQ / OBC / SparseGPT / SpQR | 局部二阶 PTQ、逐步补偿、量化/稀疏或异常值混合表示 | 本项目不把这些成熟算法简化为自己的新颖性；主要比较边界是完整字段计费和 Q/S/L 端点诊断 | 不能用代理实现冒充官方复现 |
| SLiM | 量化、2:4 稀疏、低秩联合压缩 | SLiM 已经是直接的 Q/S/L 先例；本项目的差异是冻结端点、声明编解码器和字节审计 | 不能声称首次 Q/S/L |
| Harma et al. | 量化与稀疏的非正交性、顺序和变换级理论 | Harma 研究变换组合的次可加性/顺序；本项目的 `rho_C` 是具体端点在局部 PSD 代理下的带符号交叉项，两者互补 | 不能声称首次讨论 Q/S 相互作用或顺序 |
| OBR | Hessian 下联合量化/稀疏与补偿 | OBR 的补偿可以折回并重新量化，不必作为单独部署张量；本项目显式保留并计费所声明的 S/L 字段 | 不能把“补偿张量”视作天然相同部署状态 |
| Orr et al. | 可变长权重格式、Fisher 指导位分配 | 其核心是权重码字/预期码长；本项目检查一个包含所有异构字段、描述符、填充和对齐的完整文件 | 不能声称首次 Fisher/二阶位分配 |
| ProjQ | 量化与低秩适配器的正交协调 | ProjQ 已明确处理 Q/L 正交；本项目是冻结权重端点的局部交互诊断与物理文件预算 | 不能声称首次 Q/L 正交思想 |

主要来源：

- Harma et al.: <https://openreview.net/forum?id=wJv4AIt4sK>
- OBR: <https://openreview.net/forum?id=VQIvBpL5ag>
- Choi et al.: <https://openreview.net/forum?id=rJ8uNptgl>
- Orr et al.: <https://arxiv.org/abs/2505.12988>
- ProjQ: <https://arxiv.org/abs/2606.00494>
- SLiM: <https://proceedings.mlr.press/v267/mozaffari25a.html>

完整、可再生成的 59 行方法注册表位于 `results/compression_method_comparison_20260713/method_matrix.csv`；其中新增 10 个 D-lane 多模态/视频系统条目。论文仍只抽取其中 27 个最相关的权重压缩条目，避免把训练依赖、部署状态或计费口径不同的方法放到同一直接排名中。

### 6.1 多模态谱诊断与叠加压缩补充

QuantSparse、CacheQuant 等工作表明，量化与 sparse attention/cache 的误差并不自动正交；单独优化后再拼接可能放大偏差。`multimodel_compression` 的只读诊断进一步提示，视觉/视频 attention 更适合分解为 sink/global 低秩、local/cyclic 结构、动态稀疏路由和 dense fallback，而不是统一替换为固定 BCCB 或 top-k 模板。有效秩、sink mass、局部质量、cyclic/BCCB 拟合、route 稳定性和真实 `V` 输出误差可共同决定每层/头/时间步的路径。

完整策略、论文报告数字、官方代码状态、联合目标、系统计费和 HunyuanVideo-13B 验证阶梯见 `docs/multimodal_spectral_stacked_compression_strategy_20260716.md`。该文档明确区分本仓库已提交结果、其他仓库诊断、外部论文报告和尚未执行的叠加方案；任何单项加速倍数都不得直接相乘。

## 7. 210 上声明的扩展实验

配置 `configs/large_model_method_ablation_20260716.json` 一共声明 15 个单 seed、单 rate 作业：

| 模型 | 作业数 | 权重作用域 | 目的 |
|---|---:|---|---|
| Pythia-70M | 3 | 全部 6 层 MLP 上/下投影，共 12 个张量 | global/local-guard 与 full/randomized-SVD 同设置控制 |
| OPT-125M | 1 | 首/中/末层 `fc1/fc2`，共 6 个张量 | 架构控制 |
| Qwen2.5-3B-Instruct | 9 | 第 0/17/35 层 gate/up，共 6 个张量 | 主设置及 8 个方法因子对照 |
| Llama-2-7B | 1 | 第 0/15/31 层 gate/up，共 6 个张量 | 7B 架构/尺度控制 |
| Mistral-7B-v0.1 | 1 | 第 0/15/31 层 gate/up，共 6 个张量 | 7B 架构控制 |

这些作业故意采用深度分层的选中权重，而不是整模型 7B 编码，以控制单次文件和 GPU/磁盘开销。3B/7B 作业进一步排除 `down_proj`：其输入宽度约为 11,008--14,336，当前 full-covariance/whitened-SVD 实现即便消除重复 audit 谱分解仍会成为不成比例的瓶颈。该排除项写入 tensor manifest，不能把结果称为完整 MLP。它们仍然只是 scalability/method smoke：没有多 seed、没有独立多 rate、没有全权重作用域，也不是整个 checkpoint 的压缩率。

完成 210 环境和磁盘审计后，推荐在远端工作树中执行：

```bash
cd /home/wangmeiqi/codex_worktrees/com_compression-hessian-repair-exact-rate-20260713
git fetch origin
git checkout agent/hessian-repair-exact-rate-20260713
git pull --ff-only

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COMPRESSION_QWEN25_3B_MODEL=/home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
export COMPRESSION_LLAMA2_7B_MODEL=/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf
export COMPRESSION_MISTRAL7B_MODEL=/home/wangmeiqi/zjh/mistralai/Mistral-7B-v0.1

/home/wangmeiqi/anaconda3/envs/base-2-bitnet/bin/python \
  scripts/run_large_scale_hessian_suite.py \
  --config configs/large_model_method_ablation_20260716.json \
  --output-root results/large_model_method_ablation_20260716 \
  --include-optional
```

应先用 `--dry-run --include-optional` 审核 15 条命令，再运行三个 Pythia 必选控制；只有小模型通过完整 artifact validation 后才应启动 3B/7B 作业。每个大模型作业完成后立即运行 suite `--check`，并保留 GPU、磁盘、checkpoint 路径、源代码哈希和失败日志。

## 8. 当前 210 可达状态与未提交结果边界

最新只读审查确认 210 已可访问，目标工作树仍是 `/home/wangmeiqi/codex_worktrees/com_compression-hessian-repair-exact-rate-20260713`。但是三个新结果目录均为未跟踪 live state，不能作为当前 Git 分支的可引用证据：

- `confirmatory_hessian_pythia70_frontier_v2_20260715`：24 个 job 中 1 个 `completed_valid`、23 个 planned；完成项为 Pythia-70M full MLP、seed 17、rate 0.258。
- `large_model_method_ablation_20260716`：15 个 job 中 2 个 `completed_valid`、13 个 planned；完成项为 Pythia-70M full-MLP global OBS 和 OPT-125M 三深度 global OBS，runtime manifest 报告 CUDA 不可用/CPU 执行。
- `large_model_method_ablation_v2_20260716`：28 个 job 中 2 个 `completed_valid`、26 个 planned；完成项为 Qwen2.5-3B layer-0 gate-only randomized/full-SVD sentinel，使用 A800 80GB，但每项只有一个 tensor。

这些输出只说明部分控制路径已运行，不能标记为 15-job 完成、3B/7B 泛化、完整 MLP、全模型压缩或新的 accuracy-rate frontier。机器可读设计表的 `result_status` 继续保持 `planned`，正文可描述为 “partial live, uncommitted”，而不能写成 `blocked-by-network` 或 `verified`；只有完成 fail-closed aggregate、核验 manifest/hash/endpoint、通过 review 并提交后，才能升级证据状态。

## 9. 结果成立所需的判据

新实验不能只看“组合是否赢”。至少要同时报告：

1. 完整自然文件字节、上限、剩余字节和尾部填充；
2. 每层选择的是 Q、Q+S_OBS、Q+L 还是哪一个枚举 QSL 状态；
3. 独立解码与 SHA256/字段结构检查；
4. 同一模型、同一 tensor manifest、同一评估窗口上的 paired NLL/PPL；
5. Q/S、Q/L、S/L 自项和交叉项，而不是只报告绝对相关值；
6. 方法因子是否改变自然字节、代理排序和真实端点排序；
7. validation-only 路径诊断与 test endpoint 严格分离；
8. 单 seed smoke、预注册多 seed 确认性证据和文献报告三种证据角色分别标注。

只有当全文件自然字节相同或均不超过同一上限、往返验证通过、paired endpoint NLL 稳定改善，并且在预注册 seed/rate 上复现时，才适合把“组合优于单一修复”从假设升级为经验结论。
