# 多模态谱诊断与叠加压缩策略

> 核验日期：2026-07-16。本文只更新 `com_compression`，把当前仓库方法、`multimodel_compression` 的只读诊断和外部联合压缩论文放入同一证据框架。本文提出的多模态叠加方案尚未在本仓库完成 HunyuanVideo-13B 端到端验证。

## 1. 先分清四层证据

| 证据层 | 内容 | 可以支持 | 不能支持 |
|---|---|---|---|
| 本仓库已提交、已验证 | Pythia-70M 有界物理字节实验；三个独立 scalability smoke；完整 artifact 往返、哈希和端点 NLL/PPL | 精确字节计费、局部 PSD/Hessian 交叉项、OBS 固定支持修复、当前严格 QSL 负结果 | 多模态、全模型、生产 kernel、普遍加速 |
| `multimodel_compression` 只读诊断 | ViT、Qwen3-VL、Wan2.2 的 attention 有效秩、sink、局部性、周期/BCCB、动态路由和条件 Hessian 修复 | 形成 attention 分解和门控假设 | 本仓库已复现、目标任务质量、部署加速 |
| 外部论文报告 | QuantSparse、CacheQuant、TeaCache、Sparse VideoGen 等论文及官方仓库 | 确认联合优化和系统加速已有直接先例 | 与本仓库结果直接横向排名；把论文数字当作本仓库测量 |
| 本文提出、尚未执行 | 静态权重 Q/S/L/OBS + W/A PTQ + 分解式 attention + cache + 时间复用的联合控制器 | 下一轮实验设计和计费合同 | 任何质量、压缩率或加速结论 |

## 2. 当前 `com_compression` 方法和结论

当前静态权重端点写成

\[
\widehat W = Q + S + L,\qquad
\Phi_C(\Delta)=\operatorname{tr}(\Delta C\Delta^\top),
\]

其中 `Q` 是量化基底，`S` 是带支持和索引成本的稀疏残差，`L` 是同时计费左右因子的低秩残差，`C` 是由校准激活形成的局部 PSD 代理。仓库同时记录每个分量的自项和带符号交叉项，而不是把负交叉项截断。

当前可守住的结论是：

1. 必须用完整、可解码的自然文件字节比较异构端点。codes、scales、支持/索引、低秩因子、描述符、header、padding 和 alignment 都不能省略。
2. `rho_C(S,L)` 接近零只说明当前端点在当前局部代理下近似可加；它不保证相同字节的 `Q+S+L` 优于 `Q+L`。`Q/S` 和 `Q/L` 的负交叉项表示抵消，不是正交。
3. 当前已提交的严格 QSL 候选自然文件没有耗尽 Q+L 上限，尾部填充后仍总体输给 Q+L。这否定的是保守逐层候选，不是所有可能的全局预算分配。
4. 固定支持 OBS 在已验证小作用域内以零额外字节改善了稀疏值；它说明“已付费参数是否被有效利用”与“增加了多少参数”同样重要。
5. 新 `global_exact` 分配器只对枚举的 rank/support 候选做 Pareto 剪枝，并用真实 serializer 检查最终可行性；“exact”不表示穷举所有支持、秩和跨层组合。

## 3. 外部联合压缩和视频系统证据

下表数字均为论文报告值，实验协议不同，不能组成统一排行榜。

| 方法 | 组合/范围 | 论文报告 | 代码边界 | 对本项目的含义 |
|---|---|---|---|---|
| QuantSparse | HunyuanVideo-13B，W/A 量化 + runtime sparse attention | 20.88 PSNR；相对 Q-VDiT 的 16.85；3.68x 存储缩减；1.88x 端到端加速 | 官方仓库只有图片、README 和许可证，并写明代码稍后发布 | 最直接的量化+attention 稀疏先例；证明简单拼接会放大误差，需要联合蒸馏和二阶重参数化 |
| CacheQuant | 图像 diffusion，cache + quantization | Stable Diffusion 上 5.18x 加速、4x 压缩、CLIP 下降 0.02 | 官方代码可检查 | cache 与量化误差不正交；需要联合 schedule 和解耦纠错，但其范围不是视频 DiT |
| Q-VDiT | 视频 DiT W/A PTQ + 时间蒸馏 | W3A6 scene consistency 23.40 | 官方代码可检查 | 视频 PTQ 必须保留跨帧关系，不能只沿用图像/语言模型校准目标 |
| S2Q-VDiT | 显著校准数据 + attention 引导稀疏 token 蒸馏 | W4A6，3.9x 模型压缩，1.3x 推理加速 | 官方仓库仍是占位发布 | 稀疏 token 是训练/校准信号，不等于 runtime sparse attention |
| Sparse VideoGen | 动态空间头/时间头 + 在线 profiling + sparse kernel | HunyuanVideo 最高 2.33x 端到端加速 | 官方代码可检查 | attention 路径应按 head 和输入动态选择 |
| Sparse-vDiT | diagonal、multi-diagonal、vertical-stripe + 离线硬件搜索 | HunyuanVideo 1.85x 实测加速、27.09 PSNR | 一手来源中未确认官方仓库 | 局部和条带结构可成为静态候选，但必须按层/头选择并区分 FLOPs 与实测时间 |
| TeaCache | timestep-aware feature cache | Open-Sora-Plan 最高 4.41x，VBench 下降 0.07% | 官方代码可检查 | cache gate 应依赖时间步变化，而不是固定均匀跳步 |
| VMonarch | 视频 DiT structured Monarch attention | 17.5x attention FLOPs 缩减，attention 计算超过 5x | 一手来源中未确认官方仓库 | 是结构化 attention 证据，不是 5x 端到端结论 |
| MonarchRT | 周期结构 + 动态稀疏语义 + dense mixing | kernel 1.4--11.8x；Self-Forcing 单 RTX 5090 达 16 FPS | 官方代码可检查 | 单一 top-k 稀疏不能覆盖所有 attention；必须保留结构化与 dense 通道 |
| RoPeSLR | 稀疏语义峰 + 极低秩背景 | HunyuanVideo-13B、100K+ tokens 上 2.26x 端到端；平均 VBench 下降小于 1.3% | 一手来源中未确认官方仓库 | 与“全局低秩背景 + 动态稀疏语义”分解最接近 |

不能把 TeaCache、Sparse VideoGen、PTQ 或 QuantSparse 的加速倍数相乘。组合后的真实瓶颈、kernel 占比、cache 内存、数据搬运和误差交互都会改变，唯一有效的系统结论来自同一硬件、同一 workload 下的一次完整端到端测量。

## 4. `multimodel_compression` 的谱与 attention 诊断

以下数字来自只读审查的 `agent/compression-orthogonality-ablation-20260713` 工作树，属于仓库报告的诊断，不是 `com_compression` 已提交结果。

### 4.1 attention 不是单一“局部矩阵”

- Qwen3-VL visual attention 的平均二维 BCCB 拟合 `R2` 约为 0.081，说明把所有视觉 attention 都替换为固定二维循环结构并不成立。
- Wan2.2 被选中的 self-attention 平均约有 0.61 的 cyclic/BCCB 成分，但该比例随 head、layer 和 timestep 变化。
- 目标内 oracle hybrid 的平均矩阵误差为 0.147，而单独 Grid-BCCB 为 0.876。去掉 sink/global、local-cyclic、sparse-routing 后误差分别升至 1.233、0.202、0.236，说明三个通道承担不同作用。
- 跨目标迁移失败：目标内 oracle hybrid 误差 0.154，源支持迁移 1.569，固定模板 2.007；sink Jaccard 为 0，稀疏 route Jaccard 约 0.009。固定路由不能替代动态校准。

因此，合理的 attention 结构不是“局部 attention 等于 BCCB”，而是

\[
A \approx A_{\text{sink/global-low-rank}}
 A_{\text{local/cyclic}}
 A_{\text{dynamic-sparse}}
 A_{\text{dense-fallback}}.
\]

### 4.2 有效秩、sink 和局部性如何对应路径

ViT 聚合的 192 张 attention map 中，top-2 column mass 为 0.448、argmax unique fraction 为 0.081、半径 1 局部质量为 0.140、row top-4 mass 为 0.691、有效秩比例为 0.073，平均原始有效秩为 4.70。Qwen3-VL visual 的 96 张 map 对应值约为 0.204、0.210、0.141、0.440、0.265 和 39.82。

这些量形成可操作的门控关系：

| 诊断量 | 结构含义 | 优先候选 |
|---|---|---|
| 有效秩比例低，top-column/sink mass 高 | 少量全局汇聚方向主导 | sink 显式通道或全局低秩 |
| local-radius mass 高，cyclic/BCCB `R2` 高 | 空间邻域或周期位置结构稳定 | local window、band/stripe、BCCB/Monarch |
| 有效秩较高，argmax diversity 高 | 多个内容相关语义方向，固定模板不足 | 动态 top-k/router |
| route/sink 跨输入或跨时间 Jaccard 低 | 支持不稳定 | 在线校准或 dense fallback |
| attention 输出对 strongest route 敏感 | 稀疏路由具有功能性 | 保留强路由并验证真实 `V` 输出误差 |
| 二阶残差跨 timestep 稳定 | 缓存/重参数化可能可迁移 | 二阶 cache 或 correction，但必须做 held-out 时间迁移 |

ViT/SCTM 的因果探针覆盖 256 个 CIFAR-10 样本：删除 strongest route 使 loss 增加 0.214、accuracy 下降 0.055，并造成 24.2% 预测翻转；删除 weakest route 的 loss 只增加 0.001。该结果说明强路由不是仅有几何外观，但仍不能外推到视频生成质量。

### 4.3 残差与 Hessian 的联系

attention-map 条件诊断中，structured-pruning / pruning-quant / structured-quant 的 Frobenius cosine 约为 0.095 / 0.088 / 0.797，局部 KL-Hessian cosine 约为 0.298 / 0.135 / 0.659。固定 damped Fisher 下的 full OBS 可把 pruning residual 与 retained-only quant perturbation 调整为数值正交，平均 `|rho_H|` 约 `4.38e-17`。

在约 24.03% 的理想 packed payload 下，prune+quant 在 8/8 张目标拟合 attention map 上优于对照，平均 Hessian/KL 增益约 49.2%/44.0%。这只是目标拟合、attention-map、理想打包的条件诊断；没有任务端点、跨输入迁移或部署 kernel，因此不能提升为联合压缩性能结论。

## 5. 提出的多模态叠加压缩栈

### Stage A：静态权重端点

- 用当前完整 serializer 构造 `Q`、`Q+S_OBS`、`Q+L` 和枚举 `Q+S+L`。
- 对每层/张量按自然文件字节、局部 PSD 代价和 held-out 端点损失分配。
- 分开报告 checkpoint 静态字节和所有非压缩权重例外。

### Stage B：权重/激活量化

- 在视频 DiT 上增加 W/A PTQ；校准数据覆盖 prompt、分辨率、帧数和 timestep。
- 把 activation scales、zero-points、clipping、outlier state 和 kernel workspace 纳入运行时合同。
- 若使用蒸馏或 backward，则单列训练数据、token/frame 数、steps、GPU-hours 和峰值显存。

### Stage C：按层、头、时间步门控的 attention

每个 attention 单元从以下路径选择一个或组合多个：

1. `dense`：不满足结构稳定性或质量门槛时回退。
2. `sink_global_low_rank`：低有效秩、高 sink/global mass。
3. `local_cyclic`：高局部质量、高 cyclic/BCCB/stripe 拟合。
4. `dynamic_sparse`：语义路由强但支持随输入变化。
5. `hybrid`：全局低秩 + 局部结构 + 少量动态稀疏。

选择器不能只看 attention matrix 的 Frobenius 误差；至少同时检查真实 `V` 下的 attention-output NRMSE、局部 KL/任务代理和 held-out 生成质量。

### Stage D：cache

- timestep feature cache 和 KV/cache 类状态分开计费。
- cache gate 使用时间步变化、输出敏感度和二阶残差稳定性。
- cache miss、重算、误差累计和量化交互必须进入同一个 schedule，不能先各自优化再拼接。

### Stage E：时间 token/frame/chunk 复用

- 只在跨时间路由、低秩子空间或二阶残差通过 held-out transfer 后启用。
- 对场景切换、快速运动和长序列尾部设置变化检测与 dense refresh。

### Stage F：联合控制器

联合决策变量包括静态 codec、W/A bit-width、每个 attention 单元的路径、稀疏率/秩/窗口、cache schedule 和 refresh。控制器先生成 Pareto 候选，再由真实 serializer、真实 kernel 和端点质量做最终可行性检查。

## 6. 联合目标和交互项

建议把静态文件和运行时状态分开：

\[
\min_{z,\theta}\;
\Delta\mathcal{Q}_{\mathrm{endpoint}}(z,\theta)
+\lambda_B B_{\mathrm{static}}
+\lambda_R B_{\mathrm{runtime}}
+\lambda_M M_{\mathrm{peak}}
+\lambda_T T_{\mathrm{e2e}},
\]

约束为

\[
\Delta\mathrm{VBench}\le \epsilon_V,\quad
\Delta\mathrm{PSNR}\ge -\epsilon_P,\quad
B_{\mathrm{static}}\le B_{\max},\quad
M_{\mathrm{peak}}\le M_{\max}.
\]

候选阶段的质量代理必须显式保留交互：

\[
\widehat{\Delta\mathcal Q}
=\sum_i z_i d_i
+\sum_{i<j}z_i z_j I_{ij}
+\sum_{i<j<k}z_i z_j z_k I_{ijk},
\]

其中 `i` 可以是 weight-Q、weight-S/OBS、weight-L、activation-Q、sparse-attention、cache 和 temporal reuse。QuantSparse 与 CacheQuant 已表明关键交互不应默认等于零；本仓库的带符号 `rho_C` 可作为局部诊断，但最终选择仍以 endpoint 为准。

## 7. 完整系统计费

每次报告至少同时列出：

1. 静态模型文件：codes、scales、zero-points、稀疏支持、低秩因子、router/结构参数、未压缩例外、header、padding、alignment。
2. 运行时持久状态：feature/KV cache、稀疏 index/layout、动态 router、位置参数、量化状态。
3. 临时状态：kernel workspace、重排 buffer、dequant buffer、Flash/Triton 临时内存。
4. 峰值显存：模型、cache、workspace、激活和框架开销的真实峰值。
5. 时间：attention-only kernel、DiT block、VAE/text encoder、完整生成端到端时间。
6. workload：prompt 集、seed、分辨率、帧数、steps、CFG、dtype、batch、GPU、软件和 kernel commit。
7. 质量：配对 BF16 输出 PSNR、VBench/VBench-Long、适用时 CLIP/VQA，以及时间一致性和人工检查。

## 8. HunyuanVideo-13B 验证阶梯

固定同一 prompts、seeds、分辨率、帧数、steps、CFG、硬件和 warmup/repeat 后，依次运行：

1. BF16 dense baseline。
2. 仅 W/A PTQ。
3. 仅 sparse/structured attention。
4. 仅 cache。
5. `PTQ + sparse attention`，与 QuantSparse 类似但明确实现来源。
6. `PTQ + cache`，检验 CacheQuant 类交互。
7. `sparse attention + cache`。
8. `PTQ + sparse/low-rank/local attention + cache` 完整栈。

每一级都需要独立的质量、静态字节、运行时字节、峰值显存、attention-only 时间和端到端时间。完整栈还必须做 leave-one-component-out、两两 interaction 和 dense-fallback 触发率消融。只有真实视频端点通过门槛时，谱、有效秩或 Hessian 指标才可以解释选择原因，不能替代成功判据。

## 9. 210 当前运行状态的证据边界

2026-07-16 的只读审查显示 210 已可访问，但以下目录仍是目标工作树中的未跟踪运行状态，不是该 Git 分支的可引用证据：

- `results/confirmatory_hessian_pythia70_frontier_v2_20260715/`：24 个 job 中 1 个 `completed_valid`，23 个仍 planned；完成项是 Pythia-70M full MLP、seed 17、rate 0.258。
- `results/large_model_method_ablation_20260716/`：15 个 job 中 2 个 `completed_valid`，13 个仍 planned；完成项是 Pythia-70M full-MLP global OBS 和 OPT-125M 三深度 global OBS，manifest 显示运行时 CUDA 不可用/使用 CPU。
- `results/large_model_method_ablation_v2_20260716/`：28 个 job 中 2 个 `completed_valid`，26 个仍 planned；完成项是 Qwen2.5-3B layer-0 gate-only randomized/full-SVD sentinel，使用 A800 80GB，但每项只有一个 tensor。

这些状态只说明部分控制路径已经运行，不能写成 15-job 完成、3B/7B 泛化、完整 MLP、全模型压缩或新的 accuracy-rate frontier。只有把结果纳入受控 artifact、完成 fail-closed aggregate、通过 review 并提交到分支后，才可升级证据状态。

## 10. 一手来源

- [QuantSparse paper](https://arxiv.org/abs/2509.23681)；[official repository](https://github.com/wlfeng0509/QuantSparse)
- [CacheQuant paper](https://arxiv.org/abs/2503.01323)；[official repository](https://github.com/BienLuky/CacheQuant)
- [Q-VDiT paper](https://arxiv.org/abs/2505.22167)；[official repository](https://github.com/wlfeng0509/Q-VDiT)
- [S2Q-VDiT paper](https://arxiv.org/abs/2508.04016)；[official repository](https://github.com/wlfeng0509/s2q-vdit)
- [Sparse VideoGen paper](https://arxiv.org/abs/2502.01776)；[official repository](https://github.com/svg-project/Sparse-VideoGen)
- [Sparse-vDiT paper](https://arxiv.org/abs/2506.03065)
- [TeaCache paper](https://arxiv.org/abs/2411.19108)；[official repository](https://github.com/ali-vilab/TeaCache)
- [VMonarch paper](https://arxiv.org/abs/2601.22275)
- [MonarchRT paper](https://arxiv.org/abs/2602.12271)；[official repository](https://github.com/Infini-AI-Lab/MonarchRT)
- [RoPeSLR paper](https://arxiv.org/abs/2605.20659)
