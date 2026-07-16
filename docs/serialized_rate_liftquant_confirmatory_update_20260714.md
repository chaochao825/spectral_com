# 物理同率组合压缩、参数利用率与 LiftQuant 更新

日期：2026-07-14

> **2026-07-16 补充审计：**下文的“相同物理文件字节”是尾部填充后的最终文件相等；受限 QSL 的自然编码少用 15,680 B，因此它只否定逐层保护的保守候选，不代表预算耗尽前沿。新的分配器、同类工作边界和 3B/7B 计划以 [`method_rationality_comparison_and_expansion_20260716.md`](method_rationality_comparison_and_expansion_20260716.md) 为准。

本页是 2026-07-13 Hessian 修补报告的后续审计。它以真实序列化 artifact 字节替代仅统计 value stream 的旧口径，并统一说明参数利用率、正交条件、loss landscape、LiftQuant 官方实现边界和后续确认性协议。若本页与旧报告中的“同率组合方向性占优”冲突，以本页的物理字节结果为准。

## 1. 结论先行

1. **尾部填充到相同最终物理文件字节后，本次保守组合没有胜过单一 Q+L。** `Q+L` 与受限 `Q+S+L+component scale` 最终都是 `3,248,832 B`，但后者的 PPL delta 为 `+7.319517`，比 Q+L 的 `+6.761523` 差 `0.557993`。
2. **组合优势并非不可能，但它对离散开销非常敏感。** 不受 Q+L 上限约束的组合只多 `5,056 B`（`+0.1556%`），PPL delta 降到 `+6.475780`，方向上比 Q+L 好 `0.285743`；16 个固定窗口的区间仍跨 0，因此不能写成已确认优势。
3. **当前最可靠的小参数修补是 OBS，其次是 block scale；在已测试的保守候选中 Q+L 最好。** OBS 不改变 Q+S artifact 字节，16/16 窗口改善；block scale 用额外 `82,944 B` 换来 16/16 窗口改善。Q+L 相对 Q 的单位新增物理字节 NLL 回收仍高于 block scale。
4. **达到的是 S/L 修补之间的近正交，不是 Q/S/L 三者互相正交。** 受限 scale 组合的聚合 `rho_H(S,L)=0.0304`；真正同时激活 S/L 的 3 层最大 `|rho_H(S,L)|=0.0857`。但 `rho_H(Q,S)=-0.4234`、`rho_H(Q,L)=-0.5719`，属于共同抵消 Q 误差。
5. **Hessian proxy 能解释路径形状，不能替代 endpoint。** 六条 13 点径向曲线的 proxy/NLL 相关系数为 `0.9970--0.9995`，但小扰动拟合都有非零负线性项；最终仍必须以 `epsilon=1` 的独立 NLL/PPL 和真实 artifact 字节裁决。
6. **LiftQuant 目前只有官方代码审计和本地兼容补丁后的单层 smoke，不是完整复现。** Block correction 与 optional E2E 分属 B/C 两条线，不能与无需 backward 的 PTQ 混排，也没有可用于排名的 PPL 或任务精度。

## 2. 从 value-stream 同率到物理同率

**作用域限定：**本节的 `3,248,832 B` 与“物理同率”只覆盖 Pythia-70M 中选定的 6 个 MLP linear 权重张量（`selected_linear_weights_only`）、单个 seed；它不是整模型 checkpoint、推理容器或部署包的字节数。后续大规模实验必须在相同张量集合、序列化规范和完整模型精度协议下重新比较，不能把这里的局部结论直接外推到整模型。

研究 codec 固定记录 packed signed Q codes、FP16 scales、CSR sparse values/support、FP16 low-rank factors、逐 tensor descriptor、manifest、header 和 64-byte alignment。所有 endpoint 都被独立解码，并与 runner 中最终 FP16 权重逐元素一致。该格式是可审计研究容器，不是生产推理 backend；它解决的是“比较时到底用了多少真实字节”，不宣称 kernel、吞吐或最终部署格式已经完成。

| endpoint | 文件字节 | natural 字节 | 相对 Q+L | PPL delta | norm. Hessian |
|---|---:|---:|---:|---:|---:|
| Q | 3,167,552 | 3,167,552 | -81,280 | +15.643586 | 0.0100772 |
| Q + block scale | 3,250,496 | 3,250,496 | +1,664 | +9.403732 | 0.0057666 |
| Q+S | 3,260,546 | 3,260,546 | +11,714 | +9.293155 | 0.0068027 |
| Q+S_OBS | 3,260,546 | 3,260,546 | +11,714 | +7.156156 | 0.0058573 |
| **Q+L** | **3,248,832** | **3,248,832** | **0** | **+6.761523** | **0.0046331** |
| 受限 Q+S+L | 3,248,832 | 3,233,152 | 0 | +7.419177 | 0.0051115 |
| 受限 Q+S+L + component scale | 3,248,832 | 3,233,152 | 0 | +7.319517 | 0.0050713 |
| 未约束 Q+S+L + component scale | 3,253,888 | 3,253,888 | +5,056 | +6.475780 | 0.0045737 |

受限 QSL 的 natural payload 比 Q+L 少 `15,680 B`，然后以 tail padding 补到完全相同的文件长度。这是保守的物理等率控制：它避免逐层 descriptor/alignment 的离散误差在全局相加后偷偷超预算。开发工作记录曾显示，仅施加逐层上限会在全局多出 `64 B` 并被程序拒绝；该失败控制日志未纳入当前提交，所以不把它作为仓库内可复核的结果证据。最终结果加入了 alignment guard 和 endpoint NLL 前的全局预校验。

旧 value-stream-only 结果曾显示组合比 Q+L 好约 `0.321 PPL`。真实 descriptor、CSR support 和 alignment 加入后，分配器必须减少 sparse nnz/rank，严格物理同率结果反转为组合差 `0.558 PPL`。这说明 nominal bit、逻辑 value bits 与 artifact bytes 不能混作同一证据等级。

机器可读核验见：

- [`artifact_manifest.json`](../results/pretrained_hessian_repair_pythia70m_serialized_20260714/artifact_manifest.json)：每个 artifact 的 SHA256、natural/file bytes、padding 和 roundtrip 状态；
- [`serialized_rate_summary.json`](../results/pretrained_hessian_repair_pythia70m_serialized_20260714/serialized_rate_summary.json)：endpoint/manifest/磁盘大小/SHA256 联合核验；
- [`paired_method_comparisons.csv`](../results/pretrained_hessian_repair_pythia70m_serialized_20260714/paired_method_comparisons.csv)：五组固定窗口配对差；
- [`summarize_serialized_hessian_result.py`](../scripts/summarize_serialized_hessian_result.py)：stdlib-only 的确定性重算脚本。
- `formal_run.{status,exit_code,time.txt,stdout.log,stderr.log}`：控制进程的完成状态、exit 0、`350.30 s` 和峰值 RSS `2,414,092 KiB`；其 SHA256 也写入 serialized summary。

### 运行时代码 provenance 边界

这次 serialized 数值运行的 `run_config.json` 记录了 base commit `4366a2f` 和 dirty 文件列表，但当时没有保存 dirty diff SHA 或 runner/codec 源文件 SHA。因此 artifact、endpoint CSV、磁盘大小、SHA256 和独立 decode 都可复核，**当时未提交源码的逐字节快照却不能从 run_config 完整重建**。数值运行后加入的改动仅为 fail-closed validation 与呈现加固：拒绝 1-bit 三值假编码、显式空组件、缺层 artifact、非空输出目录复用，并修正 summary 的逻辑/物理 ratio 标签；没有重新选择候选或改写本页数值。后续 runner 已把关键 source SHA256 写入 `source_snapshot`，正式多 seed 运行必须使用这一新版 provenance 门槛。

## 3. 参数利用率：什么值得优先保留

为了回答“少量参数修补是否有效”，不能只数参数个数，应看真实新增字节的边际 held-out 收益，并同时检查二阶机制：

`eta_task = -(NLL_new - NLL_control) / max(1, bytes_new - bytes_control)`。

这个量只用于本次局部比较；当两个 artifact 同字节时，应直接报告零额外字节下的 NLL 差，而不能把分母写成无穷效率。

| 修补 | 物理字节差 | 配对 mean NLL 差 | 固定窗口 | 解释 |
|---|---:|---:|---:|---|
| Q+L 相对 Q | +81,280 | -0.107694 | endpoint 聚合 | 约 `1.33e-6 NLL/B`，严格 endpoint 中最有效的新增表示 |
| block scale 相对 Q | +82,944 | -0.074428 | 16/16 改善 | 约 `8.97e-7 NLL/B`；稳定，但单位字节不及 Q+L |
| OBS 相对同 support 的 Q+S | 0 | -0.026856 | 16/16 改善 | 只重新求 sparse values，不增加 support/value 数；是最强零额外 payload 修补 |
| component scale 相对受限未 scale QSL | 0 | -0.001266 | endpoint 聚合 | 可折叠进已计费 Q/S/L FP16 state，收益小但方向正确 |
| 未约束 scale QSL 相对 Q+L | +5,056 | -0.003665 | 10/16 改善 | 约 `7.25e-7 NLL/B`，但区间跨 0，暂不确认 |
| 严格 scale QSL 相对 Q+L | 0 | **+0.007118** | 4/16 改善 | 同字节反而更差；稀疏 support 的离散开销吞掉组合收益 |

### 为什么 scale 保护有效，稀疏/剪枝怎样获得类似机制

- **Scale** 用少量连续自由度改变一整块误差幅度，适合修正“方向大致对、幅值不对”的量化误差；若 scale 能融合进已有存储，它几乎不付额外 descriptor。
- **稀疏/剪枝** 的对应机制不是盲目增加 survivor，而是固定少量 Hessian/Wanda support 后，用 OBS 解在该 support 上重新求值，使一阶 stationarity 尽量成立。这样 support 字节不变，只让每个已付费位置更有效。
- **低秩** 用两个因子修复跨行/列相关误差，没有逐元素索引，因而在本次形状和 rate 下比 CSR 更划算。
- **组合** 应先为每类分量计算“加入一个 rank、一个 sparse group、一个 scale block”的真实边际收益/字节，再做全局分配。只按参数数目或逻辑 bits 分配会系统性高估 sparse 分量。

## 4. 什么时候能达到正交，什么时候组合会占优

对两个修补扰动 `d_a,d_b`，二阶增量为

`Delta L_2 = 1/2 ||d_a||_H^2 + 1/2 ||d_b||_H^2 + <d_a,d_b>_H`。

近正交要求 `|rho_H(a,b)|=|<d_a,d_b>_H|/(||d_a||_H ||d_b||_H) <= tau`，本项目取 `tau=0.1`。组合在同率下优于单方法还需同时满足：

1. 每个分量单独位于小扰动舒适区，加入后的边际 self cost 足够小；
2. residual 子空间近正交，或相对 base error 有可解释的负交叉项，而不是两个 repair 重复修同一方向；
3. 把 scales、indices、row pointers、factors、descriptors 和 alignment 全部计入后，各分量的 marginal gain/byte 仍高于被替换的单方法自由度；
4. 分配粒度足够细，不被“一个 rank”或“一组 CSR support”的离散门槛卡住；
5. 零字节修补确实已融合，decoder 不再依赖隐藏参数；
6. Hessian/activation proxy 的排序在独立 held-out NLL、多个 seed 和 `epsilon=1` endpoint 上仍成立；
7. 模型、压缩 tensor 范围、训练/backward 预算和实际 artifact 格式完全对齐。

本次只满足了第 2 条中的 S/L 近正交，以及 OBS/scale 的部分第 5 条；严格字节分配下第 3、4 条没有满足。因此“每种方法都留在舒适区，组合达到更高压缩率”在理论上成立，但不能由这一个 strict endpoint 宣称已经实现。

## 5. Loss landscape 与舒适区

本次使用 13 个 `epsilon`：`0, 1/32, 1/16, 3/32, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 1`。只有 `epsilon=1` 是可部署 codec；其余点只是沿 `W + epsilon*(W_hat-W)` 的诊断路径。前四个正 epsilon 用于拟合 `a*epsilon + b*epsilon^2`。

| 策略 | 线性系数 a | 二次系数 b | endpoint fit error | proxy/NLL corr. |
|---|---:|---:|---:|---:|
| Q | -0.03756 | 0.22498 | 0.01081 | 0.99821 |
| Q + block scale | -0.01200 | 0.12900 | 0.00680 | 0.99946 |
| Q+S_OBS | -0.02037 | 0.10575 | 0.01019 | 0.99701 |
| Q+L | -0.01717 | 0.10451 | 0.00320 | 0.99783 |
| 严格 QSL + scale | -0.00937 | 0.09721 | 0.00981 | 0.99925 |
| 未约束 QSL + scale | -0.01574 | 0.10045 | 0.00217 | 0.99838 |

高相关表明 input-covariance Hessian 能较好跟踪这条径向曲线；所有 `a<0` 又表明 held-out slice 上 `g^T d` 不能忽略，所以不能只比较纯二次项。更重要的是，物理 Q+L 和严格 QSL 相对 dense 的 PPL 分别仍高约 `9.48%` 和 `10.26%`，不属于“精度损失很小”的完成状态。

## 6. LiftQuant 与新方法：训练后压缩和微调分开

LiftQuant 固定官方 commit 为 `72b3875c770e4579639931fed89dc95e4067edac`。审计得到：

- **Block correction（B）**：论文/README 描述 STE + block output MSE；官方 README 的 `--nsamples/--epochs` 与 parser 的 `--nsamples1/2`、`--epochs1/2` 不一致。机械映射 4096/4096 时，每个启用阶段会因代码内 `1/32` 扣减实际迭代 3968 条；“2 epochs”究竟是每阶段还是总计仍需上游确认。
- **Compatibility-patched smoke**：未补丁 Qwen2.5 layer-0 在 `Catcher.attention_type` 失败；一行透明 metadata 兼容补丁后，8-window、1+1 epoch、单层运行写出 24,566,073 B `.pth`。它没有 `--eval_ppl`、任务准确率、完整模型或部署字节，不能进入排名。
- **Optional E2E（C）**：固定 commit 在参数解析前即因缺失 `datautils_block.py` 导入失败；即使修复，`extra_args` 也可能静默吞掉未知参数。论文/README 的 4096×4096、1 epoch CE 是声明协议，不是本仓库已执行结果。

完整证据见 [`results/liftquant_official_integration_20260714`](../results/liftquant_official_integration_20260714/)。官方论文与仓库分别为 [arXiv:2606.04050](https://arxiv.org/abs/2606.04050) 和 [Heliulu/LiftQuant](https://github.com/Heliulu/LiftQuant)。

方法矩阵现有 59 行，按 A0/A1/B/C/D 明确分离无需 backward、统计/闭式 PTQ、局部 backward-assisted PTQ、全局任务恢复/QAT 和范围不匹配方法；新增的 10 个 D-lane 条目覆盖多模态/视频量化、cache、sparse/structured attention 和 sparse-low-rank attention。两项重要纠偏是：

- SqueezeLLM 主 Fisher sensitivity 流程需要先保存 loss gradient-square checkpoint，因此属 B，而不是无 backward 的 A1；
- SpinQuant 官方 optimized-rotation 路径冻结 dense base，但用 WikiText2 causal-LM loss 和 `Trainer.train()` 全局优化 R1/R2，因此属 C，而不是局部重构 B。

LiftQuant、SliderQuant、HESTIA、ADMM-Q、HAS-VQ、SEPTQ、AAAC、DAQ、Q-Palette、SharQ 等新方法的协议、真实 payload 和复现状态见 [`method_matrix.csv`](../results/compression_method_comparison_20260713/method_matrix.csv)。矩阵是文献协议框架，不是本仓库实测排行榜。

## 7. 确认性实验状态

已生成 8-seed 预注册数据/切分 manifest，但**尚未执行 8-seed 模型实验**：

- seeds：`17,29,43,59,71,89,101,113`；
- 每 seed calibration：`32 x 256` train tokens；
- 固定 validation：`32 x 256`，固定 test：`64 x 256`；
- exact NFKC/lower/whitespace SHA 去重；token 5-gram set-Jaccard `<0.8`；
- 最终检查 `61,776` 对，0 违规，最大 Jaccard `0.041322314`；
- 同一 13 点 epsilon grid，预先固定前四个正点做局部线性+二次拟合。

[`protocol.json`](../results/confirmatory_hessian_protocol_20260714/protocol.json) 只存 row ID、内容 SHA、token 长度和分配范围，不存原文或 token ID。该产物只证明确认性设计和防泄漏边界已经固定，不能被引用为多 seed 精度结果。

## 8. 当前决策

短期优先级应为：

1. 把 Q+L 作为严格物理同率的主 endpoint；
2. 保留 OBS 作为 sparse/pruning 的默认值修补，并为 block scale 做真实字节的 Pareto sweep；
3. QSL 只在全局 allocator 能证明每个新增 sparse group/rank 的真实 gain/byte 时启用，不再以“组件齐全”为目标；
4. 执行预注册 8-seed protocol，先检验 strict QSL-vs-QL 的符号是否稳定，再扩展到更大模型/更多层；
5. 外部方法只在训练 lane、校准 token、tensor scope 和物理 artifact 都对齐后进入直接比较。

因此，对“同样压缩率且精度损失很小时，组合压缩能否比单个压缩有优势”的回答是：**机制上可以，当前物理等率实验尚未达到；它只显示在放宽 0.1556% 字节后有不确定的方向性改善，而严格等字节时 Q+L 更好。**
