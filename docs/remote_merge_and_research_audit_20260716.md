# 远端合并与研究价值审计

审计日期：2026-07-16

## 1. 结论

本项目的权威活动版本统一为 237 服务器：

```text
237:/home/wangmeiqi/llm_spectral_dynamics
```

35 上的 `/data6/user20111239/llm_spectral_dynamics` 是较早的 activation spectral dynamics 分支；237 原目录主要是 structured Qwen2.5 分支，另有一个 CUDA benchmark 开发目录。统一版本已经包含三条代码线、35 的最终大模型结果以及 237 唯一的正式 Qwen2.5 结果。

`35:/data6/user24111736` 不是另一份代码仓库。审计时该目录约 132 GB，主体是 129 GB 的 Llama-2-70B 权重，另有 Self-Rewarding-Language-Models 和小型数据目录。因此模型资产保留在 35，只登记路径，不复制到仅剩约 40 GB 空间的 237。

## 2. 三处内容的关系

| 位置 | 原始角色 | 比对结果 | 合并决策 |
|---|---|---|---|
| `35:/data6/user20111239/llm_spectral_dynamics` | activation covariance、谱指标、power-law、token lag、DMD、KV cache、PCA intervention，以及三个大模型的结果 | 43 个核心文件与统一版本逐字节相同；5 个差异集中在 README、元数据和测试入口。`results/large_435` 的 76 个核心文件与统一归档逐文件哈希一致 | 代码与结果已包含；35 原项目先移入 trash，随后按用户本次特许永久删除该精确暂存路径 |
| `237:/home/wangmeiqi/llm_spectral_dynamics` | structured Qwen2.5 的低秩/块结构/Monarch-like、activation reconstruction、PPL、adapter、rotation 和 quantization | 112 个核心文件相同；10 个文件在统一版本中已继续演化。唯一未归档资产是完整 formal run | formal run 纳入统一版本，旧目录移入 trash 后用统一版本替换 |
| `237:/home/wangmeiqi/llm_spectral_dynamics_cuda_dev` | structured approximation CUDA benchmark | 56 个源码/脚本文件与统一版本相同 | 已包含，旧目录移入可恢复 trash |
| 本地整理版本 | 上述三条线加后续 Pythia/Qwen2-7B residual-stack、orthogonality、OASR 和发布检查 | 当前最完整版本 | 同步为 237 活动版本 |

Unix 所有权检查显示 35 的 `user20111239` 项目目录由 `wangmeiqi` 拥有，但路径命名空间不适合作为长期活动位置；迁移仍按“非目标归属路径”处理。

## 3. 合并和回滚记录

本次任务使用的 trash 标识为：

```text
20260716-213625-llm-spectral-merge
```

活动目标：

```text
237:/home/wangmeiqi/llm_spectral_dynamics
```

237 上保留的旧目录：

```text
237:/home/wangmeiqi/trash/20260716-213625-llm-spectral-merge/home/wangmeiqi/llm_spectral_dynamics
237:/home/wangmeiqi/trash/20260716-213625-llm-spectral-merge/home/wangmeiqi/llm_spectral_dynamics_cuda_dev
```

这两个 237 目录没有永久删除。需要回滚时，应先把当前活动目录另行移入新的 trash，再将对应保留目录移回原路径。

用户明确特许永久删除的 35 暂存为：

```text
35:/data6/user20111239/trash/20260716-213625-llm-spectral-merge/data6/user20111239/llm_spectral_dynamics
```

该路径删除前有 364 个文件、17,866,326 bytes；现已永久删除并确认不存在，不能再从 35 回滚。其有效内容已经包含在 237 活动版本和本地合并副本中。

237 formal run 的本地纳入验证：

- 目录：`results/structured_qwen25_1p5b_formal_20260610_194113`
- 文件数：47
- 文件总字节数：16,562,653
- `.state/exit_code`：0
- 下载副本与纳入副本逐文件 SHA-256 一致

238 follow-up 状态：

- 238 优先用于后续 Qwen2.5-7B Phase 1--3 扩展实验。
- `Qwen/Qwen2.5-7B` 已在 238 的 Hugging Face cache 中准备完成，仓库中对应配置为 `configs/structured_qwen25_7b.yaml`。
- 最近一次检查时 238 的 H200 正被其他任务占用；因此本次归档只同步 7B 配置和 CUDA benchmark，不把尚未完成的 238 大实验写成结果。

## 4. 结果价值分级

| 结果 | 观察 | 价值判断 |
|---|---|---|
| `mvp_real` | 平均 alpha 1.826、participation ratio 42.99、effective rank 82.92；pretrained-random alpha 差仅 -0.036 | 有工程和探索价值，但样本小，预训练/随机差异弱，不足以形成新科学结论 |
| `large_435_download_20260604_1026` | Qwen1.5-MoE、Qwen2-57B-A14B、Llama-2-70B 的本地权重、多 GPU、离线采集均成功 | 证明大模型谱采集流程可行，并提供层/模块趋势；因只有 pretrained、样本很小且使用 sample-space reservoir，不是因果或普适结论 |
| Qwen2.5-1.5B formal | dense PPL 13.85；最佳替换 PPL 约 10,610，所有替换阶段均灾难性退化；所有模块的最佳 weight approximation 都是 low-rank | 很有价值的负结果：当前直接结构化替换路线不成立，应停止把它包装成正向压缩结果 |
| formal phase 4 adapter | 最佳表格行为 structured/natural，PPL 5.763 | 不能作为泛化证据。自然文本训练和 PPL 评估都从同一个 validation split 的前缀取样，存在重叠；结果应视为 in-sample smoke |
| formal phase 5 rotation | Hadamard 使 4-bit 直接量化误差降至 0.1649 | 流程有效，但旋转降低 outlier/量化误差是已有方向，不构成新意 |
| `compression_orthogonality_mvp_20260623_v7` | Hessian overlap 与 additivity 绝对值 Spearman 0.758、与 PPL degradation 0.559；Taylor 预测与 loss 0.942；guided PPL 1.0838 优于 fixed 1.1523 | 当前最有方法学价值的信号，但模型是 toy character LM，只能支持扩展实验 |
| `structured_residual_matched_pythia70m_20260628` | structured residual 对 matched low-rank 为 0/120 胜 | 清晰负结果；不应继续把 block-circulant/Monarch-like residual 作为主线 |
| Qwen2-7B residual stack | attention+MLP 子集内，Q+S 为 -0.2760 PPL，Q+S+L 为 -0.2807；后者只比前者多改善 0.00465 | 有弱筛选信号，但实质由 sparse residual 驱动；low-rank 增量尚无可信证据。16 个小评估文本、一个模型家族和少量层不足以排除噪声 |

总体判断：项目有价值，但目前更接近“高质量诊断框架 + 若干可信负结果 + 一个待验证的组合信号”，还不是可直接投稿的完整新方法。

## 5. 已有方法与潜在新意

以下组成部分已有明确先例：

- Transformer token rank collapse 和奇异值分布分析已有理论与干预研究：[Signal Propagation in Transformers](https://arxiv.org/abs/2206.03126)、[Singular Value Transformation](https://arxiv.org/abs/2208.11790)。
- Monarch-like 结构矩阵及其稠密矩阵近似已有解析构造：[Monarch](https://arxiv.org/abs/2204.00595)。
- 量化后保留 sparse outlier 已有 [SpQR](https://arxiv.org/abs/2306.03078)；量化加 low-rank correction 已有 [LQ-LoRA](https://arxiv.org/abs/2311.12023)、[LQER](https://arxiv.org/abs/2402.02446) 和 2026 年的 [Structured Residual Reconstruction](https://arxiv.org/abs/2602.02001)。
- sparse + low-rank 的 LLM 压缩组合本身也已出现，例如 [Hierarchical Sparse Plus Low Rank Compression](https://arxiv.org/abs/2601.07839)。
- Hadamard/orthogonal rotation 降低量化 outlier 已有 [QuaRot](https://arxiv.org/abs/2404.00456)。
- Hessian/二阶信息用于压缩是长期方向，例如 [WoodFisher](https://arxiv.org/abs/2004.14340)。

因此，`W ~= Q(W) + S + L`、activation spectrum、Monarch-like approximation 或 Hadamard rotation 单独都不能作为本项目的新颖性主张。

相对更有辨识度、值得继续验证的是“interaction geometry”：

1. 在相同内存预算下构造 Q/S/L 分量；
2. 测量 Hessian-weighted cross term/cosine；
3. 同时测量 empirical additivity error 和 application order gap；
4. 把这些量作为组合可行性的 filter，而不是直接把单个 reconstruction proxy 当最终 selector。

当前检索没有证明这一整套诊断组合已经被同样定义和系统验证，但也不能据此宣称首创。要形成可投稿贡献，至少需要：

- 在 Qwen2-7B 的更多层、更大且严格不重叠的 calibration/eval 集、多 seed 上复现；
- 加入第二个 7B 模型家族；
- 使用官方或作者代码复现 SpQR、LQER/QERA/SRR 等强基线，而不是自制 proxy；
- 报告置信区间、真实压缩字节、kernel latency 和吞吐；
- 证明 interaction metrics 能在 held-out PPL 上稳定预测 Q+S、Q+L、Q+S+L 的选择。

## 6. 35 模型资产

机器可读清单见 `configs/model_paths_435.yaml`。三个可用模型保持只读：

```text
/data6/user20111239/Qwen1.5-MoE-A2.7B
/data6/user20111239/Qwen2-57B-A14B-Instruct
/data6/user24111736/meta-llama/Llama-2-70b-chat-hf
```

`/data6/user24111736/meta-llama/Llama-2-7b-chat-hf` 没有 config 或权重分片，不能使用。
