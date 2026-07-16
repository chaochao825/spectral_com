# LLM Spectral Dynamics：当前方法与结果

更新时间：2026-07-16

## 一句话结论

当前项目已经形成一套可复用的 Transformer 谱分析与压缩残差诊断框架，并积累了若干可信负结果；最值得继续验证的是 Q/S/L 压缩分量之间的 interaction geometry，而不是把 `Q + S + L` 分解本身作为新方法。现有结果还不足以支持通用压缩优势或 SOTA 声明。

## 1. 方法体系

### 1.1 Activation spectral dynamics

分析对象是指定层和 hook site 的 token activation，而不是 attention matrix 或权重矩阵。主要位置包括：

- `resid_post`
- `attn_out`
- `mlp_out`

激活按流式方式送入 float64 Welford/Chan 中心化协方差估计器，避免保存完整 token-by-hidden 激活张量。输出包括：

- 归一化 eigenspectrum 和固定 rank explained variance；
- participation ratio、entropy effective rank、stable/effective rank；
- spectral entropy、condition number、anisotropy 和 outlier score；
- power-law slope、拟合范围、置信区间和拟合质量；
- token-lag PCA autocorrelation、DMD、KV-cache spectrum；
- PCA intervention 及 pretrained/random matched controls。

大模型加载支持本地 Hugging Face 路径、offline 模式、mixed precision 和 `device_map=auto`。自动切分时，输入送到 embedding 所在设备，不再整体搬动已 dispatch 的模型。

### 1.2 Weight geometry 与结构化近似

该部分先分析矩阵几何，再做相同名义参数或内存预算下的比较：

- singular-value decay、energy rank、effective/stable rank；
- channel outlier 与 residual concentration；
- low-rank approximation；
- block-circulant approximation；
- Monarch-like structured approximation；
- activation reconstruction error、PPL 和有限 zero-shot 验证。

这些测量回答“某种结构是否存在”和“替换后是否仍能工作”两个不同问题。低 weight error 不能自动推出低 PPL degradation。

### 1.3 Quantization residual stack

当前核心假设为：

```text
R_q = W - Q(W)
R_q ~= S_res + L_res
W_hat = Q(W) + S_res + L_res
```

在相同附加内存预算下比较：

- `Q only`
- `Q + S`
- `Q + L`
- `Q + S + L`
- sequential Q/S/L、SPQ-like 和若干 proxy baselines

稀疏分量主要由 magnitude/Wanda-like 规则构造，低秩分量来自量化残差的低秩近似。仓库中的 DAM 行是公式化 proxy，不是作者官方实现。

### 1.4 Interaction geometry

这一部分是当前最有潜力形成独立贡献的方向。它不只测量单个分量，而是测量多个误差分量组合时的相互作用：

- parameter-space cosine；
- activation-covariance 或 Hessian/Gauss-Newton proxy 加权 cosine；
- empirical additivity error；
- application order gap；
- 二维 loss-landscape slice；
- interaction-aware filter 和 layerwise selector。

目前的定位应是“过滤明显不兼容的组合”，而不是直接用 proxy 代替最终 held-out PPL selector。

### 1.5 适配器、旋转与量化

附加实验覆盖 structured adapter、structured-LoRA、LoRA、MoRA、FourierFT、BCA，以及 Hadamard rotation 和直接/结构化量化误差。该部分主要用于验证接口和诊断假设，尚未形成独立正向结论。

## 2. 当前结果

| 实验 | 关键结果 | 当前判断 |
|---|---|---|
| `mvp_real` | mean alpha `1.826`，participation ratio `42.99`，effective rank `82.92`；attention effective rank `31.66`，FFN/residual 约 `108`；pretrained-random alpha delta 仅 `-0.036` | 工程与探索结果有效，但预训练/随机差异很弱，不能概括为普遍表征规律 |
| `large_435_download_20260604_1026` | 成功采集 Qwen1.5-MoE、Qwen2-57B-A14B 和 Llama-2-70B 的层/模块谱 | 证明本地大模型、多 GPU、offline 采集可行；pretrained-only 且样本小，只能报告 feasibility 和 site/depth trend |
| Qwen2.5-1.5B smoke | dense PPL `7.649`；最佳 `down_proj` 替换 PPL `9.875`；low-rank 最好 | 小规模筛选显示低秩优于测试过的结构化近似 |
| Qwen2.5-1.5B formal | dense PPL `13.85`；最佳压缩替换仍约 `1.061e4`；所有模块类型的最佳 weight approximation 均为 low-rank | 完整而有价值的负结果：当前直接 structured replacement 路线失败 |
| formal phase 4 | 最佳表格行为 structured/natural，PPL `5.763` | 训练与评估复用了 validation split 前缀，属于 in-sample smoke，不能作为泛化证据 |
| formal phase 5 | Hadamard 后 4-bit direct quantization error 最低为 `0.1649` | 实现有效，但 rotation 降 outlier/量化误差是已有方向 |
| `compression_orthogonality_mvp_20260623_v7` | `|rho_H|` 与 `|additivity|` Spearman `0.758`，与 PPL degradation `0.559`；Taylor prediction 与 loss `0.942`；guided PPL `1.0838` 优于 fixed `1.1523` | 最强方法学信号，但来自 toy character LM，尚非 pretrained LLM 证据 |
| `structured_residual_matched_pythia70m_20260628` | structured residual 对 matched low-rank 为 `0/120` 胜 | 清晰负结果；不应继续把 block-circulant/Monarch-like residual 作为主线 |
| Qwen2-7B attention-only | dense PPL `67.1155`；`Q+L -0.537`，`Q+S+L +0.165` | 不支持 residual stacking；该子集低秩单独更好 |
| Qwen2-7B attention+MLP | dense PPL `45.5759`；`Q+S -0.2760`，`Q+S+L -0.2807` | 有弱正向筛选信号，但低秩只额外改善约 `0.00465` PPL，当前信号主要由 sparse residual 驱动 |

## 3. 哪些结论可以说

目前可以较有把握地陈述：

1. 流式 covariance/spectrum pipeline 可以扩展到 57B/70B 本地模型。
2. 测试过的 Qwen2.5 直接 structured replacement 在 functional metric 上失败，即使 weight approximation 看起来合理。
3. matched Pythia-70M 实验不支持 structured residual 优于低秩。
4. interaction/additivity 指标在 toy model 上具有预测信号，值得迁移到更严格的 pretrained-LLM 实验。
5. Qwen2-7B 的小子集显示 MLP 量化残差中可能存在可恢复 sparse structure。

目前不能陈述：

- activation spectrum 或 power law 是本项目首次发现；
- `Q + S + L` 分解本身具有方法新颖性；
- structured adapter 优于 LoRA；
- Qwen2-7B 结果已经证明通用压缩优势；
- 仓库 proxy 已复现或击败 DAM、SpQR、LQER、SRR 等正式方法。

## 4. 与已有工作的关系

以下组成部分均已有明确先例：

- token rank collapse 与 singular-value 分析；
- Monarch、block structured 或 low-rank matrix approximation；
- sparse outlier preservation；
- quantization error 的 low-rank reconstruction；
- sparse + low-rank compression；
- Hadamard/orthogonal rotation quantization；
- Hessian/二阶信息用于压缩。

因此，更合理的潜在贡献是：

> 在匹配预算下，用 Hessian-weighted cross term、empirical additivity error 和 order gap 系统判断 Q/S/L 分量是否可以组合，并验证这些量能否预测 held-out PPL 最优组合。

这仍是待验证命题，不是当前已经成立的贡献。完整文献与远端来源审计见 `docs/remote_merge_and_research_audit_20260716.md`。

## 5. 下一步优先级

1. 优先等 238 的 H200 空闲后，用 `configs/structured_qwen25_7b.yaml` 扩大 Qwen2.5-7B Phase 1--3；calibration 和 eval 必须严格不重叠。
2. 至少增加一个第二模型家族及多个 seed。
3. 用官方实现加入 SpQR、LQER/QERA、SRR 等强基线。
4. 同时报告真实压缩字节、kernel latency、吞吐和置信区间。
5. 测量 interaction metrics 对 held-out PPL 排序的 precision、recall 和 rank correlation。
6. phase-4 类适配实验必须拆分 calibration、training 和 evaluation 数据。

## 6. 入口

- 英文方法与结果：`docs/methods_and_results.md`
- 远端合并、来源和新颖性审计：`docs/remote_merge_and_research_audit_20260716.md`
- 结果目录说明：`results/README.md`
- 35 模型路径：`configs/model_paths_435.yaml`
- 主实验脚本：`scripts/run_pretrained_llm_orthogonality.py`
