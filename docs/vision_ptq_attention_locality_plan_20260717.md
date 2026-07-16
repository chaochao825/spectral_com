# 视觉模型残差、有效秩、Attention Locality 与压缩的联系

更新时间：2026-07-17

## 1. 核心联系

视觉 Transformer 中应区分四个对象：

```text
residual stream X_l
attention update Delta_attn_l
MLP update Delta_mlp_l
attention/cache operator A_l, K_l, V_l
```

`resid_post` 的有效秩较高，不代表 attention 没有塌缩。恒等残差可以保留旧子空间，而 `Delta_attn` 本身可能低秩、局部化或几乎完全落在已有 residual subspace 中。

因此真正与压缩相关的量不是单个 effective rank，而是：

```text
update rank
+ update novelty outside the current residual subspace
+ spatial/local graph support
+ output-sensitive importance
+ runtime reuse and traffic
```

## 2. Attention 局部性如何影响秩

### 全局 attention

全局连接在单层内快速混合远距离 patch。若 attention operator 有明显主奇异值间隙，非共享 token mode 会快速收缩，表现为 token effective rank 和空间高频能量下降。

### 固定窗口

局部窗口减慢全局混合，但可能先在窗口内部发生 rank collapse。此时全图 effective rank 仍可能较高，因为不同窗口的 centroid 保持差异。

对行随机 attention operator `A_l`，令 `H` 为去除 token 均值的中心化算子。非共享 token mode 的单层收缩可用 `H A_l H` 的次大奇异值近似刻画：

```text
||H A_l H X||_F <= sigma_2(H A_l H) ||H X||_F
```

全局 attention 通常只有少量接近 1 的共享 mode；固定窗口的 operator 近似块对角，每个窗口都可能保留一个接近 1 的局部 mode。因此必须联合报告：

- 全图中心化 effective rank；
- 窗口内中心化 effective rank；
- 窗口 centroid matrix 的 effective rank；
- 跨窗口 principal angle 或 centroid covariance；
- `H A_l H` 的奇异值间隙。

否则“局部 attention 的全图秩更高”可能只表示窗口之间尚未混合，并不表示每个窗口内部保留了更多细节。

### shifted window、稀疏全局边和 register token

它们形成多尺度图：先局部平滑，再跨窗口传播。register/CLS 还可能吸收高能全局 mode，使 patch token 的谱看起来更平滑。必须把特殊 token 与空间 patch 分开报告。

### 与残差更新的关系

设当前 residual top-r 子空间投影为 `P_r`，定义 update novelty：

```text
novelty(Delta_l) =
    ||(I - P_r) Delta_l||_F^2 / ||Delta_l||_F^2
```

低秩但高 novelty 的 update 可能是有用的任务特化；低秩且低 novelty 更接近冗余，可优先低秩化、量化或路由裁剪。

有效秩还容易被少量高范数 token 主导。视觉侧至少同时计算：

- all-token covariance；
- patch-only covariance；
- 去除 CLS/register/outlier patch 后的 robust covariance；
- per-image rank 后再按 image bootstrap 聚合；
- 按 token norm 分桶后的谱与任务敏感度。

若去除少量高范数 token 后 top eigenvalue、effective rank 或 attention map 明显变化，应先解释为 special-token/outlier 机制，不能直接解释为全体 patch 的低维结构。

## 3. 与 PTQ 残差的联系

对视觉线性层或 projection：

```text
R_q = W - Q(W)
```

仅分析 `R_q` 的普通 SVD 不够。应在输入 covariance 或输出 Jacobian metric 下测量：

```text
cost(R_q) = 0.5 * tr(R_q C_x R_q^T)
```

然后比较：

- `R_q` 是否在空间局部 head 中更稀疏；
- `R_q` 的有效秩是否随 attention route 改变；
- sparse repair 与 low-rank repair 是否在 Hessian metric 下互相抵消；
- residual update novelty 较低的层是否允许更低 bitwidth；
- dense/global head 是否需要更高 bitwidth 或 dense fallback。

一个可检验的分配规则是：

```text
low novelty + low operator rank
    -> aggressive low-rank/cache factorization

local support + heavy-tailed residual
    -> sparse/block-local repair

high novelty + high output sensitivity
    -> higher bitwidth or dense fallback
```

## 4. 与 KV/cache 因子化的联系

视觉视频模型、autoregressive image model 和多模态模型都可能产生大 cache。cache 压缩不能只报告静态 checkpoint bytes。

对每层/head 的 K/V，测量：

- token/space effective rank；
- temporal effective rank；
- local-window 内 rank 与跨窗口 centroid rank；
- singular-value energy retained by cache factorization；
- cache reconstruction 对 attention logits/output 的误差；
- 每 token/frame 读写流量；
- peak allocated/reserved VRAM；
- end-to-end latency。

局部 attention 会减少实际读取的 cache 范围，因此同样的低秩 factorization 在全局 route 和局部 route 上可能有完全不同的延迟收益。压缩率必须与 route 的真实访存一起核算。

## 5. 建议的三因子实验

实验因子：

| 因子 | 水平 |
|---|---|
| PTQ | FP16、W8、W4、异构 W3/W4 |
| attention route | global、fixed local、shifted local、local + sparse global |
| cache | dense FP16、per-head low-rank、group-quantized、low-rank + quantized |

至少报告：

| 类别 | 指标 |
|---|---|
| 静态资源 | checkpoint natural bytes、各组件 bytes |
| 动态资源 | cache bytes、每层/每帧读写流量、kernel launch 数 |
| 显存 | peak allocated、peak reserved |
| 质量 | classification top-1、segmentation mIoU 或任务原生指标 |
| 速度 | prefill/encode latency、steady-state latency、吞吐 |
| 表征 | token effective rank、within-window rank、update novelty |
| operator | attention entropy、spectral gap、平均空间距离、局部质量占比 |

所有延迟测量需要 warmup、同步和多次重复；static bytes 不能替代真实 latency。

## 6. 两阶段选择如何迁移到视觉侧

### Stage A：资源与几何代理

按以下代理构建 top-K recipe：

```text
proxy =
    output-sensitive PTQ error
  + residual/cache reconstruction error
  + static-byte penalty
  + runtime-traffic penalty
  + peak-memory feasibility penalty
```

候选单位不只是一层 bitwidth，而是：

```text
(weight quantizer,
 weight group size,
 sparse/low-rank repair,
 attention route,
 cache codec,
 dense fallback)
```

### Stage B：独立 validation

在 validation images/videos 上用任务质量与实测 latency 重排 top-K。final test 完全隔离。若要声称某种 joint recipe 有价值，必须与禁止同层/同 head 联合的 no-joint recipe 在相同资源向量下比较：

```text
(natural checkpoint bytes,
 runtime traffic,
 peak VRAM)
```

只匹配其中一个维度不够。

## 7. 最小可行实现

第一轮建议选择：

- DeiT/ViT：全局 attention 基线；
- Swin：固定/shifted local route；
- DINOv2 或带 register 的 ViT：检查特殊 token 对 top modes 的影响；
- 早/中/晚三层；
- `resid_pre`、per-head `attn_out`、merged `attn_out`、`mlp_out`、`resid_post`；
- 256 张自然图像，加 patch shuffle 控制；
- pretrained 与 random-init；
- image-level bootstrap，而不是把所有 patch 当独立样本。

先完成小模型 factorial smoke，再扩展到更大的视觉/多模态模型。强结论应呈现以下链条：

```text
attention topology
-> operator spectral gap and spatial locality
-> residual update rank/novelty
-> PTQ/cache error geometry
-> selected heterogeneous codec/route
-> quality, traffic, memory and latency
```

单独观察“有效秩与准确率相关”不足以支持方法结论。

## 8. 最新文献与本项目的可检验推论

以下来源用于建立分析假设，不替代本项目自己的 matched-control 实验：

| 来源 | 已有证据 | 对本项目的直接推论 |
|---|---|---|
| [Attention is Not All You Need](https://arxiv.org/abs/2103.03404) | 纯 attention 在无 skip/MLP 时具有 token-uniformity 和 rank-collapse 倾向；skip 与 MLP 会阻止退化 | 必须分别测 `attn_out`、`mlp_out` 和 `resid_post`，不能从 residual stream 的高秩反推 attention update 高秩 |
| [How Do Vision Transformers Work?](https://arxiv.org/abs/2202.06709) | ViT self-attention 可表现为低通滤波，残差和 FFN 改变高频保留 | 将空间频率能量、operator spectral gap 与 effective rank 联合报告，而不是只画 activation covariance |
| [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) | 低信息背景位置可出现高范数 token，并被模型用于内部全局计算 | 特殊 token、背景 outlier patch 与普通 patch 必须分组；top mode 可能是内部 scratch-space，而非普遍语义维度 |
| [ResiDual Transformer Alignment](https://arxiv.org/abs/2411.00246) | TMLR 2025 工作发现视觉 head residual contribution 可低维且具有任务/属性专化 | 低秩不等于冗余；压缩门禁必须加入 update novelty、head ablation 或 output-sensitive loss |
| [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010) | 2025 工作将高范数 outlier 追踪到稀疏 register neurons，并观察到 DINOv2 outlier 在 MLP 后出现、随后形成 attention sink | 同时 hook MLP neuron、attention update 和 residual norm；检查“MLP 先造 outlier，attention 再聚合”的因果顺序 |
| [Interpreting Vision Transformers via Residual Replacement](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50cf815fac839ac68846304ea1613aaa-Abstract-Conference.html) | NeurIPS 2025 用 residual replacement 和 sparse features 分析跨层语义、曲线与空间位置特征 | rank/novelty 统计应按特征类别和空间位置分层，并以 replacement/ablation fidelity 验证压缩代理 |
| [Vision Transformers Need More Than Registers](https://arxiv.org/abs/2602.22394) | 2026 预印本把部分 artifact 归因于 global attention 与粗粒度监督下的背景 shortcut/lazy aggregation | 加入 foreground/background、patch shuffle 和局部路由控制；该机制目前只作为待复现实验假设 |
| [Locality-Attending Vision Transformer](https://openreview.net/forum?id=KvEjv5klWi) | ICLR 2026 工作以局部 Gaussian bias 改善 dense prediction，同时保留全局信息 | 对 classification 与 segmentation 分开评估 locality；相同 rank 变化对全局分类和空间密集任务可能含义相反 |

由这些工作得到的最关键判据是：

```text
low effective rank
    != redundant update

low effective rank
+ low subspace novelty
+ low output sensitivity
+ stable quality under replacement/ablation
    -> credible compression opportunity
```

反之，若某个 head/update 低秩但主成分高度任务专化，或承担 register/global aggregation 功能，则应保留更高 bitwidth、独立低秩 factors，或进入 dense fallback，而不是按秩直接裁剪。
