# Hessian 修补与 Exact-Rate 组合压缩：方案、理论、已有结果重分析及实验协议

日期：2026-07-13

状态：已有结果审计、Pythia-70M value-stream probe、物理序列化同率复核、loss landscape 和外部方法协议矩阵均已完成；所有实测结论均链接到生成产物，未运行的外部实现继续明确标为 pending。

> **2026-07-14 更新：** 本报告原有的“相同 value-stream bits 下组合方向性占优”已被真实 artifact 字节复核取代。严格同为 `3,248,832 B` 时，受限 QSL 比 Q+L 差 `0.558 PPL`；只在多用 `5,056 B` 时出现仍不确定的方向性改善。统一结论与完整证据见 [物理同率后续审计](serialized_rate_liftquant_confirmatory_update_20260714.md)。

## 1. 一页结论

本轮工作的核心问题不是“能否把量化、稀疏、低秩机械叠加”，而是：在**真实存储率相同**时，能否利用不同压缩分量在局部二阶几何中的互补性，使组合方法处于各自的小扰动舒适区，并以少量可部署参数修补主要误差。

当前能够成立的结论如下。

1. **正交、负相关修补和组合占优是三个不同命题。** 对两个扰动 `a,b`，`|rho_H(a,b)|≈0` 只表示二阶代价近似可加；`rho_H(a,b)<0` 表示误差抵消；二者都不能单独推出同率组合优于单一压缩。
2. **旧结果中的 Q/S 关系主要是负相关修补，不是正交。** Qwen2-7B attention-only 与 attention+MLP 的 Q/S Hessian cosine 分别约为 `-0.657`、`-0.587`。这可以是有用的恢复信号，但不能称为“达到正交”。详见 [交叉项审计](../results/exact_rate_hessian_repair_20260713/hessian_interactions.csv)。
3. **原来的 nominal `0.258` 不是统一的真实 `0.258`。** 稀疏值、稀疏位置、行指针、量化 scale、低秩因子均需计入。Qwen 两组旧结果中，`Q+S`/`Q+S+L` 在可实现 CSR 编码下达到约 `0.2645–0.2668`，不能与 `0.258` 的候选直接宣称同率。
4. **旧结果不支持“Q+S+L 普遍最优”。** 在四个已提交 smoke run 中，`Q+S+L` 仅赢得核心 `Q/Q+L/Q+S/Q+S+L` 比较的 `2/4`；Pythia-70M 明显失败，Pythia-160M 和 Qwen attention+MLP 出现正信号，Qwen attention-only 则是 `Q+L` 最优。
5. **真实 artifact 同率下，组合优势没有成立。** Q+L 与受限 Q+S+L+folded scale 都是 `3,248,832 B`；后者的 PPL delta 为 `+7.3195`，比 Q+L 的 `+6.7615` 差 `0.5580`。未约束组合多用 `5,056 B` 后方向上好 `0.2857 PPL`，但窗口区间仍跨 0。
6. **本轮达到的是 S/L 修补分量之间的近正交，而不是 Q/S/L 全正交。** 物理受限组合聚合 `rho_H(S,L)=0.0304`、3 个同时激活层的最大绝对值 `0.0857`；Q/S 与 Q/L 分别约为 `-0.423`、`-0.572`，属于共同抵消 Q 误差。
7. **OBS 是当前最稳定的零额外 payload 修补。** 同 artifact 字节的 Q+S_OBS 比 Q+S 改善 `2.1370 PPL`、16/16 固定窗口获益；block scale 也为 16/16 获益，但增加 `82,944 B`。严格 endpoint 中 Q+L 的新增字节利用率仍最好。
8. **判据必须落在物理 exact-rate endpoint。** 局部 Hessian、交叉项和 `epsilon<1` 的 loss landscape 只负责解释和筛选；只有 `epsilon=1`、真实 artifact 总字节合格、独立 NLL/PPL 仍占优，才能构成同率机制证据。

已有结果的统一审计入口是 [audit_summary.md](../results/exact_rate_hessian_repair_20260713/audit_summary.md)，可视化见 [existing_result_audit.png](../results/exact_rate_hessian_repair_20260713/figures/existing_result_audit.png)。

## 2. 方案设计之初的预期

### 2.1 基本分解

对原始权重 `W`，考虑如下部署表示：

```text
W_hat = Q + S + L
```

其中：

- `Q`：低比特量化主干，负责绝大多数参数；
- `S`：少量高价值稀疏残差，修补局部异常值或量化后尖锐误差；
- `L`：低秩残差，修补跨行、跨列的相关误差；
- `scale/OBS/basis repair`：不改变或仅少量增加编码自由度的二阶修补。

设计初衷是让每个分量只承担它擅长的误差形态，而不是要求任一分量独自进入高损失区：

- `Q` 留在 4-bit 附近的稳定区；
- `S` 只保留少量收益最高的位置；
- `L` 只覆盖最显著的相关子空间；
- scale 或 support 内重估用少量参数回收剩余的高敏感误差。

### 2.2 设计时应预先声明的假设

| 假设 | 设计预期 | 若观察不到，应如何解释 |
|---|---|---|
| H1：量化残差包含可由少量 scale 回收的幅度误差 | block/global scale 在不显著增加 payload 时降低 Hessian cost | 误差主要是方向/码字错误，而非幅度失配 |
| H2：稀疏位置集中承载高敏感误差 | `Q+S` 的每新增 bit 收益高于继续增加均匀量化精度 | support 太分散，或索引开销抵消收益 |
| H3：固定 support 后重新估值有效 | `Q+S_OBS` 在相同 support、相同 payload 下支配 naive `Q+S` | 输入协方差 proxy 不准确，或 FP16 存储舍入抹去收益 |
| H4：剩余误差具有低秩相关性 | `Q+L` 在相同新增 bit 下优于稀疏修补，或二者在层类型上互补 | 残差谱不集中，低秩因子成本过高 |
| H5：S 与 L 处理不同子空间 | `S/L` 的 `|rho_H|` 较小，且 `Q+S+L` 的单位 bit 收益优于单分量延伸 | 两个分量重复修同一方向，组合只增加 payload |
| H6：组合把各组件保持在舒适区 | 相同总 rate 下，组合 endpoint 的 held-out NLL/PPL 优于任一单方法 | rate 分配不当、交叉项放大、或 proxy 与任务 loss 错位 |
| H7：二阶局部模型能解释小扰动 | 小 `epsilon` 区间的真实 loss 与线性/二次拟合一致 | 非局部效应、梯度项或跨层传播占主导 |

这些假设是待证伪的实验命题，不是由分解形式自动保证的结论。

## 3. 理论框架：正交、修补与组合占优

### 3.1 本仓库使用的 Hessian proxy

对线性层输入样本协方差 `C = E[x x^T]` 和权重扰动 `Delta`，代码采用：

```text
<Delta_a, Delta_b>_H = tr(Delta_a C Delta_b^T)
D_H(Delta) = 1/2 <Delta, Delta>_H
```

对应激活重构 MSE 的局部度量 `C ⊗ I_out`。实现位于 [hessian_repair.py](../src/llm_spectral_dynamics/structured/hessian_repair.py)。

必须明确其边界：

- 它不是带输出通道耦合的完整任务 Hessian；
- 它不自动包含后续非线性、残差连接和跨层误差传播；
- 它依赖校准分布，校准集上的几何关系不保证迁移到 held-out 文本；
- 若 `C` 奇异，正交是在半范数/商空间意义下成立；实现通过 PSD 检查、伪逆和可选 damping 处理。

因此，本文称其为“输入协方差 Hessian proxy”或“activation-Hessian proxy”，不称为精确任务 Hessian。

### 3.2 二阶交叉项

设最终扰动由多个压缩分量组成：

```text
Delta = Delta_Q + Delta_S + Delta_L
```

则：

```text
D_H(Delta)
= 1/2 ||Delta_Q||_H^2
+ 1/2 ||Delta_S||_H^2
+ 1/2 ||Delta_L||_H^2
+ <Delta_Q, Delta_S>_H
+ <Delta_Q, Delta_L>_H
+ <Delta_S, Delta_L>_H
```

定义 Hessian cosine：

```text
rho_H(a,b) = <a,b>_H / (||a||_H ||b||_H)
```

三种区域的含义是：

| 区域 | 几何含义 | 压缩解释 |
|---|---|---|
| `|rho_H|≈0` | 二阶正交，交叉项接近零 | 代价近似可加；不代表总代价小 |
| `rho_H<0` | 反向分量，交叉项降低总代价 | 修补/误差抵消；通常是希望看到的信号 |
| `rho_H>0` | 同向分量，交叉项增加总代价 | 重复或放大误差；组合可能比单方法更差 |

旧实现中若使用 `max(0,rho)` 作为 conditional overlap，会把强负相关截成零，从而把“有益抵消”误判成“无交叉项”。新的分析必须同时保存 `rho_H`、`|rho_H|` 和原始 cross term。

### 3.3 达到正交效果需要什么条件

对目标扰动 `e` 和可调基 `B_1,...,B_k`，若修补后扰动为：

```text
e(alpha) = e + sum_g alpha_g B_g
```

希望它与约束方向 `U_1,...,U_m` 正交，需要解：

```text
A alpha = -c
A[j,g] = <U_j, B_g>_H
c[j]   = <U_j, e>_H
```

因此，可实现的必要条件至少包括：

1. **同一稳定度量。** `C` 必须 PSD 或经合理 damping；所有分量用同一校准分布和同一度量计算。
2. **自由度充分。** `c` 必须位于 `A` 的列空间；一般而言，一个全局 scale 不能同时消除多个独立交叉项。
3. **可部署约束可行。** 解出的 scale/系数需满足范围、符号、FP16/整数舍入、support 和 rank 约束。
4. **编码后仍成立。** 连续解经过存储舍入后需重新测交叉项；否则只证明了浮点优化问题可行。
5. **顺序不破坏结果。** 后续重新量化、重新选 support 或低秩截断会重新引入交叉项，必须在最终解码权重上测量。
6. **held-out 几何能迁移。** 校准集上的正交还需在独立文本、不同 seed 和足够 token 上复核。

[hessian_constrained_basis_repair](../src/llm_spectral_dynamics/structured/hessian_repair.py) 会显式返回约束矩阵秩、相对残差和可行性，避免把欠定/不可行问题静默称为正交。

### 3.4 正交不等于组合占优

在真实总预算 `R` 下，组合占优需要满足：

```text
D_combo(R) < min_j D_single_j(R)
```

其中每个 `D` 必须来自**同一真实 payload**，而非 nominal value-only 预算。即使所有分量两两正交，组合代价也只是各自二次代价之和，并不保证比单方法的最优 `D_single(R)` 更小。

同率组合可能占优的条件包括：

1. **误差方向互补。** Q、S、L 分别覆盖量化网格误差、少量尖峰和相关子空间，而不是重复修同一方向。
2. **组件处在舒适区。** 每种方法只承担其边际收益仍高的区间，避免把单一方法推入急剧恶化区。
3. **边际失真/bit 合理。** 连续近似下，最优 rate 分配满足 `-∂D_i/∂R_i = lambda`；若某组件每 bit 收益始终更低，则不应分配预算。
4. **交叉项非正或足够小。** 正交只保证不额外变坏；负交叉项才可能额外修补。
5. **元数据开销足够低。** support、scale、行指针、rank、padding 和 decoder metadata 都要计入。
6. **proxy 能正确排序 endpoint。** 局部 activation-Hessian 指标必须与 held-out NLL/PPL 至少在候选集内保持稳定排序。
7. **最终编码可实现。** 不能用只存在于连续插值、未量化系数或理想熵编码中的模型证明部署优势。

所以“每种方法在小扰动时都有舒适区，组合可达到更高压缩率”是合理假设，但不是普遍定理。需要用 rate–distortion 曲线和 exact-rate endpoint 对照验证。

## 4. 小参数修补机制与参数利用效率

### 4.1 与 scale 保护量化的类比

AWQ 类方法的核心直觉是：少量显著通道不宜直接增加混合精度开销，可通过等价缩放改善其量化条件。本项目把这一思路扩展为“在固定 codec 族内，用少量可折叠 scale 沿 Hessian 高收益方向修补”。

对当前解码权重 `W_hat`、目标 `W` 和修补基 `B`，二阶最优系数满足：

```text
alpha* = -(B^T H B + lambda I)^dagger B^T H (W_hat - W)
```

部署时有三种口径：

- **全局/组件 scale 折叠：** 系数可吸收到已有量化 scale、稀疏值或低秩因子中时，不新增独立 payload，但必须按实际存储精度重新编码并复测；
- **row-block scale：** 用每输出行、每输入列块一个 FP16 scale 替换原来的每行 scale，新增 scale 数可精确计算；
- **显式 repair 参数：** 不能折叠时必须把参数位数计入 payload，不能作为“免费修补”。

### 4.2 稀疏/剪枝中的类似机制：固定 support 后 OBS 重估

稀疏方法的可类比机制不是简单把保留值整体放大，而是：先固定 support，再调整保留值，使剪枝误差与所有可调保留坐标在 Hessian 度量下正交。

对一行权重，把坐标分为保留集合 `R` 与删除集合 `P`。删除扰动 `d_P` 固定后，最优保留扰动为：

```text
C_RR d_R = -C_RP d_P
```

它满足保留坐标的一阶 stationarity，并把 naive 剪枝代价降到相应 Schur complement 代价。这正是 [obs_retained_support_correction](../src/llm_spectral_dynamics/structured/hessian_repair.py) 的作用。

重要性质：

- support 不变；
- 调整后的值折叠进原有 sparse value payload；
- 与同 support 的 naive sparse 表示相比不新增位置开销；
- 连续解仍需转换为实际 FP16 sparse value 后复测；
- 这是输入协方差 proxy 下的 OBS 式修补，不等价于完整任务 Hessian 的经典 OBS 全模型重构。

### 4.3 低秩小修补

低秩项用 `r(m+n)` 个因子值覆盖跨行/跨列相关误差，适合残差谱集中时。与无结构稀疏相比，它不需要逐非零位置索引；与 block scale 相比，它能改变方向而不仅是幅度。

低秩是否参数有效，取决于：

- whitened/activation-aware 残差的谱衰减；
- `16 r(m+n)` 实际因子位数；
- 与 Q/S 的交叉项；
- 因子本身是否还需 scale、量化或 metadata；
- endpoint PPL，而不仅是 Frobenius/SVD 重构误差。

### 4.4 参数利用效率的统一口径

对相同基线 `A` 和候选 `B`，推荐同时报告：

```text
eta_H   = (D_H(A) - D_H(B)) / (payload_bits(B) - payload_bits(A))
eta_NLL = (NLL(A) - NLL(B)) / (payload_bits(B) - payload_bits(A))
```

以及更易读的“每增加 1% dense-FP16 payload 的 PPL/NLL 改善”。

注意以下例外：

- 对 payload 完全相同的 OBS/refolded-scale，分母为零，不应报告无穷效率；应报告**同 payload 支配关系**、Hessian 恢复率和每拟合自由度恢复量。
- 拟合自由度不等于存储参数。可折叠系数可有优化自由度但没有独立 payload；必须分列记录。
- 若候选实际 payload 超预算，再高的 nominal 参数效率也不能用于同率结论。
- 稀疏方法还应报告 index/value 的 bit 分解和可实现 kernel 的吞吐；参数少不自动等于延迟低。

## 5. Codec value-stream exact payload：同率比较的基础

### 5.1 统一公式

对 `m × n` 权重、FP16 参考模型：

```text
B_ref = 16 m n
B_Q   = b_q N_code + b_scale N_scale
B_S   = b_value k + B_support(mask) + b_s_scale N_s_scale
B_L   = b_factor r(m+n) + b_l_scale N_l_scale
B_rep = 0                          # 仅当可证明已折叠
        b_rep N_rep                # 否则
B_all = align(B_Q) + align(B_S) + align(B_L) + align(B_rep) + metadata
rate  = B_all / B_ref
```

本文的 `rate/payload_ratio` 是“压缩后位数 / FP16 位数”，越小越省；对应压缩倍数为 `1/rate`。例如 `rate=0.258` 约等于相对 FP16 的 `3.876×` 压缩。不要把这个数与“压缩倍数”混用。

[exact_payload_accounting](../src/llm_spectral_dynamics/structured/hessian_repair.py) 分项记录 codes、scales、sparse values、support、low-rank factors、repair、metadata 和 padding。

### 5.2 support 编码层级

报告中区分三类口径：

| 口径 | 含义 | 能否作为真实部署 payload |
|---|---|---|
| value-only / nominal | 只数非零值或因子值 | 否；仅用于复核旧实验定义 |
| entropy lower bound | `ceil(log2 binom(mn,k))` | 信息论下界，不一定可实现 |
| realizable support | bitmap、COO、CSR、fixed-row 中的具体编码；headline 使用固定宽度 CSR | 是，但仍需与实际文件/kernel 对齐 |

`csr_fixed` 包括每个非零的列索引和 `(m+1)` 个行指针；索引宽度按列数选择 8/16/32 bit，行指针默认 32 bit。任何“同压缩率”声明都应指明采用的 codec。

### 5.3 旧 Qwen 结果的 payload 重算

以下数字来自 [payload_audit.csv](../results/exact_rate_hessian_repair_20260713/payload_audit.csv)。PPL delta 相对各自 dense smoke baseline，负值只表示该极小样本上的数值更低，不表示统计显著提升。

| 设置 | 方法 | nominal | 熵支持下界 | 可实现 CSR | PPL delta |
|---|---:|---:|---:|---:|---:|
| Qwen2-7B attention | Q | 0.250000 | 0.250279 | 0.250279 | -0.099638 |
| Qwen2-7B attention | Q+L | 0.257812 | 0.258092 | 0.258092 | -0.536973 |
| Qwen2-7B attention | Q+S | 0.258000 | 0.262480 | 0.266837 | +0.443284 |
| Qwen2-7B attention | Q+S+L | 0.257674 | 0.261260 | 0.264511 | +0.165230 |
| Qwen2-7B attn+MLP | Q | 0.250000 | 0.250176 | 0.250176 | +0.384142 |
| Qwen2-7B attn+MLP | Q+L | 0.257950 | 0.258126 | 0.258126 | +0.028776 |
| Qwen2-7B attn+MLP | Q+S | 0.258000 | 0.262377 | 0.266527 | -0.276042 |
| Qwen2-7B attn+MLP | Q+S+L | 0.257963 | 0.261446 | 0.264491 | -0.280689 |

解释：

- Q 的轻微修正来自行 scale；
- Q+L 的修正也很小，因为低秩因子不需要逐元素 support；在目标 `0.258` 的相对 `±1%` 容差下仍可视为接近目标，但不能把 `0.258092/0.258126` 写成严格等于 `0.258`；
- Q+S/Q+S+L 的 support 和 FP16 value 使真实 rate 明显增加；
- attention+MLP 中 Q+S+L 相对 Q+S 的 PPL 优势只有 `0.004646`，而二者在减少 nnz 回到 exact-rate 后可能重新排序，必须重跑。

## 6. 已有结果统一重分析

### 6.1 四个 smoke run 的稳定性

数据来自 [strategy_stability.csv](../results/exact_rate_hessian_repair_20260713/strategy_stability.csv)。

| Run | Q | Q+L | Q+S | Q+S+L | 核心四法最优 |
|---|---:|---:|---:|---:|---|
| Pythia-70M PPL delta | +5.523 | +10.071 | +5.516 | +18.192 | Q+S |
| Pythia-160M | +0.196 | +0.702 | +0.384 | -0.748 | Q+S+L |
| Qwen2-7B attention-only | -0.100 | -0.537 | +0.443 | +0.165 | Q+L |
| Qwen2-7B attn+MLP | +0.384 | +0.029 | -0.276 | -0.281 | Q+S+L |

统一判断：

- `Q+S+L` 赢 `2/4`，信号依赖模型和层类型；
- Pythia-70M 对组合方法给出强负证据；
- Pythia-160M 和 Qwen attention+MLP 给出需 exact-rate 复核的正信号；
- Qwen attention-only 明确支持 Q+L，而不支持三分量堆叠；
- block-circulant/Monarch-like structured residual 在此前 matched-memory probe 中没有获得支持，不应与本轮 Hessian 小参数修补混为同一正结果。

### 6.2 正交/抵消重分析

[hessian_interactions.csv](../results/exact_rate_hessian_repair_20260713/hessian_interactions.csv) 显示：

- Q+S 的 `rho_H(Qerr,Sres)`：attention 为 `-0.6568`，attn+MLP 为 `-0.5874`；
- Q+L 的 `rho_H(Qerr,Lres)`：约 `-0.3186`、`-0.2385`；
- Q+S+L 中 `rho_H(Sres,Lres)` 接近零：约 `+0.0180`、`+0.0300`；
- 但 `rho_H(Q+S err,Lres)` 仍为负：约 `-0.0813`、`-0.1014`。

因此更准确的描述是：

1. S/L 两个**修补分量之间**接近二阶正交；
2. 它们相对当前累计误差主要表现为负相关修补；
3. “组合有效”的候选机制是互补修补，不是所有分量天然正交。

### 6.3 参数利用率判断

在旧 attention-only 结果中，Q+L 相对 Q 每增加 1% dense nominal payload 约改善 `0.56` PPL，且 payload 修正小，是当前最可信的高效率修补信号。

在旧 attn+MLP 结果中，Q+S/Q+S+L 的 nominal 收益看起来更高，但计入理想 support 后效率下降，计入可实现 CSR 后进一步下降，而且它们不再是 `0.258` endpoint。故当前只能说“稀疏修补可能对 MLP 层更有价值”，不能说它在 exact-rate 下已经优于 Q+L。

### 6.4 证据等级

旧结果统一定级为 **exploratory smoke evidence**，原因见 [evidence_flags.csv](../results/exact_rate_hessian_repair_20260713/evidence_flags.csv)：

- 单一 seed `17`；
- Qwen 两组实际只有 `63/126` 个 PPL token；
- calibration 和 evaluation 使用 shared text pool；
- 没有逐样本 NLL，不能 bootstrap 置信区间；
- Pythia-70M/160M 原始 run 未同步，只剩 aggregate CSV；
- 已保存 Pythia-70M 早期运行使用 fallback texts，不是标准 Wikitext benchmark。

这些限制不会让旧结果“无效”，但它们限制结论只能用于选实验方向，不能作为方法优越性的最终证据。

## 7. 新 exact-rate 实验设计

### 7.1 目标与对照

新实验应在同一模型、层集合、校准/验证划分和实际 codec 下比较：

| 策略 | 目的 | 关键 payload |
|---|---|---|
| Q | 4-bit RTN 主干 | codes + row scales |
| Q + global/component scale | 检验近零新增 bit 的幅度修补 | 系数折叠后重新计数 |
| Q + row-block scale | 检验增加少量 scale 是否比 S/L 更有效 | block scales 替换 row scales |
| Q+S | naive 稀疏残差 | FP16 values + CSR support |
| Q+S_OBS | 固定 support 后二阶重估 | 与 Q+S 相同 payload |
| Q+L | 低秩误差重构 | FP16 factors |
| Q+S+L | 三分量 residual stack | Q + CSR S + factors |
| Q+S_OBS+L | OBS 与低秩互补 | 同 support/rank 下比较 |
| Q+S+L + component scale | 可折叠小参数联合修补 | 编码后复测，不能用连续解 |

headline 目标首先是 `payload_ratio≈0.258`；可增加 `0.275/0.300` 作为 rate–distortion 曲线，但所有同率胜负必须在同一目标、同一 codec 和预先声明的相对容差内判断。

特别需要一个“只用量化自由度也把预算花到目标附近”的单方法对照，避免用 `rate≈0.250` 的普通 Q 与 `rate≈0.258` 的组合做不对称比较。例如输入维度 `n=512` 时，每行 4 个 FP16 block scale 的 4-bit Q payload 为 `0.25 + 4/512 = 0.2578125`。它既是 scale 机制消融，也是接近目标率的 quant-only baseline。

### 7.2 数据与层选择

建议的首轮可复现实验：

- 模型：`EleutherAI/pythia-70m`；
- 数据：真实 `wikitext/wikitext-2-raw-v1` 缓存，不允许静默 fallback；
- calibration/evaluation：按文本窗口严格分离；
- 层：MLP `dense_h_to_4h/dense_4h_to_h` 的 first/middle/last，最多 6 个模块；
- 量化：4-bit symmetric row-wise RTN；
- scale/sparse/low-rank 存储：按脚本实际 codec 记录，默认 FP16 repair values/factors；
- 随机性：首轮 seed `17` 用于工程闭环，正式结论至少补 3 seeds；
- endpoint：held-out token 数必须显著高于旧 `63/126`，并保存逐窗口 NLL。

对每个目标率，候选生成应直接在 bit 预算内枚举可行的 `rank/nnz/block-size`：先扣除 Q codes/scales 和固定 metadata，再用可实现 CSR 的逐非零成本及 `16r(m+n)` 的低秩成本分配剩余 bit。`Q+S+L` 不应先按浮点比例构造后再声称“约同率”，而应只从 codec 可行集合中选 endpoint。

运行脚本、默认参数和最终输出以 [run_pretrained_hessian_repair.py](../scripts/run_pretrained_hessian_repair.py) 为准；若该脚本尚未出现在当前提交中，本节仅是实验协议，不表示实验已经完成。

### 7.3 理论假设—实验字段映射

| 理论问题 | 直接观测 | 通过标准 | 结果文件 |
|---|---|---|---|
| scale 能否高效修补 | `cost_before/after`、scale 数、payload、PPL | 同 payload 支配 Q，或单位新增 bit 收益最高 | `candidate_ablation.csv`, `strategy_endpoints.csv` |
| OBS 是否有效 | naive/OBS 同 support 的 cost、stationarity、PPL | 相同 payload 下 OBS 不劣且 stationarity 降低 | 同上 |
| 是否达到正交 | raw cross、signed `rho_H`、`abs_rho_H` | 预设阈值内 `|rho_H|` 小；不能用负值截零 | `candidate_ablation.csv` |
| 是否存在有益抵消 | signed `rho_H<0` 与总代价下降 | cross term 与 endpoint 改善方向一致 | 同上 |
| 组合是否同率占优 | 声明 codec 的 value-stream payload 与 held-out NLL/PPL；E5 再核真实 artifact bytes | `D_combo(R)<min D_single(R)` 且 rate 合格 | `strategy_endpoints.csv` |
| proxy 是否可靠 | Hessian/activation ranking vs NLL ranking | 候选内排序稳定，不被明显差 baseline 虚高 | `strategy_endpoints.csv` |
| 是否有舒适区 | `epsilon` 路径真实 NLL 与小扰动拟合 | 小 epsilon 拟合好，endpoint 误差可接受 | `comfort_sweep.csv`, `comfort_summary.csv` |
| 参数是否有效 | recovered loss / actual bit、/ fitted DOF | 优于继续向单一组件分配预算 | `candidate_ablation.csv` |

### 7.4 loss landscape / comfort-zone probe

对最终 codec 扰动 `Delta=W_hat-W`，定义径向路径：

```text
W(epsilon) = W + epsilon Delta,  epsilon in [0,1]
```

真实任务 loss 局部展开为：

```text
L(W+epsilon Delta)-L(W)
= epsilon g^T Delta
+ 1/2 epsilon^2 Delta^T H Delta
+ O(epsilon^3)
```

注意：即使预训练模型总体接近驻点，替换少量层并在特定文本上测量时，线性项也未必为零。因此应拟合线性项和二次项，不能强迫曲线通过纯 `epsilon^2` 模型。

建议输出 `epsilon={0,0.125,0.25,0.5,0.75,1}`：

- `epsilon≤0.25` 用于局部拟合；
- 最大拟合误差不超过预设阈值的区间定义为 empirical comfort zone；
- `epsilon<1` 是诊断插值，不是可部署 codec；
- 只有 `epsilon=1` 对应实际编码 endpoint。

如需进一步观察 Q/S 或 S/L 的耦合，可增加二维方向面：

```text
L(W + alpha Delta_a + beta Delta_b)
```

二维面的混合二阶系数直接对应 `<Delta_a,Delta_b>_H`。该扩展若未生成独立 CSV/图，只能列为后续 probe，不能在结果章节中引用为已完成证据。

### 7.5 已完成的 Pythia-70M value-stream probe（同率结论已被物理复核取代）

结果目录为 [results/pretrained_hessian_repair_pythia70m_20260713](../results/pretrained_hessian_repair_pythia70m_20260713/)。本轮使用本地固定 revision 的 Pythia-70M、真实缓存 `wikitext-2-raw-v1/validation`、seed 17、CPU FP32；校准 8 个 128-token batch，评估 16 个 128-token batch。源文本按内容去重后再切分，校准 64 条、评估 128 条，`identical_text_overlap_count=0`，没有 fallback。实际 NLL token 为 2032。被替换的是 first/middle/last 的 6 个 MLP linear，共 6,291,456 参数；payload ratio **只针对这些选中权重的 FP16 reference**，不是整个 70,426,624 参数模型的端到端文件压缩率。

Dense baseline 为 NLL `4.26762648`、PPL `71.3520792`。目标 `R=0.258` 附近的关键 endpoint 如下；所有 sparse 行都计 FP16 value 与 fixed-width CSR，所有 low-rank 行都计两侧 FP16 factor。这里的“exact”是**声明 codec 的 tensor value-stream exact**：模型容器和 tensor descriptor 等格式相关共享 header 未计入；同 shape 的本轮配对中它们可抵消，但 E5 部署结论仍必须以真实导出 artifact bytes 为准。

| 策略 | 实际 ratio | 实际 bits | norm. H cost | H gain / added bit | `rho_H(S,L)` | Q<-S 抵消 | Q<-L 抵消 | PPL delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | 0.251221 | 25,288,704 | 0.0100771 | -- | -- | -- | -- | +15.6459 |
| Q + global folded scale | 0.251221 | 25,288,704 | 0.0099510 | zero-bit | -- | -- | -- | +15.2850 |
| Q + row-block scale | 0.257812 | 25,952,256 | 0.0057666 | 3.35287e-5 | -- | -- | -- | +9.4039 |
| Q+S | 0.257999 | 25,971,072 | 0.0068027 | 2.47674e-5 | -- | 0.6391 | -- | +9.2936 |
| Q+S_OBS | 0.257999 | 25,971,072 | 0.0058573 | 3.19181e-5 | -- | 0.8375 | -- | +7.1604 |
| Q+L | **0.257324** | **25,903,104** | 0.0046331 | 4.57327e-5 | -- | -- | 1.0804 | +6.7618 |
| Q+S+L，逐层受 Q+L bit cap | **0.257324** | **25,903,104** | 0.0046148 | 4.58866e-5 | 0.0230 | 0.2712 | 0.8129 | +6.5498 |
| 上行 + folded component scale | **0.257324** | **25,903,104** | **0.0045814** | **4.61677e-5** | **0.0231** | 0.2716 | 0.8112 | **+6.4408** |
| Q+S+L，普通 target cap | 0.257418 | 25,912,512 | 0.0046071 | 4.52583e-5 | 0.0291 | 0.2796 | 0.8061 | +6.5785 |
| 上行 + folded component scale | 0.257418 | 25,912,512 | 0.0045737 | 4.55346e-5 | 0.0293 | 0.2800 | 0.8044 | +6.4768 |

#### 同 value-stream bits 组合是否占优

这次旧运行在**仅计 value stream** 时观察到方向性优势：`Q+L` 与 `Q+S+L_QL_budget_component_scale` 都声明 `25,903,104 bits`，组合把 PPL delta 从 `+6.76182` 降到 `+6.44081`。但它没有计入真实容器 descriptor/alignment，不能再称为物理同率。2026-07-14 的研究 codec 复核加入所有这些字节后，严格同为 `3,248,832 B` 的组合反而比 Q+L 差 `0.55799 PPL`。因此本小节只保留为“为什么需要物理 codec”的历史对照，不再支持当前组合占优结论。

16 个固定窗口的配对 NLL 差 `combo - Q+L` 均值为 `-0.00411794`，11/16 窗口为负；描述性 `mean ± 1.96 SE` 区间为 `[-0.00862872, +0.00039285]`，仍包含 0。由于这些是同一语料的连续窗口而非独立样本，该区间不是总体置信区间；因此不能把一次 aggregate 胜出写成统计显著或普适优越。

#### 达到的“正交”是什么

在真正同时激活 S/L 的 3 个 down-projection 中，逐层 `|rho_H(S,L)|` 最大为 `0.0542`，聚合为 `0.0231`，低于预设 `0.1` 阈值；因此本轮确实观察到 **S 与 L 两个修补分量之间的近 Hessian 正交**。但 Q/S 和 Q/L 分别约为 `-0.376`、`-0.641`，它们相对 Q 是抵消修补。`Q<-S` 与 `Q<-L` 的 cross-term cancellation gain 为 `0.272`、`0.811`。所以准确结论不是“三个分量互相正交”，而是“两个 residual 修补子空间近正交，并分别抵消共同的 Q 误差”。

#### 小参数修补消融

- **OBS：** Q+S 与 Q+S_OBS payload 完全相同。OBS 改善 `2.13319 PPL`；配对窗口 NLL 改善为 `-0.0268075`，描述性区间 `[-0.0360356,-0.0175794]`，15/16 窗口获益。连续解 stationarity 为约 `1e-17--1e-15`，FP16 存储后最大相对 stationarity 为 `4.52e-4`。这是本轮最稳定的零额外 payload 修补证据。
- **scale：** global scale 折叠进已有 FP16 Q scale，不增加 payload，回收 Q 的 `1.25%` Hessian cost 并改善约 `0.361 PPL`；row-block scale 回收 `42.78%`，单位新增 bit 的 Hessian gain 为 `3.35e-5`。组合上的 component multipliers 也折叠进已计费的 Q scales、S values 和 L factors；相对未 scale 的同-bit 组合改善 `0.109 PPL`，但配对窗口区间仍跨 0，因此只算弱正信号。
- **参数利用率：** 普通 Q+L 已优于 block scale 与 CSR sparse；同-bit Q+S+L + folded scale 的 `4.61677e-5` 则是本轮最高 Hessian gain/added-bit。关键不是 repair 参数个数少，而是其每个真实存储 bit 是否对准高曲率误差，同时避免大量 support metadata。

#### loss landscape 与证据边界

径向 epsilon probe 的六个策略在当前 `20% relative / 1e-4 absolute` 容差下都延伸到 codec endpoint `epsilon=1`，Hessian proxy 与路径 NLL 的 Pearson correlation 为 `0.9973--0.9995`。但小 epsilon 拟合出的线性系数均为负，例如 Q 为 `-0.0378`、同-bit 最佳组合为 `-0.0154`；这说明当前 held-out slice 上不能假设 `g^T Delta=0`，纯二次 Hessian 排序并不完整。endpoint 拟合绝对误差为约 `0.0039--0.0098 NLL`。

这一旧结果仍只有一个模型、一个 seed、6 个局部 MLP tensor 和 2032 个 NLL token。其 value-stream 诊断可用于解释机制，但物理同率胜负以 [2026-07-14 后续审计](serialized_rate_liftquant_confirmatory_update_20260714.md) 为准；8-seed 产物目前也只是 data/split manifest，不是已运行的模型结果。

| 已验证产物 | 用途 |
|---|---|
| [candidate_ablation.csv](../results/pretrained_hessian_repair_pythia70m_20260713/candidate_ablation.csv) | 逐层候选、exact payload、self/cross、stationarity、修补效率 |
| [strategy_endpoints.csv](../results/pretrained_hessian_repair_pythia70m_20260713/strategy_endpoints.csv) | 三个目标率与 held-out endpoint |
| [endpoint_window_nll.csv](../results/pretrained_hessian_repair_pythia70m_20260713/endpoint_window_nll.csv) | dense 与每个 endpoint 的 16 窗口配对 NLL |
| [comfort_sweep.csv](../results/pretrained_hessian_repair_pythia70m_20260713/comfort_sweep.csv) / [comfort_summary.csv](../results/pretrained_hessian_repair_pythia70m_20260713/comfort_summary.csv) | epsilon loss landscape 与局部拟合 |
| [run_config.json](../results/pretrained_hessian_repair_pythia70m_20260713/run_config.json) | revision、数据 digest、环境、seed、codec、git 状态 |
| [summary.md](../results/pretrained_hessian_repair_pythia70m_20260713/summary.md) | 自动汇总 |
| [PNG](../results/pretrained_hessian_repair_pythia70m_20260713/figures/pretrained_hessian_repair_probe.png) / [PDF](../results/pretrained_hessian_repair_pythia70m_20260713/figures/pretrained_hessian_repair_probe.pdf) | exact-rate、endpoint、loss landscape、signed rho 图 |

## 8. 与同类型工作的逐项对比

本项目不能声称首次组合量化、稀疏和低秩，也不能声称首次使用 Hessian 做量化/剪枝补偿。更准确的定位是：**在明确计入 scale、稀疏 support/value、低秩因子和 repair 参数的实际 payload 下，对 Q/S/L 与小参数修补做局部 Hessian 交互诊断、舒适区验证和同率 endpoint 比较。**

为避免把不同训练预算、不同压缩对象和不同部署格式混成一张排行榜，本仓库新增了可复现的方法矩阵：

- 机器可读明细：[method_matrix.csv](../results/compression_method_comparison_20260713/method_matrix.csv)；
- 分层摘要：[summary.md](../results/compression_method_comparison_20260713/summary.md)；
- 生成与一致性检查：[build_compression_method_matrix.py](../scripts/build_compression_method_matrix.py)。

矩阵不填未经一手论文或官方实现核实的数值；“有论文结果”也不等于“本仓库已复现”。每一行分别记录方法范围、训练/梯度信号、实际更新状态、目标函数、必须计入的 payload、严格直接比较条件、复现状态和一手来源。

### 8.1 比较车道：训练预算和作用范围必须先对齐

| 顶层车道 | 细分 | 定义 | 与当前 exact-rate probe 的关系 |
|---|---|---|---|
| A：gradient-free / closed-form PTQ | A0 | data-free；不使用校准数据和反向传播 | 在相同权重范围、codec 与实际字节下可严格比较 |
| A：gradient-free / closed-form PTQ | A1 | 使用校准激活/统计、搜索、k-means、二阶代理或闭式解，但不通过梯度/STE 训练参数 | 当前方法的主车道；还须对齐校准 token 与统计成本 |
| B：calibration/backward-assisted PTQ | B | dense pretrained base 冻结；允许为局部重构、曲率统计执行 backward/HVP/STE，或优化最终可融合的量化/辅助状态，但不做全局任务恢复 | 单列统计/优化预算；既要区分“只收集 HVP”与“学习参数”，也不能伪装成 training-free |
| C：global recovery / QAT / downstream PEFT | C | 以全局 CE/KL 做梯度恢复、QAT、下游 PEFT/微调，或主流水线用任务损失改变全模型行为 | 只能作为允许全局训练的上界或单独比较，不与 A/B 车道混排 |
| D：范围不匹配 | D | 原生低比特训练、KV-cache、activation-only、动态运行时路径或 NAS/全局结构搜索 | 只做机制/系统边界对照，不作 weight-only PTQ 直接胜负 |

“严格直接比较”至少要求同时匹配：同一 dense checkpoint、同一压缩张量范围、W/A/KV 口径、相同校准数据与 token 预算、相同是否允许 backward/HVP/STE、相同未压缩张量，以及实际导出 artifact 的总字节。否则矩阵只给出机制性关系。RTN、GPTQ、AWQ、SparseGPT、Wanda、SpQR、LQER/QERA/EoRA、QTIP、D²Quant、ADMM-Q、HAS-VQ、SEPTQ、AAAC、Q-Palette data-free/data-aware 等属于 A0/A1；SqueezeLLM 主 Fisher 路径、OmniQuant、SliderQuant/SliderQuant+、LiftQuant block correction，以及只用 backward/HVP 收集曲率而不学习 rounding 参数的 YAQA 属于 B；SpinQuant optimized-rotation、AQLM 主报告流程、QuIP# 主流程、LiftQuant E2E、HESTIA、EfficientQAT、LLM-QAT 等属于 C；TurboQuant、SharQ、BitNet/QuEST 以及作用范围不匹配的结构搜索属于 D。具体变体以矩阵逐行为准，不能只凭方法名归类。

### 8.2 LiftQuant 必须拆成两条结果线

[LiftQuant](https://arxiv.org/abs/2606.04050) 的官方[实现](https://github.com/Heliulu/LiftQuant)包含两种性质不同的优化，不能合并成“training-free PTQ”一行：

1. **Block correction（车道 B）**：论文/README 声明 RedPajama 4096×2048、2 epochs、block output MSE 和 STE；但固定 commit parser 只有 `nsamples1/2`、`epochs1/2`，机械映射还会让每阶段实际迭代 3968 条，epoch 语义尚未由可执行协议确认。未补丁 Qwen2.5 layer-0 smoke 因 `Catcher.attention_type` 缺失失败；兼容补丁后的单层 smoke 不含 PPL/任务精度。
2. **可选 E2E（车道 C）**：论文/README 声明 block correction 后使用 4096×4096、1 epoch CE；固定 commit 当前因缺少 `datautils_block.py` 在参数解析前失败，且未知 `extra_args` 可能被静默忽略。它仍是微调 lane 的声明协议，不是本仓库已执行结果。

LiftQuant 的 nominal `D/d` 不是完整 rate。严格计费至少包括：

\[
B_{\text{Lift}}
=B_{\text{1-bit lifted codes}}
+B_{\text{scale}}
+\begin{cases}
B_{T^*}, & \text{部署存储 fused decoder},\\
B_M+B_{T^{-1}\text{ factors}}, & \text{部署存储 unfused representation},
\end{cases}
+B_{\text{padding/header/alignment}}.
\]

尤其要计 `ceil(IC/d)` 带来的 code padding，以及 transform 是按层存储、已融合还是能跨层摊销。`T^*=MT^{-1}` 已经包含 mapping 与 inverse whitening；若 artifact 存的是 `T^*`，就不能再次计 `M`，只有 unfused artifact 才按 `M + inverse factors` 计费。论文报告 70B 模型中 FP16 transform 的摊销开销约为 **0.008--0.011 bpw**；这不是零开销。LiftQuant 的核心实验聚焦 2--3 bit，当前仓库的 4-bit probe 只能验证“变换/修补是否改善局部 Hessian 或 endpoint”的机制，不能当作 LiftQuant 主设置的直接复现或精度排名。

### 8.3 小参数修补：从 scale 保护扩展到 sparse/pruning

[D²Quant](https://arxiv.org/abs/2602.02546) 是当前最直接的“小参数修补”启发。其 DSQ 对 down-projection 交替优化列尺度与量化值；若该尺度确实吸收到相邻 up-projection 已有量化尺度中，部署增量可以为零。其 DAC 则根据校准前向估计 post-attention LayerNorm 输出均值漂移，并新增每层 hidden-size bias。因而实验必须分别报告“可折叠 scale 的零增量条件”和 `dtype_bits × hidden_size × layers` 的 DAC bias 实际字节，不能把算法参数少直接写成零 payload。

[AAAC](https://arxiv.org/abs/2605.08692) 用 activation-weighted k-means 在每层学习两个 16-entry BF16 codebook，无梯度或 Hessian 微调。它说明 4-bit 权重也可用很小的离散 side information 修补非均匀误差，但必须计每层 64 B codebook；只有 selection group 与 scale group 对齐、且导出后端保留正 scale 的 sign bit 时，codebook 选择位才可能复用为零，否则还要计一位/组。截至 2026-07-13 未找到官方 GitHub，因此本仓库只把它列作 paper-only 设计基线。

这两类机制可以迁移到稀疏/剪枝，但条件比量化 scale 更严格：固定 support 后的 OBS value refit、每组保留值 scale、每层 bias 或小 rank residual 都可以用少量参数抵消删除误差；support 本身的 mask/index 却不能省略。应比较的是

\[
\eta_{\text{repair}}
=\frac{\Delta \mathcal L_{\text{compression}}
-\Delta \mathcal L_{\text{compression+repair}}}
{B_{\text{repair}}},
\]

并与“把同样 bit 直接用于保留更多权重、提高局部 bitwidth 或降低稀疏率”对照。只有当小修补方向集中覆盖高曲率误差、且每 bit 回收损失更高时，它才具有参数利用优势。

[DAQ](https://arxiv.org/abs/2603.22324) 提供的是另一条窄边界：它需要 base 与 post-trained 两份权重，data-free 搜索 FP8 scale，使任务增量 `Delta W` 的符号保持率和 cosine 更高。它目前面向 DeepSeek-V3 toy-style SFT 的 FP8 pilot，并非通用 sub-4-bit PTQ。只有确实存在 base/post checkpoint 时，才可把 delta sign/cosine 加入本项目的次级诊断；不能替代 held-out NLL、actual bytes 或常规 weight-only 基线。

### 8.4 最新方法的逐项协议、预期和可验证关系

以下比较只固定“应该测什么”，不把论文结果当成本仓库实测。每个方法都要先回答训练预算、更新状态和 payload，再讨论它能否支持本项目关于正交、小参数修补或同率优势的假设。

- **[ADMM-Q](https://arxiv.org/abs/2605.11222)（A1）**：它把 GPTQ 的逐列贪心替换为整层 Hessian-weighted 离散优化：连续闭式更新、量化投影、可选 grid refresh 与至多 5 轮 pair-swap local search 都发生在 encoder 端，没有模型反传或参数微调，因此“有迭代”并不使其进入 B/C。它最适合作为本项目 Hessian proxy 的 solver 对照：固定同一 `H=X^T X`、grid、clip、group 和 payload，只替换 RTN/GPTQ/ADMM-Q，再检查较低的层 reconstruction 是否迁移到 held-out NLL。部署计最终 codes、scale/zero-point 与外围 rotation/scale 状态；ADMM primal/dual、特征分解和 Hessian 只有在不被序列化时才可免计。截至 2026-07-13，论文未给出官方代码链接，也未确认作者指定仓库，故只列作 paper-only、待外部复现。
- **[HAS-VQ](https://arxiv.org/abs/2601.06959)（A1）**：它用对角敏感度掩掉高曲率坐标，对剩余 dense body 做 block k-means VQ，再以 sparse residual 精确修补敏感位置。论文与[官方实现](https://github.com/VladimerKhasia/HASVQ)显示该流程不反传；参考脚本用 128 个长度 1024 的 WikiText-2 前向估计 activation second moment，随后迭代 k-means。它是“稀疏小参数修补能否改善量化”的直接对照，但 sparse residual 不是零开销：必须同时计 vector indices、完整 codebook、channel scales、residual values、support indices/bitmap/row pointers、block padding 和 headers。论文/脚本的分析 BPP 不能替代真实 packed artifact。
- **[SEPTQ](https://arxiv.org/abs/2604.10091)（A1）**：它先用 Hessian-based importance 在整层静态选择需要保护的权重，再在其余位置做 GPTQ 式逐列量化与闭式误差传播；论文使用 128 个长度 2048 的 C4 校准片段，没有 gradient/STE 学习。它可检验“先把 bit 用于保留高曲率值，还是用于 sparse/low-rank repair”的参数利用率。严格 rate 必须计低比特 codes/scales、保留高精度值以及区分两类位置的 support；除非真实混合格式能隐式编码 support，mask 不能只当 encoder 临时量。截至核验日论文仅说明基于 GPTQ 实现，未给出可确认的官方公开仓库，因此列作 paper-only、待外部复现。
- **[HESTIA](https://arxiv.org/abs/2601.20745)（C）**：它先用 Hutch++ Hessian trace 为每个 tensor 设定 softmax 温度退火，再在 10B Ultra-FineWeb token 上以 AdamW 和全局 causal-LM loss 做 1.58-bit ternary QAT；[官方实现](https://github.com/hestia2026/Hestia)也明确是 calibration + distributed QAT 两阶段。故 Hessian 只决定 annealing schedule，并不把后续全模型训练变成 PTQ。它应作为“允许大规模训练的低比特上界”单列，并计真实 packed ternary stream、group-128 scale、未量化 tensor 与 artifact header，而不是直接采用名义 `log2(3)=1.58` bit；训练用 Hessian/temperature/optimizer state 只有在部署产物不依赖它们时才不计。
- **[SliderQuant](https://arxiv.org/abs/2603.25284) / SliderQuant+（B）**：默认路径的预期是借助跨层滑窗，把浅层/深层较高敏感性和相邻层误差传播纳入同一个局部 reconstruction problem；它在 128 个长度 2048 的样本上用 AdamW 学习 channel scales 与所有线性层的 rank-4 LoRA，默认 20 epochs，W2A16 为 60 epochs。dense base 冻结，因此按本报告定义属于 B，而不是下游 PEFT；默认导出把 scale/LoRA 吸收到权重。SliderQuant+ 另加不可吸收的 Hadamard 运行时变换，必须单列 transform plan、kernel、workspace 与 latency。对本项目最直接的消融是：固定同一 codec 和训练预算，对比单层、固定窗口、adaptive window、`+scale`、`+rank-4`，并检查窗口扩大是否降低 held-out endpoint，而不只降低 calibration reconstruction。
- **[Q-Palette](https://arxiv.org/abs/2509.20214)（A0/A1）**：data-free 路径从 scalar/vector/trellis quantizer palette 中按 rate--distortion 与系统约束选方案，属于 A0；data-aware 路径复用 QTIP proxy Hessian，并以实际 validation perplexity degradation 做 mixed-scheme allocation，仍无 retraining，但属于 A1。理论预期是异质 codec 能把 bit 分给局部边际 distortion/bit 更高的位置；实验必须把 scheme-selection data 与最终 held-out evaluation 分开，并计 streams、scales、codebooks/LUTs、incoherence signs/seeds、scheme IDs、fusion/merge plan 和 padding。它是本项目“同率组合是否优于单一 codec”的最直接离散分配对照，但不能用熵下界代替 serialized bytes。
- **[SINQ/A-SINQ](https://arxiv.org/abs/2509.22944)（A0/A1）**：SINQ 用权重本身和 Sinkhorn-style normalization 找第二轴 scale，预期在无数据时近似恢复 activation-aware channel importance；A-SINQ 再叠加 AWQ 式校准统计。两者的额外自由度很小，适合检验“少量 scale 参数是否以较高 loss-recovery/bit 改善舒适区”。必须计普通 scale、第二轴 scale、zero-point 和 axis/group metadata，并分别报告默认运行时 scale 与满足共享尺度约束时的可吸收版本。
- **[SRR](https://arxiv.org/abs/2602.02001)（A1）**：Structured Residual Reconstruction 先保护 activation-scaled weight 的 top-`k` singular subspace，只量化其 residual，再把剩余 `r-k` rank 用于误差重构。其设计预期不是单纯增大 low-rank rank，而是在固定 rank/payload 下把容量分给“保留高曲率方向”和“修复量化残差”。应 sweep `k/r`、factor dtype 和实际 bytes，并与全 rank 都用于 residual reconstruction 的 QERA/LQER 风格基线比较；可选 QPEFT 属 C，不能与这里的 PTQ 行合并。
- **[ResComp](https://arxiv.org/abs/2604.07955)（A1）**：它把逐步量化目标重新对齐到原始 FP 输出，并把 compensated weight 与 original weight 的差异纳入 compensation-aware error。预期是修正 sequential compensation 中被错误参考目标累积的偏差。对本项目应在同一 GPTQ/GPTAQ codec 下只开关 ResComp 项，报告每列/每 block loss landscape 与最终 PPL；若补偿完全硬化进最终 quantized weights，则没有独立部署 tensor，否则按 artifact 计费。
- **[FOEM](https://arxiv.org/abs/2507.11017)（A1）**：FOEM 指出 progressive compensation 后的 latent weights 已偏离原始最优点，因此一阶项不再可忽略；它用 latent--original weight difference 与二阶结构近似该项，而不是实时反传。其预期正好对应本项目要检查的“纯 `r^T H r` 是否漏掉线性项”。最小消融是在相同 GPTQ codec、校准集和顺序下比较 second-order-only 与 FOEM，记录实际一阶项、proxy 排序和 held-out endpoint。论文算法把修正硬化进最终权重，没有独立 FOEM runtime state；官方仓库是 [Xingyu-Zheng/FOEM](https://github.com/Xingyu-Zheng/FOEM)，GPTQModel 只作集成入口。
- **[YAQA](https://arxiv.org/abs/2505.22988)（B）**：YAQA 用全模型 KL 的 Kronecker Hessian sketch 改善 layerwise rounding proxy；曲率收集可能需要 backward/HVP，但 quantized representation 由 fixed-point/LDL-style rounding 产生，并不学习连续 rounding 参数。其理论预期是更接近真实 end-to-end Hessian 的 sketch 会给出更可靠的 rounding 排序；实验应比较相同底层 quantizer 下 input-covariance proxy 与 YAQA sketch 的 cosine/ranking/endpoint，同时报告 sketch rank、tokens、HVP/GPU-hours。Hessian、curvature factor 和 search state 都是 encoder-side，不进入部署 payload。
- **[SharQ](https://arxiv.org/abs/2606.26587)（D）**：SharQ 在线生成 input-adaptive N:M activation mask，把 activation 分成 sparse FP4 backbone 与相对该量化 backbone 定义的 dense FP4 residual，两条 GEMM 共享一个 FP4 weight payload并使用不同 scale views。预期是 dense residual 同时补偿 mask loss 和 sparse-path quantization loss；这与本项目的静态 weight Q/S/L residual 几何有机制相似性，但压缩对象、动态 metadata 和 kernel 均不同。它只能单列 activation mask、path-specific scales、runtime workspace、fused preparation kernel、latency/throughput，不能用其系统结果证明 weight-only exact-rate frontier。

[MXFP PTQ benchmark](https://arxiv.org/abs/2601.09555) 不作为新方法结果行，而作为数值格式协议参考：MXFP4/8 的 block value stream 之外还必须计共享 E8M0 scale、block padding 与 header；论文指出 scale 的 power-of-two 约束是重要误差源，并用量化前固定 `3/4` pre-scale 缓解 clipping bias。因而后续若加入 MXFP，必须把 `pre-scale on/off` 作为独立消融，并禁止把 nominal MXFP4 的 4 bit 与 INT4 的实际 artifact 字节直接等同。

### 8.5 正交、抵消和理论—实验关系

令压缩残差为 `r`、候选修补方向为 `c`，并记 `〈a,b〉_H=a^T H b`。对两个独立压缩组件，关注

\[
\rho_H(i,j)=\frac{\langle\delta_i,\delta_j\rangle_H}
{\sqrt{\langle\delta_i,\delta_i\rangle_H}
 \sqrt{\langle\delta_j,\delta_j\rangle_H}},
\]

希望 `|rho_H|` 小，从而二阶损失近似可加。对 compression--repair 则相反：有用修补应使 `〈r,c〉_H<0`。若目标写成 `||r+gamma c||_H^2`，最优系数为

\[
\gamma^*=-\frac{\langle c,r\rangle_H}{\langle c,c\rangle_H},
\qquad
\text{recovered cost}=\frac{\langle c,r\rangle_H^2}{\langle c,c\rangle_H}.
\]

若沿用更新参数化 `r+(alpha/2)c`，同一最优点写成项目要求记录的 **repair cancellation gain**：

\[
\alpha^*=-\frac{2\langle c,r\rangle_H}{\langle c,c\rangle_H}.
\]

这一定义澄清了因子 2 来自参数化，而不是另一种物理效应。实验上，compression--compression 报告 `|rho_H|` 和组合二阶项；compression--repair 报告负交叉项、最优 gain、实际 endpoint 回收量和每 payload bit 收益。小扰动 comfort zone 内的二阶可加性只能支持局部机制；最终“同率组合优于单方法”仍必须由 `epsilon=1` 的 held-out NLL/PPL、重复 seed 与实际导出字节证明。

跨方法统一 payload 应为：

\[
B_{\rm total}=B_{\rm codes}+B_{\rm scale/zp}+B_{\rm transform}
+B_{\rm codebook}+B_{\rm mask/index}+B_{\rm lowrank}
+B_{\rm bias}+B_{\rm padding/header/alignment}.
\]

同时记录 calibration samples/tokens、优化 steps/epochs、是否反传、更新变量、GPU-hours、峰值显存、部署 kernel 与实测 latency。理论负责提出局部几何和 bit-efficiency 的可证伪预测；实验负责检查 proxy 排序是否迁移到 held-out endpoint，而不是用二阶公式替代最终质量证据。

| 工作 | 主要机制 | 与本项目的关系 | 本项目不能越界的声明 |
|---|---|---|---|
| [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper_files/paper/1993/file/b056eb1587586b71e2da9acfe4fbd19e-Paper.pdf) | 经典二阶剪枝与其余权重补偿 | 本项目的 support 内重估沿用 OBS 式 stationarity/Schur 思想 | 当前只用输入协方差 proxy，不是完整全网络 OBS |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 近似二阶 one-shot 权重量化和误差传播 | 提供 Hessian-aware quantization 基准思想 | 当前 Q 主干是 RTN 时，不能称为 GPTQ 复现或优于 GPTQ |
| [SparseGPT](https://arxiv.org/abs/2301.00774) | 二阶 one-shot LLM 剪枝，支持半结构模式并可与量化兼容 | 与 OBS sparse refit 最接近的成熟剪枝线 | 本项目未完整复现 SparseGPT 的逐列更新和系统结果 |
| [Wanda](https://arxiv.org/abs/2306.11695) | 权重幅度与激活统计结合的 pruning saliency | 可用于选 sparse support；本项目进一步固定 support 后重估值 | saliency 选择与 Hessian 最优重估是两个步骤 |
| [AWQ](https://arxiv.org/abs/2306.00978) | 保护少量 activation-salient 通道，通过等价缩放降低量化误差 | 直接启发 scale 保护和低存储开销修补 | 本项目的 post-decode block scale 不等同于 AWQ 的等价通道变换 |
| [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html) | 在权重与激活之间迁移量化难度，面向 W8A8 | 说明 scale 可改变量化条件而不必保留混合精度异常值 | 本项目目前重点是 weight-only residual，不是 SmoothQuant 复现 |
| [OmniQuant](https://arxiv.org/abs/2308.13137) | 通过校准学习等价变换和 clipping | 同属少参数量化修补 | 本项目强调闭式局部修补和 exact payload，不能声称替代其端到端校准优化 |
| [SpQR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) | 低比特主体加稀疏高精度异常值 | 与 Q+S 表示最接近；support overhead 是关键 | 旧 nominal Q+S 未计 support，不能直接与 SpQR 的实际格式比较 |
| [SqueezeLLM](https://arxiv.org/abs/2306.07629) | dense-and-sparse quantization，敏感异常值稀疏保留 | 同样说明少量 sparse correction 可能高效 | 本项目需补真实 kernel/latency，不能只凭参数率声称系统优势 |
| [LQER](https://proceedings.mlr.press/v235/zhang24j.html) / [QERA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/21718991f6acf19a42376b5c7a8668c5-Abstract-Conference.html) / [EoRA](https://arxiv.org/abs/2410.21271) | 低秩量化误差重构及其激活/特征空间感知改进 | 与 Q+L、whitened low-rank 修补直接相关 | 本项目的价值不在首次低秩误差补偿，而在 Q/S/L 交互与同率诊断 |
| [LQ-LoRA](https://arxiv.org/abs/2311.12023) / [LoftQ](https://arxiv.org/abs/2310.08659) | 量化基座与低秩适配/初始化联合，通常服务微调 | 说明 quantized base + low-rank residual 是成熟范式 | 本项目是 training-free PTQ probe，不能把微调收益当作直接对照 |
| [HAWQ-V2](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html) | Hessian trace 指导混合精度 rate 分配 | 与“边际失真/bit”和层敏感度分配相关 | 当前输入协方差二次型不是 HAWQ 的完整 Hessian trace |
| [QuIP](https://arxiv.org/abs/2307.13304) | 通过 incoherence processing 和二次目标改善极低比特量化 | 说明几何预处理可改变量化舒适区 | 本项目未实现 QuIP 的随机正交预处理或 lattice codec |
| [Effective Interplay between Sparsity and Quantization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ed032b08a8822c3635cdcd961012ce60-Abstract-Conference.html) | 从理论和实验说明 Q/S 非正交且顺序重要 | 直接支持本项目拆分 signed cross term、顺序和 endpoint 的必要性 | 不能把“希望正交”当默认事实；需实测最终解码误差 |
| [SLiM](https://deepmind.google/research/publications/148040/) | one-shot 量化、半结构稀疏和低秩补偿的统一流程 | 是最直接的 Q+S+L 同类工作，且包含硬件友好目标 | 本项目绝不能声称首次 Q+S+L；应与其表示、2:4 稀疏和速度指标区分 |
| [QSLR](https://doi.org/10.1109/ACCESS.2025.3615473) | quantized sparse + low-rank factorization，并对组件做 Hessian-aware quantization | 同样直接覆盖 Q/S/L，说明三分量组合本身不是新颖性来源 | 未按其论文协议和官方实现复现前，不能写成直接精度优劣比较 |
| [Optimal Brain Restoration](https://arxiv.org/abs/2509.11177) | 以二阶代理和闭式 group compensation 联合量化与稀疏 | 与本项目 OBS/group-scale 修补高度接近，是必须对照的最新工作 | 当前小规模 Pythia probe 不能声称在 joint Q/S restoration 上领先 OBR |
| [HWPQ](https://arxiv.org/abs/2501.16376) | Hessian-free pruning-quantization，强调压缩时间和 2:4 执行 | 提供“不计算 Hessian”的效率对照 | 本项目若保留协方差求解，必须报告预处理成本，不能只比 PPL |
| [Joint Structural Pruning and Mixed-Precision Quantization](https://arxiv.org/abs/2606.07819) | 跨层误差传播下联合结构剪枝与混合精度搜索 | 提醒局部逐层 proxy 可能漏掉全局传播 | 本项目目前是局部候选诊断，不是全模型联合搜索 |
| [HASSLE-free sparse+low-rank](https://openreview.net/forum?id=hyN75SAJTI) | 对 sparse + low-rank 的精确局部重构和系统协同 | 与不含 Q 的 S+L 分配相关 | 本项目应避免只用 relaxed Frobenius 指标声称分解最优 |

### 8.6 可信的差异化贡献边界

若新实验得到支持，最稳妥的贡献表述是：

> 在声明 codec 的 value-stream payload 精确计数下，系统分析量化、稀疏、低秩及可折叠小参数修补在输入协方差 Hessian 几何中的 self/cross term；区分二阶正交与负相关修补，并用同 codec-stream rate、held-out NLL/PPL 与 loss-landscape comfort-zone 验证组合何时优于单一压缩；完整部署结论另以真实序列化 artifact bytes 验证。

不应使用的表述包括：

- “首次组合 Q/S/L”；
- “证明量化与稀疏天然正交”；
- “Hessian proxy 下降必然带来 PPL 下降”；
- “nominal 0.258 即真实 0.258”；
- “小 epsilon 曲线好就代表 codec endpoint 好”；
- “参数更少就必然推理更快”；
- “小模型、少层、单 seed smoke 已超过同类方法”。

## 9. 复现与审计命令

### 9.1 重新生成旧结果审计

在仓库根目录执行：

```bash
PYTHONPATH=src python scripts/audit_existing_compression_results.py \
  --repo-root . \
  --output-dir results/exact_rate_hessian_repair_20260713 \
  --target-ratio 0.258 \
  --rate-tolerance 0.01
```

重新绘图：

```bash
python results/exact_rate_hessian_repair_20260713/figures/plot_existing_result_audit.py
```

运行单元测试：

```bash
PYTHONPATH=src pytest -q \
  tests/test_hessian_repair.py \
  tests/test_exact_rate_audit.py
```

### 9.2 新预训练模型 probe

完整参数以脚本 `--help` 和提交中的 [run_config.json](../results/pretrained_hessian_repair_pythia70m_20260713/run_config.json) 为准。本次 210 CPU 运行使用：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
MODEL_SNAPSHOT=/home/wangmeiqi/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
PYTHONPATH=src /home/wangmeiqi/anaconda3/envs/base-2-bitnet/bin/python \
  scripts/run_pretrained_hessian_repair.py \
  --model "$MODEL_SNAPSHOT" \
  --output-dir results/pretrained_hessian_repair_pythia70m_20260713 \
  --device cpu --svd-device cpu --torch-dtype float32 --local-files-only \
  --dataset wikitext --subset wikitext-2-raw-v1 --split validation \
  --calib-limit 8 --eval-limit 16 --sequence-length 128 --batch-size 1 \
  --selector-activation-sample-rows 256 \
  --module-types dense_h_to_4h,dense_4h_to_h \
  --layer-positions first,middle,last --max-modules 6 \
  --bits 4 \
  --target-ratios 0.258,0.275,0.300 \
  --endpoint-target 0.258 \
  --support-encoding csr_fixed --s-method wanda --l-method whitened_svd \
  --repair-block-sizes 32,64,128,256,512 --max-allocation-ranks 32 \
  --seed 17
```

若实际脚本参数名与上述协议不同，应更新本命令，并以生成的 `run_config.json` 为唯一事实来源。不能为了让命令通过而启用 fallback texts 后继续把结果称为 Wikitext。

## 10. 环境漂移与复现边界

本轮 210 服务器实际环境为：

```text
conda env: base-2-bitnet
Python:    3.10.18
PyTorch:   2.9.0+cu128
Transformers: 4.57.1
Datasets:  4.3.0
NumPy:     2.2.6
Matplotlib: 3.10.7
Compute:   CPU FP32，taskset 88-95，nice 10；机器虽有 A800，本次未占用 GPU
```

历史环境记录为 Python 3.10.20、PyTorch 2.10.0+cu128、Transformers 4.45.2、Datasets 2.15.0，当前无法原样复现。新实验必须在 `run_config.json` 中记录：

- git commit；
- Python/package/CUDA/GPU；
- 模型 revision 和本地 snapshot/hash；
- 数据集 fingerprint、文本来源和 split policy；
- calibration/evaluation token 数；
- seed；
- 每层 shape；
- codec、index width、scale/value/factor 位宽、alignment；
- 实际命令和 wall-clock 时间。

旧 Pythia smoke 采用 fallback texts，而新实验采用真实缓存 Wikitext 且按内容去重切分。两者的数据不同，不能把 PPL 数字直接做纵向提升百分比；只能分别作为旧 smoke 与新 benchmark-like probe 报告。

## 11. 证据升级路线

建议按以下等级推进：

| 等级 | 最低要求 | 可支持的结论 |
|---|---|---|
| E0 单元测试 | payload、OBS stationarity、scale monotonicity、约束可行性 | 实现与数学性质基本正确 |
| E1 离线层重构 | 多层 activation/Hessian、exact payload、交叉项 | 机制是否值得做 endpoint |
| E2 单模型 smoke | disjoint data、足够 token、逐窗口 NLL | 该模型/层子集上的初步端到端信号 |
| E3 重复实验 | ≥3 seeds、置信区间、多个 rate | 组合优势是否稳定 |
| E4 跨模型/任务 | 多模型 family、全层、标准 PPL/zero-shot | 方法泛化性 |
| E5 系统证据 | 真实编码文件、峰值显存、吞吐、延迟、能耗 | 部署收益 |

当前旧结果主要处于 E1–E2 的 exploratory 区间；已完成的新 probe 达到 E2，但单 seed、局部层覆盖和 2032 tokens 尚不足以升级为 E3/E4。

## 12. 最终判定模板

新结果生成后，用以下顺序作结论，避免被单一 proxy 误导：

1. 实际 payload 是否在目标容差内？若否，停止“同率”比较。
2. 最终解码权重的 self/cross term 是正交、抵消还是放大？
3. 修补收益是否在 FP16/实际存储舍入后仍保留？
4. 同 payload 下，OBS/scale 是否支配对应 naive 方法？
5. 组合是否同时优于最佳单方法的 Hessian cost、held-out NLL/PPL？
6. 小 epsilon 的二阶拟合是否能预测 `epsilon=1`？若不能，Hessian 只作局部解释。
7. 优势是否跨 seed、层类型和 rate 稳定？
8. support/factor/scale 的真实文件大小和 kernel 性能是否与参数率一致？

只有第 1、3、5、7 项同时通过，才适合写“组合压缩在相同压缩率、很小精度损失下优于单一压缩”；若只通过第 2 或第 6 项，则只能写“观察到可解释的局部几何机制”。

## 13. 文件索引

- 核心 Hessian/repair/payload 实现：[src/llm_spectral_dynamics/structured/hessian_repair.py](../src/llm_spectral_dynamics/structured/hessian_repair.py)
- 旧结果审计脚本：[scripts/audit_existing_compression_results.py](../scripts/audit_existing_compression_results.py)
- 新 exact-rate/loss-landscape runner：[scripts/run_pretrained_hessian_repair.py](../scripts/run_pretrained_hessian_repair.py)
- 新 probe 自动摘要：[results/pretrained_hessian_repair_pythia70m_20260713/summary.md](../results/pretrained_hessian_repair_pythia70m_20260713/summary.md)
- 旧结果审计摘要：[results/exact_rate_hessian_repair_20260713/audit_summary.md](../results/exact_rate_hessian_repair_20260713/audit_summary.md)
- payload 重算：[results/exact_rate_hessian_repair_20260713/payload_audit.csv](../results/exact_rate_hessian_repair_20260713/payload_audit.csv)
- Hessian 交叉项：[results/exact_rate_hessian_repair_20260713/hessian_interactions.csv](../results/exact_rate_hessian_repair_20260713/hessian_interactions.csv)
- 策略稳定性：[results/exact_rate_hessian_repair_20260713/strategy_stability.csv](../results/exact_rate_hessian_repair_20260713/strategy_stability.csv)
- 同率配对审计：[results/exact_rate_hessian_repair_20260713/rate_matched_pairs.csv](../results/exact_rate_hessian_repair_20260713/rate_matched_pairs.csv)
- 证据风险标记：[results/exact_rate_hessian_repair_20260713/evidence_flags.csv](../results/exact_rate_hessian_repair_20260713/evidence_flags.csv)
- 外部方法矩阵生成器：[scripts/build_compression_method_matrix.py](../scripts/build_compression_method_matrix.py)
- 外部方法机器可读矩阵：[results/compression_method_comparison_20260713/method_matrix.csv](../results/compression_method_comparison_20260713/method_matrix.csv)
- 外部方法分层摘要：[results/compression_method_comparison_20260713/summary.md](../results/compression_method_comparison_20260713/summary.md)
- 单元测试：[tests/test_hessian_repair.py](../tests/test_hessian_repair.py)、[tests/test_exact_rate_audit.py](../tests/test_exact_rate_audit.py)、[tests/test_compression_method_matrix.py](../tests/test_compression_method_matrix.py)
