# 官方 SLiM、OBR、QERA、EoRA 对照执行合同

更新时间：2026-07-17

## 1. 固定的官方源码

210 上的只读审计目录：

```text
/home/wangmeiqi/codex_external/baseline_audit_20260717/
```

| 方法 | 官方仓库 commit | 当前可比性 |
|---|---|---|
| SLiM | `d5ae86762db4f75d2f1fb83c9f23849ce489811a` | 最接近完整 Q/S/L，但默认是模型级 2:4/量化/LoRA 表示 |
| OBR | `960913ce40e662f6045edf3c06e77b5d8e597d5f` | 官方主路径是 rotation + W4A4KV4 + 50% sparsity，不是 selected-weight W-only |
| QERA | `bd7fc86a2e44d41f95b9b0421f27f5624dd37064` | 可作 Q+L 训练-free PTQ 对照，但需要统一模型/数据/rank/factor bytes |
| EoRA | `6a42e2edcc7559422d14ccf79b0105b2d8a78c76` | 可作 eigenspace Q+L 对照；legacy CUDA 和新版 GPTQModel 路径需分别标记 |

不得用仓库名称代替具体 commit、命令和导出 artifact。

## 2. 公平比较必须同时匹配

每个可进入主表的 official baseline 必须匹配：

1. 相同 dense checkpoint 和 revision；
2. 相同被压缩 tensor 集合；
3. 相同未压缩 tensor；
4. 相同 W/A/KV 范围；
5. 相同 calibration split、文本和 token 数；
6. 相同 validation/test 分离；
7. 相同是否允许 backward、fine-tuning、learned rotation；
8. 相同 natural serialized bytes；
9. 相同质量评估窗口；
10. 若比较速度，相同 backend、batch、sequence length 和硬件。

任何一项不满足时，只能放入机制对照或 Pareto 图，不能标成 rate-matched 胜负。

## 3. 方法级适配结论

### SLiM

官方实现支持 quantization、Wanda/SparseGPT/magnitude pruning、2:4 或 unstructured sparsity、low-rank adapter、adapter quantization、checkpoint 保存和 Sparse Marlin 加速。

主表适配需要：

- 禁止 optional fine-tuning；
- 固定 selected MLP tensor scope；
- 记录 base codes、scale、sparse support/value、LoRA factors 的实际文件；
- 将低秩 rank fraction 转为明确 rank；
- 分开报告 separate LoRA 与 merge-back 表示；
- exact-natural bytes 与本方法匹配后再评估。

### OBR

官方示例主要是：

```text
rotation + GPTQ/activation/KV quantization + 50% pruning + group compensation
```

这与当前 W-only selected-linear scope 不同。第一阶段只做机制对照；要进入主表，需要新增官方代码最小 patch，使其：

- 只处理同一 selected tensor scope；
- 关闭 A/KV quantization；
- 固定相同 W4 group size；
- 导出 compensation 折回后的真实 sparse low-bit artifact；
- 使用相同 train/validation/test 数据角色。

若 patch 改变核心算法，必须同时保留 unmodified official run，并把适配版标为 `official-derived scope adapter`。

### QERA

QERA 是最直接的 Q+L 对照。需要固定：

- quantizer 与 group size；
- rank 和 factor dtype；
- activation scaling/calibration token；
- selected tensor scope；
- 是否保存完整 quantized model 或仅 adapter；
- natural bytes。

QERA 不含 sparse/no-joint，因此用于判断异构 S/Q/L 是否优于同字节 Q+L，而不是证明同层 S+L 的唯一价值。

### EoRA

EoRA 提供 eigen/activation/SVD low-rank compensation，并包含量化 factor/kernel 路径。需要分开：

- legacy repository implementation；
- 当前 GPTQModel EoRA integration；
- fused 与 non-fused inference。

主质量表应使用同一 selected scope 和 natural bytes；速度表只能比较实际可执行的 fused backend。

## 4. 建议执行顺序

1. 在 Llama-2-7B 单层 `gate_proj` 做 API/import smoke；
2. 在三深度 `gate_proj,up_proj` 做相同 scope 的 Q+L official comparison；
3. 先接 QERA/EoRA，再接 SLiM；
4. OBR 先做 whole-model mechanism reproduction，再决定 selected-scope adapter；
5. 每个方法输出独立 manifest：

```text
official_repo
official_commit
patch_sha256
model_revision
compressed_tensor_names
calibration_digest
validation_digest
test_digest
artifact_file_sha256
artifact_natural_bytes
runtime_backend
quality_metrics
```

## 5. 主表与附表

主表只接受同范围同 natural bytes 的结果：

```text
Q
Q+L official QERA/EoRA
pure-S
pure-L
no-joint
QSL
SLiM official-derived matched-scope
OBR official-derived matched-scope
```

whole-model W4A4KV4、2:4 kernel speedup、learned rotation 或 fine-tuned adapter 放在附表/Pareto 图中，不能与 selected-weight W-only NLL 直接排序。
