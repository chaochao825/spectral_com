# 外部压缩方法分层对比矩阵

> 生成日期与来源核验截止：2026-07-16。本页由 `scripts/build_compression_method_matrix.py` 生成。

## 这张表是什么、又不是什么

这是一个**文献协议矩阵（literature matrix）**，用于在实验前固定方法范围、优化信号、可变状态、真实载荷和可比边界；它**不是本仓库已经测得的结果表**。`reproduction_status` 明确区分“有官方代码”和“已在本仓库复现”，当前行不得据此推断精度或速度。

矩阵不填未经一手来源核验的精度、PPL、吞吐、GPU 小时或优胜结论。LiftQuant 的样本长度、epoch 与变换开销来自论文/README 协议；固定 commit 的 README flags 与两阶段 parser 并不一一对应，不能写成已验证的 executable protocol。仓库另有一层 compatibility-patched smoke，但没有 PPL/任务精度，不能进入方法排名。

## 分层

共 59 行：A=27、B=8、C=9、D=15；细分 A0=6、A1=21、B=8、C=9、D=15。

- **A0**：冻结稠密模型，data-free，无校准样本/激活统计，无反向传播。
- **A1**：冻结稠密模型，允许校准样本、激活统计或闭式二阶量，但无反向传播。
- **B**：冻结 dense pretrained base；允许为局部 calibration reconstruction、曲率统计执行 STE/backward/HVP，或优化最终可融合的量化/辅助状态，但不做全局任务恢复。
- **C**：使用全局 CE/KL 梯度优化、QAT、下游 PEFT/微调，或主结果流水线以任务损失恢复全模型行为。
- **D**：压缩对象或训练范式不一致，如 KV-cache、仅激活、原生低比特训练和结构/NAS 搜索。

## 严格直接比较规则

`yes_if_protocol_matched` 不是自动可比；至少要匹配模型/检查点、压缩对象、校准数据与 token 数、评测集、实际部署载荷和 kernel 可执行性。B 类还必须记录 backward/HVP/STE 次数、是否实际学习参数、更新变量、GPU-hours 与峰值显存。C 类可以作为允许全局训练的前沿或机制参照，但不能用于证明冻结 PTQ 的严格优势。D 类单列，不把其名义压缩率混进权重 PTQ 横轴。

同压缩率必须使用实际 artifact 字节，而不是名义 bit 或熵下界。统一计费形式为：

`actual_payload_bytes = codes + scales/zero_points + transforms + codebooks + masks/indices + low_rank + biases + scheme_metadata + padding + headers/alignment + uncompressed_exceptions`

只有在导出产物中真实折叠、且解码/执行不再需要独立参数时，才能把某个 correction/scale/transform 计为零额外字节。除载荷外，统一记录 calibration tokens、steps/epochs、是否 backprop、updated variables、GPU-hours、peak memory 与 kernel/backend。

## LiftQuant 必须拆成两行

- **Block correction（B）**：论文/README 写为 STE、4096 个 RedPajama 样本 × 2048 token、2 epochs；固定 commit parser 只有 `nsamples1/2` 与 `epochs1/2`，机械映射还会把每阶段 4096 减为 3968，语义尚未由可执行协议确认。更新 binary `Wq` 与 fused `T*=M T^-1`，目标为 block output MSE，不使用全局 CE。部署必须计 `1-bit lifted codes + scales + [fused T* OR (M + inverse-whitening factors)] + padding/alignment`；两种等价表示不能重复计费。
- **Optional E2E（C）**：4096 个样本 × 4096 token，1 epoch CE，继续更新连续量化参数。论文主结果采用 E2E；仓库将其标为可选，并推荐部署优先使用 block correction。

固定 commit `72b3875` 的未补丁 Qwen2.5 layer-0 smoke 在 Transformers 4.57 上因 `Catcher.attention_type` 缺失失败；一行透明 metadata 兼容补丁后，8-window、1+1 epoch、单层 smoke 以 exit 0 写出 artifact。它只证明 bounded control flow，不含 PPL/任务精度、全模型压缩或部署 payload。完整证据见 `results/liftquant_official_integration_20260714/`。

## 需要显式拆分的其他协议

- **SqueezeLLM 主 Fisher 路径（B）**：官方 from-scratch quantization 先在独立框架中计算目标模型 loss gradient-square checkpoint，再把它用于 sensitivity-weighted k-means。它不训练 dense 参数，但确实使用 backward，不能放入 no-backward A1。
- **SpinQuant optimized rotation（C）**：官方 `optimize_rotation.py` 冻结 dense base 后，在 WikiText2 上用 Hugging Face `Trainer.train()` 和全局 causal-LM loss 优化 R1/R2。按目标函数属于全局恢复 C，而不是局部 reconstruction B。
- **SliderQuant（B）**：默认路径在 128×2048-token 校准集上以 AdamW 优化 channel scales 与所有线性层的 rank-4 LoRA；默认 20 epochs，W2A16 为 60 epochs。两者在导出时吸收到权重。**SliderQuant+（B）** 另含不可吸收的 Hadamard 运行时变换，必须单列 transform、kernel、workspace 和 latency。
- **Q-Palette A0/A1**：data-free codec/search 属 A0；使用 QTIP proxy Hessian 和 validation perplexity loss 做 mixed-scheme selection 的 data-aware 路径属 A1。二者都不 retrain，但校准预算和数据泄漏边界不同。
- **YAQA（B）**：B 来自全模型 KL Hessian sketch 可能需要 backward/HVP；fixed-point/LDL rounding 本身不做梯度学习，Hessian 与 search state 不进入部署 payload。
- **ADMM-Q / HAS-VQ / SEPTQ（A1）**：ADMM 闭式交替、k-means 和逐列二阶补偿都是冻结模型上的 encoder-side 局部迭代，不等于 backward/STE 微调。最终只免计确实不随 artifact 部署的 Hessian、分解和搜索状态；HAS-VQ 的 codebook+sparse residual 与 SEPTQ 的保留值+support 必须计费。
- **HESTIA（C）**：离线 Hutch++ 只负责温度调度；随后在 10B token 上以 AdamW 优化全模型 causal-LM loss，明确属于 QAT，不得放进 frozen PTQ 排名。
- **MXFP 协议参考**：arXiv:2601.09555 是 benchmark/format study 而非本矩阵新增实测方法。MXFP 比较必须计 block value stream、共享 E8M0 scale 和 padding，并单列其 3/4 pre-scale；不得把 MXFP4 的 nominal 4 bit 与 INT4 artifact 直接等同。

## 方法索引

| 方法 | 变体 | lane | 严格 PTQ 直接比较 | 复现状态 |
|---|---|:---:|---|---|
| RTN | round-to-nearest baseline | A0 | yes_if_protocol_matched | native_baseline_implemented; measured_in_existing_exploratory_runs |
| GPTQ | one-shot second-order weight quantization | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| AWQ | activation-aware weight quantization | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SparseGPT | one-shot unstructured or semi-structured pruning | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| Wanda | activation-magnitude pruning | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SpQR | sparse-quantized representation | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SqueezeLLM | dense-and-sparse non-uniform quantization | B | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| LQER | low-rank quantization error reconstruction | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| QERA | quantization error reconstruction analysis | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| EoRA | eigen-oriented low-rank adaptation | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| QuIP# | incoherence processing plus vector quantization | C | mechanism_only | official_code_available; external_reproduction_pending |
| QTIP | trellis-coded quantization | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| AQLM | additive quantization | C | mechanism_only | official_code_available; external_reproduction_pending |
| OmniQuant | learnable weight/activation PTQ | B | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SpinQuant | learned rotations for quantization | C | mechanism_only | official_code_available; external_reproduction_pending |
| SliderQuant | default; fused channel scales and rank-4 LoRA | B | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SliderQuant | SliderQuant+ with runtime rotations | B | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| LiftQuant | block correction only | B | mechanism_only | official_code_audited; official_unpatched_qwen_smoke_failed; compatibility_patched_layer0_smoke_passed; full_external_reproduction_pending |
| LiftQuant | optional end-to-end correction | C | mechanism_only | official_code_audited; e2e_entrypoint_blocked_missing_datautils_block; external_reproduction_pending |
| D²Quant | DSQ plus DAC | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| ADMM-Q | joint Hessian-weighted discrete solver | A1 | yes_if_protocol_matched | no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending |
| HAS-VQ | Hessian-masked sparse vector quantization | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SEPTQ | static-global selective GPTQ-style PTQ | A1 | yes_if_protocol_matched | no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending |
| AAAC | two-codebook activation-aware adaptive coding | A1 | yes_if_protocol_matched | no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending |
| DAQ | Delta-Aware Quantization | A0 | mechanism_only | official_documentation_available; external_reproduction_pending |
| Q-Palette | data-free palette codec | A0 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| Q-Palette | data-aware Hessian/perplexity-guided palette codec | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SLiM | quantization + 2:4 sparsity + low-rank compensation | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| OBR | closed-form group compensation | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| HESTIA | Hessian-guided differentiable ternary QAT | C | mechanism_only | official_code_available; external_reproduction_pending |
| EfficientQAT | block-AP plus end-to-end quantization-parameter training | C | mechanism_only | official_code_available; external_reproduction_pending |
| LLM-QAT | data-free distillation QAT | C | mechanism_only | official_code_available; external_reproduction_pending |
| TurboQuant | KV-cache/vector quantization | D | no_scope_mismatch | no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending |
| SharQ | online activation sparse-dense FP4 decomposition with shared FP4 weights | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| Joint structural pruning + MPQ | architecture/search method | D | no_scope_mismatch | no_official_repo_found_as_of_2026-07-14; literature_only; external_reproduction_pending |
| SINQ | calibration-free second-axis scaling | A0 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| A-SINQ | calibration-assisted second-axis scaling | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| SRR | Structured Residual Reconstruction PTQ | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| ResComp | residual compensation PTQ | A1 | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| FOEM | first-order error-aware compensation | A1 | yes_if_protocol_matched | official_code_available; third_party_integration_available; external_reproduction_pending |
| YAQA | full-model KL adaptive rounding | B | yes_if_protocol_matched | official_code_available; external_reproduction_pending |
| Effective Interplay S+Q | max-scaled quantization plus magnitude sparsity | A0 | mechanism_only | primary_sources_verified; official_code_available; external_reproduction_pending |
| Hessian ECSQ | Hessian-weighted quantization plus Huffman/ECSQ | B | mechanism_only | primary_sources_verified; no_official_repo_found; literature_only; external_reproduction_pending |
| Optimal Formats | variable-length weight formats | A0 | yes_if_protocol_matched | primary_sources_verified; no_official_repo_found; external_reproduction_pending |
| Optimal Formats | Fisher layer-wise bit allocation | B | mechanism_only | primary_sources_verified; no_official_repo_found; external_reproduction_pending |
| ProjQ | orthogonal projection for quantization and low-rank adapters | C | mechanism_only | official_code_available; external_reproduction_pending |
| ParetoQ | low-bit QAT scaling-law study | C | mechanism_only | official_code_available; external_reproduction_pending |
| Q-VDiT | video-DiT weight/activation quantization with temporal distillation | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| S2Q-VDiT | salient calibration and sparse-token distillation for W/A quantization | D | no_scope_mismatch | official_repository_placeholder; code_not_released_as_of_2026-07-16; external_reproduction_pending |
| QuantSparse | joint model quantization and attention sparsification | D | no_scope_mismatch | official_repository_placeholder; code_not_released_as_of_2026-07-16; external_reproduction_pending |
| TeaCache | timestep-embedding-aware feature caching | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| Sparse VideoGen | dynamic spatial-temporal sparse attention | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| Sparse-vDiT | offline searched diagonal, multi-diagonal, and vertical-stripe attention | D | no_scope_mismatch | primary_sources_verified; no_official_repo_found; external_reproduction_pending |
| CacheQuant | joint diffusion caching and quantization | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| VMonarch | structured Monarch attention for video diffusion | D | no_scope_mismatch | primary_sources_verified; no_official_repo_found; external_reproduction_pending |
| MonarchRT | periodic structured, dynamic sparse, and dense-mixing attention | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| RoPeSLR | 3D-RoPE sparse semantic spikes plus low-rank background | D | no_scope_mismatch | primary_sources_verified; no_official_repo_found; external_reproduction_pending |
| BitNet b1.58 | native ternary model training | D | no_scope_mismatch | official_code_available; external_reproduction_pending |
| QuEST | native/QAT low-bit weights and activations | D | no_scope_mismatch | official_code_available; external_reproduction_pending |

## Multimodal and video-system supplement

The D-lane supplement records Q-VDiT, S2Q-VDiT, QuantSparse, TeaCache, Sparse VideoGen, Sparse-vDiT, CacheQuant, VMonarch, MonarchRT, and RoPeSLR. These rows cover video/image diffusion W/A quantization, cache state, runtime sparse attention, structured attention, or sparse-low-rank attention, so they are mechanism and stacking references rather than direct frozen LLM weight-PTQ controls.

Reported speed, storage, and quality numbers remain in the dedicated multimodal strategy document instead of this registry. They use different models, prompts, frame counts, denoising steps, hardware, kernels, sparsity definitions, and training budgets. Standalone speedups must never be multiplied; a joint stack requires one measured end-to-end run plus separate storage, runtime-state, peak-memory, attention-only, and quality accounting.

QuantSparse and S2Q-VDiT have official repository placeholders but no released implementation as of the verification date. Q-VDiT, TeaCache, Sparse VideoGen, CacheQuant, and MonarchRT have inspectable official code. Sparse-vDiT, VMonarch, and RoPeSLR are marked without a confirmed official repository in the audited primary sources.

## 来源与复现边界

每一行都给出 `primary_paper` 与 `official_repo` 字段；`not_found_in_primary_sources` 表示截至核验日未在一手来源中确认官方仓库，不等于证明仓库永远不存在。RTN 是定义性 baseline，因此使用 `not_applicable_baseline`。

截至核验日未确认官方仓库的方法：ADMM-Q、SEPTQ、AAAC、TurboQuant、Joint structural pruning + MPQ、Hessian ECSQ、Optimal Formats、Sparse-vDiT、VMonarch、RoPeSLR。

除本仓库已实现并用于既有 exploratory runs 的 RTN baseline 外，矩阵只把“官方代码可获得”记为可复现入口；外部方法在本仓库实际运行前不能标记为 measured/reproduced。完整逐行字段与来源见 `method_matrix.csv`。

## 重新生成与防漂移

```bash
python scripts/build_compression_method_matrix.py
python scripts/build_compression_method_matrix.py --check
pytest -q tests/test_compression_method_matrix.py
```
