# Residual-Stack 7B 对比实验整理

日期：2026-07-10

## 实验目标

本轮实验用于检查 residual-space stacked compression 在更接近 7B LLM 的设置下是否仍有价值：

```text
W ~= Q(W) + S_res + L_res
```

核心比较对象包括：

- `Q only`
- `Q+L`
- `Q+S`
- `Q+S+L`
- residual-stack selector
- sequential QSR
- fixed SPQ-like
- Hessian-guided SPQ
- DAM-like / L->Q factor-quant proxy

所有主要比较都使用同预算约束：`nominal_memory_ratio <= 0.258`。

## 覆盖范围

本轮实际完成的是 `Qwen/Qwen2-7B` 的层子集实验：

- attention-only：3 个 `o_proj` 层，来自 first/middle/last layer。
- attention+MLP：6 个模块，覆盖 layer 0 和 layer 14 的 `down_proj`、`up_proj`、`o_proj`。

未完成的部分：

- LLaMA/Mistral 7B 没有本地可直接复用缓存。
- 236 服务器访问 Hugging Face model info 时连接被 reset，因此没有继续下载 Mistral/LLaMA。
- zero-shot 只使用很小的 `zero_shot_backup:arc_easy,hellaswag` 子集，当前全部方法均为 0.75，不能用于区分方法优劣。

## 工程环境与路径

本轮实验和同步涉及三个位置，需要区分“实验运行目录”和“GitHub 同步目录”：

| 类型 | 位置 |
|---|---|
| 实验服务器 | SSH profile `236`，host `amax`，IP `172.25.5.236`，2x RTX 4090 24GB |
| 实验工程目录 | `/home/wangmeiqi/llm_spectral_dynamics` |
| 实验脚本路径 | `/home/wangmeiqi/llm_spectral_dynamics/scripts/run_pretrained_llm_orthogonality.py` |
| Qwen2-7B HF cache snapshot | `/home/wangmeiqi/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e` |
| 校准 / zero-shot backup root | `/home/wangmeiqi/dataset_backup` |
| 已确认可用 backup 数据 | `/home/wangmeiqi/dataset_backup/ai2_arc_easy`, `/home/wangmeiqi/dataset_backup/hellaswag`, `/home/wangmeiqi/dataset_backup/ai2_arc_challenge` |
| GitHub 同步服务器 | SSH profile `210`，host `hi-X640-G40`，IP `172.25.5.210` |
| GitHub clone 路径 | `/home/wangmeiqi/com_compression` |
| GitHub remote | `git@github.com:chaochao825/com_compression.git` |
| Windows 本地归档工程 | `E:\Codex_work\ssh_experiment\llm_spectral_dynamics` |

236 上的模型权重 snapshot 文件包括：

| 文件 | 说明 |
|---|---|
| `model-00001-of-00004.safetensors` | Qwen2-7B shard 1 |
| `model-00002-of-00004.safetensors` | Qwen2-7B shard 2 |
| `model-00003-of-00004.safetensors` | Qwen2-7B shard 3 |
| `model-00004-of-00004.safetensors` | Qwen2-7B shard 4 |
| `model.safetensors.index.json` | shard index |
| `config.json`, `generation_config.json` | model config |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | tokenizer files |

236 原始实验结果目录：

| 实验 | 远端路径 | 备注 |
|---|---|---|
| Qwen2-7B attention-only | `/home/wangmeiqi/llm_spectral_dynamics/results/residual_stack_validate_Qwen_Qwen2-7B_20260707_014041` | first/middle/last 的 3 个 `o_proj` |
| Qwen2-7B attention+MLP | `/home/wangmeiqi/llm_spectral_dynamics/results/residual_stack_validate_Qwen_Qwen2-7B_20260707_015218` | layer 0/14 的 `down_proj`, `up_proj`, `o_proj` |

GitHub 仓库内同步后的路径：

| 内容 | 仓库路径 |
|---|---|
| 中文总结报告 | `docs/residual_stack_7b_comparison_report_20260710.md` |
| 汇总对比结果 | `results/compare_7b_dam_residual_stack_20260707/` |
| Qwen2-7B attention-only 原始结果 | `results/residual_stack_validate_Qwen_Qwen2-7B_20260707_014041/` |
| Qwen2-7B attention+MLP 原始结果 | `results/residual_stack_validate_Qwen_Qwen2-7B_20260707_015218/` |
| 复现实验脚本 | `scripts/run_pretrained_llm_orthogonality.py` |

## 关键结果

PPL delta 是相对 dense baseline 的变化，负数表示该 smoke eval 子集上 PPL 更低。

| 设置 | dense PPL | 最优同预算方法 | Q+L delta | Q+S delta | Q+S+L delta | SPQ-like delta | Hessian-SPQ delta | DAM-grid proxy delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Qwen2-7B attention-only, 3 `o_proj` | 67.1155 | Q+L | -0.537 | +0.443 | +0.165 | +0.719 | +0.969 | +3.833 |
| Qwen2-7B attention+MLP, 6 modules | 45.5759 | Q+S+L | +0.029 | -0.276 | -0.281 | +0.091 | +0.289 | +4.323 |

对应的跨模型汇总：

| 设置 | 最优同预算方法 | Q+L delta | Q+S+L delta | DAM-grid proxy delta |
|---|---|---:|---:|---:|
| Pythia-70M | Sequential QSR | +10.071 | +18.192 | +35.188 |
| Pythia-160M | Q+S+L | +0.702 | -0.748 | +23.151 |
| Qwen2-7B attention-only | Q+L | -0.537 | +0.165 | +3.833 |
| Qwen2-7B attention+MLP | Q+S+L | +0.029 | -0.281 | +4.323 |

## 方法有效性分析

1. 不能用 Pythia-70M 否定整体方向。
   Pythia-70M 上 residual-stack 表现很差，但 Pythia-160M 和 Qwen2-7B attention+MLP 子集出现了同预算收益，说明小模型和少层 smoke 的外推风险很高。

2. 层类型很关键。
   Qwen2-7B 只看 `o_proj` 时，`Q+L` 最好；加入 `up_proj/down_proj` 后，`Q+S+L` 最好。这说明 MLP 残差中的稀疏可恢复结构对当前方法很重要。

3. residual-space 比 sequential QSR 更稳定。
   Qwen2-7B attention+MLP 中，sequential QSR 的 PPL delta 为 `+1.536`，而 residual `Q+S+L` 为 `-0.281`。这支持“先量化，再在量化残差中分解 S/L”的基本动机。

4. 当前 selector 仍然不够好。
   Qwen2-7B attention+MLP 中 greedy selector 选择了所有层的 `Q+S`，但最终 PPL 最低的是 `Q+S+L`。这说明 activation/Hessian proxy 与最终 PPL 仍有错位，orthogonality 更适合作为 filter/constraint，而不是最终 selector。

5. DAM-like proxy 不竞争，但不能作为反驳论文的证据。
   当前没有官方代码复现。仓库中的 DAM 行只是根据论文公式实现的 proxy：
   - `paper_lq_factor_quant_matched_budget`
   - `paper_dam_closed_matched_budget`
   - `paper_dam_activation_grid_matched_budget`

   这些 proxy 在 Qwen2-7B 两个子集上均明显弱于 residual-stack 和 SPQ-like，但只能说明该 proxy 不足，不能说明原论文方法无效。

## 当前结论

更合理的表述是：

- block-circulant / Monarch-like structured residual 方向在此前离线 matched-memory activation reconstruction 中没有支持。
- residual-space `Q + S_res + L_res` 在 7B attention+MLP 子集上出现了正信号。
- 这个正信号不是 SOTA 结论，因为 eval 子集很小、zero-shot 不敏感、模型 family 只覆盖 Qwen2-7B。
- 下一步应该优先扩大到更多 Qwen2-7B 层，并在有缓存或网络可用时补 Mistral/LLaMA 7B；同时改进 selector，使其能在 `Q+S` 和 `Q+S+L` 之间更稳定地做 PPL-aware 选择。

## 文件索引

- 汇总 CSV：`results/compare_7b_dam_residual_stack_20260707/strategy_comparison.csv`
- 汇总图：`results/compare_7b_dam_residual_stack_20260707/figures/ppl_delta_heatmap.png`
- Qwen2-7B attention-only 原始结果：`results/residual_stack_validate_Qwen_Qwen2-7B_20260707_014041/`
- Qwen2-7B attention+MLP 原始结果：`results/residual_stack_validate_Qwen_Qwen2-7B_20260707_015218/`
- 运行脚本：`scripts/run_pretrained_llm_orthogonality.py`

## 复现实验命令

Qwen2-7B attention+MLP 6 模块 smoke：

```bash
PYTHONPATH=src python scripts/run_pretrained_llm_orthogonality.py \
  --mode residual_stack_validate \
  --include-dam-comparison \
  --model Qwen/Qwen2-7B \
  --local-files-only \
  --device auto \
  --torch-dtype float16 \
  --module-types down_proj,o_proj,up_proj \
  --layers 0,14 \
  --max-modules 6 \
  --calib-limit 2 \
  --eval-limit 2 \
  --selector-activation-sample-rows 64 \
  --sequence-length 64 \
  --batch-size 1 \
  --text-source zero_shot_backup \
  --zero-shot-tasks arc_easy,hellaswag \
  --zero-shot-strategy-limit 4 \
  --residual-stack-zero-shot-limit 4 \
  --residual-stack-memory-targets 0.258 \
  --residual-stack-eval-budget 0.258 \
  --residual-stack-q-methods rtn \
  --residual-stack-s-methods wanda,magnitude \
  --residual-stack-r-methods svd \
  --residual-stack-splits 0.25,0.5,0.75 \
  --dam-alpha-grid 0.0,0.25,0.5,0.75,1.0 \
  --q-method rtn \
  --s-method wanda \
  --r-method svd \
  --svd-device cpu \
  --keep-fraction 0.8 \
  --rank-fraction 0.5 \
  --spq-s-method wanda \
  --spq-r-method svd
```

实现注意：7B MLP 候选构造需要 CPU-side baseline/candidate proxy，否则 24GB 4090 上容易在 Hessian/activation proxy 阶段 OOM。
