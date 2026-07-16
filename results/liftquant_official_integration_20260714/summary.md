# LiftQuant 官方仓库兼容性审计

> 核验日期：2026-07-14；固定 commit：`72b3875c770e4579639931fed89dc95e4067edac`。
> 本审计程序没有启动训练或量化；下文另行汇总主任务已经结束的 bounded smoke 证据。没有产生可用于方法排名的精度结果。

## 结论

当前固定 commit **尚不能进入外部复现实验排名**。Block correction 与 optional E2E 必须分开看：

- Block correction：`blocked_pending_upstream_or_local_adapter`。`main.py --help` 返回码为 `0`，但 README 的 `--epochs, --nsamples` 不存在于实际 parser。
- Optional E2E：`blocked_at_import`。`e2efinetune.py --help` 返回码为 `1`，在解析参数前即失败。
- Compatibility-patched block smoke：`control_flow_and_artifact_passed`；它只覆盖 Qwen2.5-3B 的 layer 0，不改变完整外部复现仍为 pending 的结论。
- 因此方法矩阵保持 `external_reproduction_pending`，同时分别记录 official code audit、compatibility-patched layer-0 smoke 与 E2E import blocker；不能填写 PPL、准确率、速度或优胜结论。

## 固定版本与工作树

- 期望 commit：`72b3875c770e4579639931fed89dc95e4067edac`；匹配：`True`。
- 整体工作树干净：`True`；源代码工作树干净：`True`。
- 本次静态证据来自干净副本；compatibility-patched smoke 在下节单列。

## Official 与 compatibility-patched smoke 的边界

干净 official commit 是兼容性基线；patched run 只是本地 adapter smoke，不得重新标成官方完整复现。

- **Official pinned source（未打补丁）**：exit=`1`，elapsed=`33.14` s，max RSS=`9004140` KiB；失败签名：`AttributeError: 'Catcher' object has no attribute 'attention_type'`。
- **兼容补丁**：`external_patches/liftquant_72b3875_qwen2_attention_type.patch`，SHA256=`47d5437744873f9d2b65074ebf4a07322f4a92d73ea6bf427ce5b2afc6f7d7a2`，`git apply --check` 返回码=`0`，patch mode 与 clean target 一致=`True`。
- **Pinned source + compatibility patch**：物理 GPU=`2`，exit=`0`，elapsed=`195.71` s，max RSS=`9000700` KiB，status=`COMPLETED`。
- 成功运行证据 SHA256：stdout=`ea1f8a3f8778d283ae2fb7181d2b06c80472e072b09d4eb21ac4ae651a3ab1d5`；stderr=`59daa2d797699a16574e0e8236300dfc36336c706550a5f0cc5cb9852f75a793`；time=`2ec2d343d69ad7689fd0fa725d1417c21eafab29a3732ead43bad8b5a72f2774`。
- Layer-0 artifact：`/home/wangmeiqi/codex_external_runs/liftquant_72b3875_block_smoke2_20260714/qmodels/Qwen2.5-3B/Qwen2.5-3B+20to8-layer0.pth`；bytes=`24566073`；SHA256=`5382f0f65da351df04ae2c84b028b2c3ad18370b720f68d1fc007f27ddefa5e1`。

成功 smoke 的 resolved scope 是 Qwen2.5-3B、`20to8`/`wbits=2`、WikiText2、`seqlen=128`、`nsamples1=8`、`nsamples2=8`、`epochs1=1`、`epochs2=1`、batch=2、`quant_start=0`、`quant_end=1`。两阶段复用由 `max(nsamples1, nsamples2)` 加载的 8-window 池；不能写成 16 个独立校准窗口。运行未设置 `--eval_ppl`，tasks 为空，所以 exit=0 只证明 layer-0 控制流和 artifact 写出。

该 smoke 使用本地兼容补丁，既不是原始 official commit 的成功，也不是完整模型 PPL/任务准确率、真实部署 payload 或同率方法比较。

## CLI 与 README 漂移

`main.py` 静态提取到 101 个 long flags，README block 命令含 26 个。未实现的是：

```text
--epochs --nsamples
```

建议的机械映射只是候选，不是已验证的论文协议：

- `--nsamples 4096` → `--nsamples1 4096 --nsamples2 4096`；
- `--epochs 2` → `--epochs1 2 --epochs2 2`。

代码把两组样本的最大值作为加载池；当每阶段样本数等于最大值时，又各减去 `1/32`。因此上述映射加载 4096 条，但每个启用阶段实际迭代 3968 条。必须分别记录“池大小”和“每阶段优化 token 数”，且需要上游确认 2 epochs 是每阶段 2 个，还是总计 2 个。

E2E 使用 `HfArgumentParser(..., return_remaining_strings=True)`，却没有消费 `extra_args`。即使补齐缺失模块，未知/过期参数仍可能被静默忽略，应改成非空即失败。

## 数据与模型路径

- `datautils_e2e.py:122`：`/data/shared_data/datasets`；210 上存在：`False`。
- `datautils_e2e.py:127`：`/data/shared_data/datasets`；210 上存在：`False`。
- `datautils_e2e.py:131`：`/data/shared_data/datasets`；210 上存在：`False`。
- `datautils.py:16`：`/mnt/bn/adsinfra-gpu-dev-hl/heliulu/datasets/redpajama_cache`；210 上存在：`False`。

这些路径没有由 README 的 `--cache_dir` 控制；首次加载 RedPajama 时仍会命中作者机器的绝对路径。Block 与 E2E 两条路径均需参数化并固定数据集 revision/split。

Qwen 缩小 smoke 候选缓存存在：`True`；snapshot revision：`None`；索引引用 shard 缺失：`0`；最低文件/header gate：`True`。这不是完整 shard 哈希或模型加载证明。

## 环境一致性

README 建议 Python `3.12`；审计环境 Python `3.13.5`。

| 依赖 | requirements | 已安装 | 判定 |
|---|---:|---:|---|
| transformers | 5.9.0 | 4.57.1 | version_mismatch |
| lm-eval | 0.4.11 | 0.4.11 | exact_match |
| torch | 2.6.0 | 2.6.0 | exact_match |
| bitblas | 0.1.0.post1 | None | missing |
| matplotlib | unpinned | 3.10.0 | present_unpinned |
| termcolor | unpinned | 3.3.0 | present_unpinned |
| tqdm | unpinned | 4.67.1 | present_unpinned |

仓库直接 import、但 `requirements.txt` 未声明：`accelerate`, `datasets`, `numpy`, `scipy`。

E2E 还在仓库内 import `datautils_block`，但固定 commit 没有该模块；因此 E2E `--help` 已构成确定性失败，不需要启动 GPU 才能发现。

## Findings

| 严重度 | 范围 | 证据 | 影响 |
|---|---|---|---|
| HIGH | block_correction | README uses ['--epochs', '--nsamples']; they are absent from both main.py AST and the successful runtime --help output. | The published block-correction command exits in argparse before model loading. |
| HIGH | block_and_e2e | datautils_e2e.py:122 -> /data/shared_data/datasets (missing); datautils_e2e.py:127 -> /data/shared_data/datasets (missing); datautils_e2e.py:131 -> /data/shared_data/datasets (missing); datautils.py:16 -> /mnt/bn/adsinfra-gpu-dev-hl/heliulu/datasets/redpajama_cache (missing) | README RedPajama calibration/fine-tuning cannot start on the audited 210 host. |
| HIGH | e2e_finetuning | ModuleNotFoundError: No module named 'datautils_block' | The optional E2E path cannot reach argument parsing, even for --help. |
| MEDIUM | environment | transformers declared=5.9.0 installed=4.57.1; bitblas declared=0.1.0.post1 installed=None | A help-capable environment is not evidence of a requirements-faithful training environment. |
| MEDIUM | environment | Repository imports not listed in requirements.txt: ['accelerate', 'datasets', 'numpy', 'scipy']. | A clean install can fail or depend on accidental transitive packages. |
| HIGH | e2e_finetuning | e2efinetune.py:20 imports datautils_block | The affected entrypoint is incomplete at the pinned commit. |
| MEDIUM | block_correction | Both stages iterate over 3968 samples after the in-code 1/32 reduction; the loaded calibration pool remains 4096. | Loaded-pool size and per-stage optimization-example count must not be reported as the same quantity. |
| MEDIUM | e2e_finetuning | HfArgumentParser returns remaining strings into extra_args, which is not consumed after assignment. | Misspelled or version-removed E2E flags can be silently ignored once imports are repaired. |
| HIGH | block_correction_qwen_smoke | AttributeError: 'Catcher' object has no attribute 'attention_type' | The unmodified pinned commit cannot complete the Qwen2.5 layer-0 smoke in the audited Transformers 4.57 environment. |
| MEDIUM | block_correction_qwen_smoke | patched layer-0 exit=0, elapsed=195.71s, scope={'model': 'Qwen2.5-3B', 'mapping': '20to8', 'wbits': '2', 'seqlen': '128', 'nsamples1': '8', 'nsamples2': '8', 'epochs1': '1', 'epochs2': '1', 'batch_size': '2', 'quant_start': '0', 'quant_end': '1', 'eval_ppl': False, 'tasks': ''} | This validates a bounded execution path and artifact write, not full-model PPL, task accuracy, or deployment bytes. |

## 后续命令模板（审计脚本未运行）

### 更小的 Qwen2.5-3B 零优化 smoke

该模板把两个 optimization epoch 都设为 0，且没有 endpoint metric；只能做更小的控制流排障，不能复现 LiftQuant 精度。实际执行时仍需要 CUDA。

```bash
PYTHONDONTWRITEBYTECODE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
/home/wangmeiqi/codex_envs/liftquant-72b3875/bin/python main.py \
  --model /home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --net Qwen2.5-3B-Instruct \
  --cache_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/cache \
  --output_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/log \
  --save_dir /home/wangmeiqi/codex_scratch/liftquant-smoke/qmodels \
  --wbits 2 --expc 20to8 --w_sym --abits 16 --kbits 16 --vbits 16 \
  --calib_dataset wikitext2 --seqlen 128 \
  --nsamples1 2 --nsamples2 2 --epochs1 0 --epochs2 0 --batch_size 1 \
  --quant_start 0 --quant_end 1 --training_trans --usefullfp --limit 1
```

### Block correction 映射后 full 模板

该模板仍被 RedPajama 绝对路径和两阶段 epoch 语义阻塞，不能直接当作复现命令。

```bash
CUDA_VISIBLE_DEVICES=0 /home/wangmeiqi/codex_envs/liftquant-72b3875/bin/python main.py \
  --model /home/spco/base-2-bitnet/.hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 --net Qwen2.5-3B-Instruct \
  --save_dir /path/to/qmodels --output_dir /path/to/log --cache_dir /path/to/cache \
  --eval_ppl --wbits 2 --expc 20to8 --w_sym \
  --abits 16 --kbits 16 --vbits 16 --true-sequential --act-order --use_fpinps \
  --Rres_init Hadamard --nsamples1 4096 --nsamples2 4096 \
  --epochs1 2 --epochs2 2 --batch_size 2 --calib_dataset redpajama \
  --usefullfp --training_trans --finetuning_weights --align 1 \
  --lscale_lr 5e-3 --lexw_lr 2e-2 --lw_lr 2e-5 --la_lr 2e-3 --lt_lr 2e-4
```

## 进入正式比较前的闸门

1. 上游确认并修复 README 的两组 flag 映射，记录 resolved args。
2. 参数化 RedPajama 路径，固定 revision、split、样本去重与实际 token 数。
3. 补齐 E2E 缺失模块，并令未知 `extra_args` 直接报错。
4. 用 Python 3.12 建立可满足的锁定环境，验证 Torch/CUDA/BitBLAS ABI；block 与 E2E 使用独立环境记录。
5. 先运行缩小 smoke，再单独排队正式 block correction；E2E 属于微调 lane，不与 frozen/no-backward PTQ 混排。
6. 正式结果必须记录 GPU-hours、峰值显存、训练/校准 tokens、随机种子、真实 artifact bytes、kernel 与相同 endpoint。

完整机器可读证据见 `audit.json`。
