from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import importlib.util
import itertools
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_spectral_dynamics.structured.data import load_model_and_tokenizer_from_config, load_texts_from_config, token_batches
from llm_spectral_dynamics.structured.evaluation import evaluate_zero_shot
from llm_spectral_dynamics.structured.orthogonality import (
    empirical_additivity_error,
    parameter_cosine,
    spearmanr,
    spectrum_summary,
)


EPS = 1e-12
PAIR_LABELS = (("q", "s"), ("q", "r"), ("s", "r"))
FALLBACK_TEXTS = [
    "Language models compress information through repeated linear transformations and nonlinear mixing.",
    "Calibration data gives a practical view of which approximation errors matter for model behavior.",
    "Structured compression changes weight-space geometry and can alter downstream predictions.",
    "A useful diagnostic should connect perturbation overlap with loss and task degradation.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: str | None) -> list[int]:
    if value is None or not str(value).strip():
        return []
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_float_csv(value: str | None, default: list[float]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def linear_layer_index(name: str) -> int | None:
    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.block\.(\d+)\.", r"\.blocks\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def module_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def layer_family(name: str) -> str:
    lower = name.lower()
    if ".self_attn." in lower or ".attention." in lower or ".attn." in lower:
        return "attention"
    if ".mlp." in lower or ".ffn." in lower or lower.endswith(".fc1") or lower.endswith(".fc2"):
        return "mlp"
    suffix = module_type(lower)
    if suffix in {"dense_h_to_4h", "dense_4h_to_h"}:
        return "mlp"
    return "other"


def spq_ops_for_layer(name: str) -> tuple[str, ...]:
    family = layer_family(name)
    if family == "attention":
        return ("r", "q")
    if family == "mlp":
        return ("s", "q")
    return ("q",)


def _choice_texts_from_zero_shot_backup(tasks: list[str], *, limit: int) -> tuple[list[str], list[dict[str, object]]]:
    from datasets import load_from_disk

    backup_root = Path(os.environ.get("LLM_SC_DATASET_BACKUP_ROOT", "~/dataset_backup")).expanduser()
    backup_names = {"arc_easy": "ai2_arc_easy", "arc_challenge": "ai2_arc_challenge", "hellaswag": "hellaswag"}
    task_texts: list[list[str]] = []
    metadata: list[dict[str, object]] = []
    for task in tasks:
        backup_name = backup_names.get(task)
        if not backup_name:
            continue
        path = backup_root / backup_name
        if not path.exists():
            continue
        saved = load_from_disk(str(path))
        dataset = saved["validation"] if hasattr(saved, "keys") and "validation" in saved else saved
        candidates: list[str] = []
        for row in list(dataset)[: max(limit, 1)]:
            if task.startswith("arc"):
                choices = row.get("choices", {})
                choice_text = " ".join(str(item) for item in choices.get("text", []))
                candidates.append(str(row.get("question", "")).strip() + " " + choice_text)
            elif task == "hellaswag":
                endings = " ".join(str(item) for item in row.get("endings", []))
                candidates.append(str(row.get("ctx", "")).strip() + " " + endings)
        cleaned = [text.strip() for text in candidates if text.strip()]
        task_texts.append(cleaned)
        metadata.append(
            {
                "task": task,
                "backup_path": str(path),
                "split": "validation",
                "rows_available": len(dataset),
                "rows_loaded_from_backup": len(cleaned),
                "rows_used_for_text_source": 0,
                "fingerprint": getattr(dataset, "_fingerprint", ""),
            }
        )
    texts: list[str] = []
    max_rows = max((len(items) for items in task_texts), default=0)
    for row_idx in range(max_rows):
        for task_idx, items in enumerate(task_texts):
            if row_idx >= len(items) or len(texts) >= max(limit, 1):
                continue
            texts.append(items[row_idx])
            metadata[task_idx]["rows_used_for_text_source"] = int(metadata[task_idx]["rows_used_for_text_source"]) + 1
        if len(texts) >= max(limit, 1):
            break
    return texts, metadata


def load_eval_texts(args: argparse.Namespace, *, limit: int) -> tuple[list[str], str, list[dict[str, object]]]:
    source = str(args.text_source)
    if source in {"dataset", "auto"}:
        try:
            return load_texts_from_config(args.data_cfg, limit=limit), "dataset:" + str(args.data_cfg.get("dataset")), [
                {
                    "source": "dataset",
                    "dataset": args.data_cfg.get("dataset"),
                    "subset": args.data_cfg.get("subset"),
                    "split": args.data_cfg.get("split"),
                    "backup_name": args.data_cfg.get("backup_name"),
                    "rows_requested": limit,
                }
            ]
        except Exception:
            if source == "dataset":
                raise
    if source in {"zero_shot_backup", "auto"}:
        texts, metadata = _choice_texts_from_zero_shot_backup(args.zero_shot_tasks, limit=max(limit, 1))
        if texts:
            for row in metadata:
                row["source"] = "zero_shot_backup"
                row["backup_root"] = str(Path(os.environ.get("LLM_SC_DATASET_BACKUP_ROOT", "~/dataset_backup")).expanduser())
            return texts, "zero_shot_backup:" + ",".join(args.zero_shot_tasks), metadata
        if source == "zero_shot_backup":
            raise RuntimeError("no zero-shot backup texts were available")
    needed = max(limit, 1)
    reps = (needed + len(FALLBACK_TEXTS) - 1) // len(FALLBACK_TEXTS)
    return (FALLBACK_TEXTS * reps)[:needed], "fallback_builtin", [{"source": "fallback_builtin", "rows_requested": needed}]


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def args_snapshot(args: argparse.Namespace) -> dict[str, object]:
    excluded = {"eval_texts", "calib_texts", "recovery_texts", "text_source_metadata", "data_cfg"}
    snapshot: dict[str, object] = {}
    for key, value in sorted(vars(args).items()):
        if key in excluded:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                snapshot[key] = list(value)
            else:
                snapshot[key] = [str(item) for item in value]
        elif isinstance(value, dict):
            snapshot[key] = value
        else:
            snapshot[key] = str(value)
    return snapshot


def split_text_windows(args: argparse.Namespace, texts: list[str]) -> None:
    if not bool(getattr(args, "disjoint_text_splits", False)):
        args.calib_texts = list(texts)
        args.eval_texts = list(texts)
        args.recovery_texts = list(texts)
        args.text_split_policy = "shared_text_pool"
        return
    calib_count = max(int(args.calib_limit) * int(args.texts_per_batch_window), 1)
    eval_count = max(int(args.eval_limit) * int(args.texts_per_batch_window), 1)
    recovery_batches = max(int(getattr(args, "spq_lora_train_limit", 0)), int(getattr(args, "spq_lora_steps", 0)), 1)
    recovery_count = max(recovery_batches * int(args.texts_per_batch_window), 1)
    needed = calib_count + eval_count + recovery_count
    if len(texts) < needed:
        reps = (needed + max(len(texts), 1) - 1) // max(len(texts), 1)
        texts = (texts * reps)[:needed]
    args.calib_texts = texts[:calib_count]
    args.eval_texts = texts[calib_count : calib_count + eval_count]
    args.recovery_texts = texts[calib_count + eval_count : calib_count + eval_count + recovery_count]
    args.text_split_policy = "disjoint_sequential_text_windows"


def evaluate_perplexity_on_texts(
    model: nn.Module,
    tokenizer: object,
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> dict[str, float | int]:
    nll_total = 0.0
    token_total = 0
    model.eval()
    with torch.no_grad():
        for batch in token_batches(tokenizer, texts, sequence_length=sequence_length, batch_size=batch_size, limit=eval_limit):
            batch = batch.to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            nll_total += float(loss.detach().cpu())
            token_total += int(labels.numel())
    mean_nll = nll_total / max(token_total, 1)
    return {"nll": mean_nll, "perplexity": float(math.exp(min(mean_nll, 50.0))), "tokens": token_total}


def select_layer_indices(all_indices: list[int], specs: list[str], explicit: list[int]) -> set[int]:
    if explicit:
        return set(explicit)
    if not all_indices:
        return set()
    unique = sorted(set(all_indices))
    selected: set[int] = set()
    for spec in specs:
        if spec == "first":
            selected.add(unique[0])
        elif spec == "middle":
            selected.add(unique[len(unique) // 2])
        elif spec == "last":
            selected.add(unique[-1])
        else:
            try:
                selected.add(int(spec))
            except ValueError:
                raise ValueError(f"unknown layer position spec: {spec}") from None
    return selected


def discover_target_linears(
    model: nn.Module,
    *,
    module_types: list[str],
    layer_positions: list[str],
    layers: list[int],
    max_modules: int,
) -> dict[str, nn.Linear]:
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    layer_indices = [idx for name, _ in candidates if (idx := linear_layer_index(name)) is not None]
    selected_layers = select_layer_indices(layer_indices, layer_positions, layers)
    wanted_types = set(module_types)
    out: list[tuple[str, nn.Linear]] = []
    for name, module in candidates:
        idx = linear_layer_index(name)
        if selected_layers and idx not in selected_layers:
            continue
        if wanted_types and module_type(name) not in wanted_types:
            continue
        if module.weight.ndim != 2:
            continue
        out.append((name, module))
    out.sort(key=lambda item: (linear_layer_index(item[0]) if linear_layer_index(item[0]) is not None else 10**9, item[0]))
    if max_modules > 0:
        out = out[:max_modules]
    if not out:
        available = sorted({module_type(name) for name, _ in candidates})
        raise RuntimeError(f"no target nn.Linear modules matched. Available module types include: {available[:40]}")
    return dict(out)


def short_layer_name(name: str) -> str:
    idx = linear_layer_index(name)
    suffix = module_type(name)
    return f"L{idx}:{suffix}" if idx is not None else suffix


def clone_weights(modules: dict[str, nn.Linear]) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().cpu().clone() for name, module in modules.items()}


def restore_weights(modules: dict[str, nn.Linear], baseline: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in modules.items():
            module.weight.copy_(baseline[name].to(device=module.weight.device, dtype=module.weight.dtype))


def apply_replacements(modules: dict[str, nn.Linear], replacements: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, weight in replacements.items():
            modules[name].weight.copy_(weight.to(device=modules[name].weight.device, dtype=modules[name].weight.dtype))


def get_parent_and_attr(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parent: nn.Module = root
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]  # type: ignore[index]
    return parent, parts[-1]


class LoRALinearRecovery(nn.Module):
    def __init__(self, base_layer: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / max(int(rank), 1)
        in_features = int(base_layer.in_features)
        out_features = int(base_layer.out_features)
        self.lora_a = nn.Parameter(torch.empty(self.rank, in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        hidden = F.linear(x.float(), self.lora_a)
        update = F.linear(hidden, self.lora_b) * self.scaling
        return base + update.to(dtype=base.dtype)


def train_lora_recovery(
    model: nn.Module,
    tokenizer: object,
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    steps: int,
    lr: float,
    train_limit: int,
) -> int:
    lora_params = [param for name, param in model.named_parameters() if "lora_" in name and param.requires_grad]
    if steps <= 0 or not lora_params:
        return 0
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    old_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.train()
    done = 0
    while done < steps:
        progressed = False
        for batch in token_batches(
            tokenizer,
            texts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            limit=max(train_limit, 1),
        ):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            done += 1
            progressed = True
            if done >= steps:
                break
        if not progressed:
            break
    if old_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = old_cache
    model.eval()
    return done


def evaluate_recovered_replacements(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
    *,
    recovery_texts: list[str],
    eval_texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
    zero_shot_tasks: list[str],
    zero_shot_limit: int,
    lora_steps: int,
    lora_rank: int,
    lora_alpha: float,
    lora_lr: float,
    lora_train_limit: int,
) -> tuple[dict[str, float | int], float, list[dict[str, object]], dict[str, object]]:
    restore_weights(modules, baseline_weights)
    apply_replacements(modules, replacements)

    original_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad_(False)

    wrapped: list[tuple[nn.Module, str, LoRALinearRecovery]] = []
    trained_steps = 0
    lora_params = 0
    try:
        for name in replacements:
            parent, attr = get_parent_and_attr(model, name)
            base = getattr(parent, attr)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"LoRA recovery currently expects nn.Linear at {name}, got {type(base)}")
            wrapper = LoRALinearRecovery(base, rank=lora_rank, alpha=lora_alpha).to(device)
            setattr(parent, attr, wrapper)
            wrapped.append((parent, attr, wrapper))

        trained_steps = train_lora_recovery(
            model,
            tokenizer,
            texts=recovery_texts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device=device,
            steps=lora_steps,
            lr=lora_lr,
            train_limit=lora_train_limit,
        )
        metrics = evaluate_current_model(
            model,
            tokenizer,
            texts=eval_texts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device=device,
            eval_limit=eval_limit,
        )
        zero_mean, zero_rows = evaluate_zero_shot_mean(model, tokenizer, tasks=zero_shot_tasks, limit=zero_shot_limit, device=device)
        lora_params = sum(param.numel() for _name, param in model.named_parameters() if "lora_" in _name)
    finally:
        for parent, attr, wrapper in reversed(wrapped):
            setattr(parent, attr, wrapper.base_layer)
        restore_weights(modules, baseline_weights)
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad_(original_requires_grad[name])

    return metrics, zero_mean, zero_rows, {"lora_steps_completed": trained_steps, "lora_params": int(lora_params)}


def collect_activation_covariances(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    calib_limit: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    accum = {name: torch.zeros(module.weight.shape[1], module.weight.shape[1], dtype=torch.float64) for name, module in modules.items()}
    counts = {name: 0 for name in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            x = inputs[0].detach()
            if x.ndim == 1:
                x2 = x.reshape(1, -1)
            else:
                x2 = x.reshape(-1, x.shape[-1])
            x2 = x2.float()
            accum[name] += x2.transpose(0, 1).matmul(x2).double().cpu()
            counts[name] += int(x2.shape[0])

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in token_batches(
            tokenizer,
            texts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            limit=max(calib_limit, 1),
        ):
            model(input_ids=batch.to(device))

    for handle in handles:
        handle.remove()

    covariances: dict[str, torch.Tensor] = {}
    for name, cov in accum.items():
        denom = max(counts[name], 1)
        cov = cov / float(denom)
        diag_mean = float(torch.diag(cov).mean().item()) if cov.numel() else 1.0
        ridge = max(diag_mean, EPS) * 1e-5
        covariances[name] = (cov + torch.eye(cov.shape[0], dtype=torch.float64) * ridge).float()
    return covariances, counts


def collect_activation_samples(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    calib_limit: int,
    max_rows: int,
) -> dict[str, torch.Tensor]:
    if max_rows <= 0:
        return {name: torch.empty(0, module.weight.shape[1], dtype=torch.float32) for name, module in modules.items()}
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in modules}
    counts = {name: 0 for name in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            remaining = max_rows - counts[name]
            if remaining <= 0 or not inputs:
                return
            x = inputs[0].detach()
            x2 = x.reshape(1, -1) if x.ndim == 1 else x.reshape(-1, x.shape[-1])
            take = min(remaining, int(x2.shape[0]))
            if take <= 0:
                return
            chunks[name].append(x2[:take].float().cpu())
            counts[name] += take

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for batch in token_batches(
            tokenizer,
            texts,
            sequence_length=sequence_length,
            batch_size=batch_size,
            limit=max(calib_limit, 1),
        ):
            model(input_ids=batch.to(device))
            if all(count >= max_rows for count in counts.values()):
                break

    for handle in handles:
        handle.remove()

    samples: dict[str, torch.Tensor] = {}
    for name, module in modules.items():
        if chunks[name]:
            samples[name] = torch.cat(chunks[name], dim=0)
        else:
            samples[name] = torch.empty(0, module.weight.shape[1], dtype=torch.float32)
    return samples


def symmetric_rtn_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = max(1, 2 ** (int(bits) - 1) - 1)
    work = weight.float()
    scale = torch.amax(torch.abs(work), dim=1, keepdim=True) / qmax
    scale = torch.clamp(scale, min=1e-12)
    return (torch.clamp(torch.round(work / scale), -qmax, qmax) * scale).to(dtype=weight.dtype)


def sinq_like_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Small non-uniform signed quantization proxy used only for sensitivity checks."""
    levels = max(2, 2 ** (int(bits) - 1))
    work = weight.float()
    scale = torch.amax(torch.abs(work), dim=1, keepdim=True).clamp_min(1e-12)
    codebook = torch.linspace(0.0, 1.0, levels, device=work.device, dtype=work.dtype).pow(2.0)
    normalized = torch.clamp(torch.abs(work) / scale, 0.0, 1.0)
    nearest = torch.argmin(torch.abs(normalized.unsqueeze(-1) - codebook.reshape(1, 1, -1)), dim=-1)
    quantized = torch.sign(work) * codebook[nearest] * scale
    return quantized.to(dtype=weight.dtype)


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def fwht_last_dim(matrix: torch.Tensor) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard transform over the last dimension."""
    n = int(matrix.shape[-1])
    if n & (n - 1):
        raise ValueError(f"FWHT requires a power-of-two dimension, got {n}")
    original_shape = tuple(matrix.shape)
    out = matrix.float().clone()
    out = out.reshape(-1, n)
    step = 1
    while step < n:
        out = out.reshape(-1, n // (step * 2), step * 2)
        left = out[:, :, :step].clone()
        right = out[:, :, step:].clone()
        out[:, :, :step] = left + right
        out[:, :, step:] = left - right
        out = out.reshape(-1, n)
        step *= 2
    return (out / math.sqrt(float(n))).reshape(original_shape)


def rotated_rtn_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Hadamard-basis RTN proxy: quantize W H, then de-rotate by H^T."""
    work = weight.float()
    cols = int(work.shape[1])
    padded_cols = next_power_of_two(cols)
    if padded_cols != cols:
        padded = torch.zeros((work.shape[0], padded_cols), device=work.device, dtype=work.dtype)
        padded[:, :cols] = work
    else:
        padded = work
    rotated = fwht_last_dim(padded)
    quantized = symmetric_rtn_quantize(rotated, bits).float()
    recovered = fwht_last_dim(quantized)[:, :cols]
    return recovered.to(device=weight.device, dtype=weight.dtype)


def magnitude_prune(weight: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    if float(keep_fraction) >= 1.0:
        return weight.clone()
    keep = max(1, min(weight.numel(), int(round(float(keep_fraction) * weight.numel()))))
    flat = torch.abs(weight.float()).reshape(-1)
    threshold = torch.topk(flat, keep, largest=True).values[-1]
    return torch.where(torch.abs(weight.float()) >= threshold, weight.float(), torch.zeros_like(weight.float())).to(dtype=weight.dtype)


def wanda_prune(weight: torch.Tensor, cov: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    if float(keep_fraction) >= 1.0:
        return weight.clone()
    keep = max(1, min(weight.numel(), int(round(float(keep_fraction) * weight.numel()))))
    diag = torch.clamp(torch.diag(cov).to(device=weight.device, dtype=torch.float32), min=0.0).sqrt()
    score = torch.abs(weight.float()) * diag.reshape(1, -1)
    threshold = torch.topk(score.reshape(-1), keep, largest=True).values[-1]
    return torch.where(score >= threshold, weight.float(), torch.zeros_like(weight.float())).to(dtype=weight.dtype)


def svd_low_rank(weight: torch.Tensor, rank_fraction: float) -> torch.Tensor:
    if float(rank_fraction) >= 1.0:
        return weight.clone()
    work = weight.float()
    rows, cols = work.shape
    rank = max(1, min(rows, cols, int(round(float(rank_fraction) * min(rows, cols)))))
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vh[:rank, :]).to(dtype=weight.dtype)


def whitened_svd_low_rank(weight: torch.Tensor, cov: torch.Tensor, rank_fraction: float, *, device: str) -> torch.Tensor:
    if float(rank_fraction) >= 1.0:
        return weight.clone()
    work = weight.float().to(device)
    h = cov.float().to(device)
    rows, cols = work.shape
    rank = max(1, min(rows, cols, int(round(float(rank_fraction) * min(rows, cols)))))
    evals, evecs = torch.linalg.eigh(h)
    floor = torch.clamp(evals.max() * 1e-5, min=torch.tensor(1e-8, device=device, dtype=evals.dtype))
    evals = torch.clamp(evals, min=floor)
    sqrt_h = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.transpose(0, 1)
    inv_sqrt_h = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.transpose(0, 1)
    transformed = work @ sqrt_h
    u, s, vh = torch.linalg.svd(transformed, full_matrices=False)
    low_transformed = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
    approx = low_transformed @ inv_sqrt_h
    return approx.to(device=weight.device, dtype=weight.dtype)


def compress_weight(
    weight: torch.Tensor,
    cov: torch.Tensor,
    op: str,
    method: str,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    svd_device: str,
) -> torch.Tensor:
    if op == "q" and method == "rtn":
        return symmetric_rtn_quantize(weight, bits)
    if op == "q" and method == "sinq_like":
        return sinq_like_quantize(weight, bits)
    if op == "q" and method == "rotated_rtn":
        return rotated_rtn_quantize(weight, bits)
    if op == "s" and method == "magnitude":
        return magnitude_prune(weight, keep_fraction)
    if op == "s" and method == "wanda":
        return wanda_prune(weight, cov, keep_fraction)
    if op == "r" and method == "svd":
        return svd_low_rank(weight, rank_fraction)
    if op == "r" and method == "whitened_svd":
        return whitened_svd_low_rank(weight, cov, rank_fraction, device=svd_device)
    raise ValueError(f"unsupported compression op/method: {op}/{method}")


def apply_order(
    weight: torch.Tensor,
    cov: torch.Tensor,
    order: Iterable[str],
    methods: dict[str, str],
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    svd_device: str,
) -> torch.Tensor:
    out = weight
    for op in order:
        out = compress_weight(
            out,
            cov,
            op,
            methods[op],
            bits=bits,
            keep_fraction=keep_fraction,
            rank_fraction=rank_fraction,
            svd_device=svd_device,
        )
    return out


def residual_compensated_weight(
    weight: torch.Tensor,
    cov: torch.Tensor,
    base_op: str,
    residual_op: str,
    methods: dict[str, str],
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    svd_device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = compress_weight(
        weight,
        cov,
        base_op,
        methods[base_op],
        bits=bits,
        keep_fraction=keep_fraction,
        rank_fraction=rank_fraction,
        svd_device=svd_device,
    )
    residual = weight - base
    component = compress_weight(
        residual,
        cov,
        residual_op,
        methods[residual_op],
        bits=bits,
        keep_fraction=keep_fraction,
        rank_fraction=rank_fraction,
        svd_device=svd_device,
    )
    return base + component, base, component


def hessian_inner_matrix(delta_a: torch.Tensor, delta_b: torch.Tensor, cov: torch.Tensor) -> float:
    a = delta_a.float()
    b = delta_b.float()
    h = cov.float().to(device=a.device)
    if a.shape != b.shape or a.shape[1] != h.shape[0]:
        raise ValueError(f"shape mismatch: delta_a={tuple(a.shape)}, delta_b={tuple(b.shape)}, cov={tuple(h.shape)}")
    return float(torch.sum((a @ h) * b).detach().cpu())


def hessian_cosine_matrix(delta_a: torch.Tensor, delta_b: torch.Tensor, cov: torch.Tensor) -> float:
    numerator = hessian_inner_matrix(delta_a, delta_b, cov)
    norm_a = math.sqrt(max(hessian_inner_matrix(delta_a, delta_a, cov), 0.0))
    norm_b = math.sqrt(max(hessian_inner_matrix(delta_b, delta_b, cov), 0.0))
    value = numerator / max(norm_a * norm_b, EPS)
    if not math.isfinite(value):
        return value
    return max(-1.0, min(1.0, float(value)))


def trace_only_cost(delta_a: torch.Tensor, delta_b: torch.Tensor, cov: torch.Tensor) -> float:
    scale = float(torch.trace(cov.float()).item()) / max(int(cov.shape[0]), 1)
    return 0.5 * scale * float(torch.sum(delta_a.float().pow(2)).item() + torch.sum(delta_b.float().pow(2)).item())


def activation_self_cost(delta: torch.Tensor, cov: torch.Tensor) -> float:
    return hessian_inner_matrix(delta, delta, cov)


def candidate_memory_ratio(order_name: str, weight: torch.Tensor, *, bits: int, keep_fraction: float, rank_fraction: float) -> float:
    if "_res" not in str(order_name):
        ops = "".join(ch for ch in str(order_name) if ch in {"q", "s", "r"})
        return nominal_memory_ratio_for_order(ops, weight, bits=bits, keep_fraction=keep_fraction, rank_fraction=rank_fraction)
    ratio = 0.0
    if "q" in str(order_name):
        ratio += float(bits) / 16.0
    if "s_res" in str(order_name):
        ratio += float(keep_fraction)
    if "r_res" in str(order_name):
        ratio += factorized_rank_ratio_for_weight(weight, rank_fraction)
    return ratio


def activation_candidate_metrics(weight: torch.Tensor, final: torch.Tensor, cov: torch.Tensor, samples: torch.Tensor | None) -> dict[str, float]:
    delta = final - weight
    hessian_cost = 0.5 * hessian_inner_matrix(delta, delta, cov)
    baseline_activation = max(hessian_inner_matrix(weight, weight, cov), EPS)
    normalized_hessian_cost = hessian_inner_matrix(delta, delta, cov) / baseline_activation
    if samples is None or samples.numel() == 0:
        return {
            "predicted_hessian_cost": hessian_cost,
            "normalized_hessian_cost": normalized_hessian_cost,
            "activation_reconstruction_error": normalized_hessian_cost,
            "worst_token_risk": normalized_hessian_cost,
            "token_risk_p95": normalized_hessian_cost,
        }
    x = samples.to(device=weight.device, dtype=torch.float32)
    err = x.matmul(delta.float().transpose(0, 1))
    ref = x.matmul(weight.float().transpose(0, 1))
    token_err = err.pow(2).sum(dim=1)
    token_ref = ref.pow(2).sum(dim=1).clamp_min(EPS)
    ratios = token_err / token_ref
    activation_error = float(token_err.sum().detach().cpu().item() / max(float(token_ref.sum().detach().cpu().item()), EPS))
    return {
        "predicted_hessian_cost": hessian_cost,
        "normalized_hessian_cost": normalized_hessian_cost,
        "activation_reconstruction_error": activation_error,
        "worst_token_risk": float(ratios.max().detach().cpu().item()),
        "token_risk_p95": float(torch.quantile(ratios.float(), 0.95).detach().cpu().item()),
    }


def evaluate_current_model(
    model: nn.Module,
    tokenizer: object,
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> dict[str, float | int]:
    return evaluate_perplexity_on_texts(
        model,
        tokenizer,
        texts=texts,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        eval_limit=eval_limit,
    )


def evaluate_zero_shot_mean(
    model: nn.Module,
    tokenizer: object,
    *,
    tasks: list[str],
    limit: int,
    device: str,
) -> tuple[float, list[dict[str, object]]]:
    if limit <= 0 or not tasks:
        return float("nan"), []
    rows = evaluate_zero_shot(model, tokenizer, tasks=tasks, limit=limit, device=device)
    ok = [float(row["accuracy"]) for row in rows if row.get("status") == "ok" and math.isfinite(float(row.get("accuracy", float("nan"))))]
    mean = sum(ok) / len(ok) if ok else float("nan")
    return mean, rows


def evaluate_replacements(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
    zero_shot_tasks: list[str],
    zero_shot_limit: int,
) -> tuple[dict[str, float | int], float, list[dict[str, object]]]:
    restore_weights(modules, baseline_weights)
    apply_replacements(modules, replacements)
    metrics = evaluate_current_model(
        model,
        tokenizer,
        texts=texts,
        sequence_length=sequence_length,
        batch_size=batch_size,
        device=device,
        eval_limit=eval_limit,
    )
    zero_mean, zero_rows = evaluate_zero_shot_mean(model, tokenizer, tasks=zero_shot_tasks, limit=zero_shot_limit, device=device)
    restore_weights(modules, baseline_weights)
    return metrics, zero_mean, zero_rows


def method_status_rows() -> list[dict[str, object]]:
    status = [
        {"component": "q", "method": "rtn", "status": "available", "reason": "native symmetric row-wise round-to-nearest"},
        {"component": "q", "method": "rotated_rtn", "status": "available", "reason": "native Hadamard-basis RTN proxy: quantize W H and de-rotate by H^T"},
        {
            "component": "q",
            "method": "gptq",
            "status": "unavailable" if importlib.util.find_spec("auto_gptq") is None else "installed_not_invoked",
            "reason": "auto-gptq package is not installed in the current environment" if importlib.util.find_spec("auto_gptq") is None else "external package detected; integration is intentionally separate from this native MVP script",
        },
        {
            "component": "q",
            "method": "awq",
            "status": "unavailable" if importlib.util.find_spec("awq") is None and importlib.util.find_spec("autoawq") is None else "installed_not_invoked",
            "reason": "AWQ/AutoAWQ package is not installed in the current environment" if importlib.util.find_spec("awq") is None and importlib.util.find_spec("autoawq") is None else "external package detected; integration is intentionally separate from this native MVP script",
        },
        {"component": "s", "method": "magnitude", "status": "available", "reason": "native top-k magnitude pruning"},
        {"component": "s", "method": "wanda", "status": "available", "reason": "native activation-aware |W| sqrt(diag(XtX)) pruning"},
        {"component": "s", "method": "sparsegpt", "status": "unavailable", "reason": "SparseGPT package/integration is not installed in the current environment"},
        {"component": "r", "method": "svd", "status": "available", "reason": "native truncated SVD"},
        {"component": "r", "method": "whitened_svd", "status": "available", "reason": "native activation-whitened SVD proxy in the spirit of SVD-LLM"},
        {"component": "r", "method": "svd_llm", "status": "proxy", "reason": "represented by native whitened_svd; official SVD-LLM code is not invoked"},
        {"component": "recipe", "method": "spq_like_rsq_no_lora", "status": "available", "reason": "native SPQ-like fixed recipe proxy: attention R+Q, MLP S+Q, no recovery"},
        {"component": "recipe", "method": "hessian_guided_spq_no_lora", "status": "available", "reason": "native SPQ-like layer-type prior with Hessian-guided order/method selection under the same nominal bit/keep/rank budget"},
        {"component": "recipe", "method": "spq_like_rsq_lora", "status": "available", "reason": "optional native LoRA recovery wrapper for the SPQ-like fixed recipe; enabled when --spq-lora-steps > 0"},
        {"component": "recipe", "method": "orthofilter_spq_refine", "status": "available", "reason": "optional SPQ-prior refinement: conditional-Hessian filtering followed by activation/worst-token/proxy scoring; enabled in fair benchmark by --include-orthofilter-spq-refine"},
        {"component": "recipe", "method": "fixed_qsr_rotated_q", "status": "available", "reason": "optional fixed QSR strategy using Hadamard rotated RTN for Q; enabled by --include-rotation-analysis"},
        {"component": "recipe", "method": "low_loss_triple_stack", "status": "available", "reason": "optional global Q+S+R candidate search targeting benchmark drop below the configured threshold"},
    ]
    return status


def build_base_deltas(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    methods: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    weights: dict[str, dict[str, torch.Tensor]] = {}
    deltas: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in baseline_weights.items():
        weights[name] = {}
        deltas[name] = {}
        for op in ("q", "s", "r"):
            compressed = compress_weight(
                weight,
                covariances[name],
                op,
                methods[op],
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            weights[name][op] = compressed
            deltas[name][op] = compressed - weight
    return weights, deltas


def max_over_median(values: torch.Tensor) -> float:
    flat = values.float().abs().reshape(-1)
    median = float(torch.median(flat).item())
    maximum = float(torch.max(flat).item())
    return maximum / max(median, EPS)


def rotation_quantization_rows(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        cols = int(weight.shape[1])
        padded_cols = next_power_of_two(cols)
        if padded_cols != cols:
            padded = torch.zeros((weight.shape[0], padded_cols), device=weight.device, dtype=torch.float32)
            padded[:, :cols] = weight.float()
        else:
            padded = weight.float()
        rotated_basis = fwht_last_dim(padded)
        base_norm = max(float(torch.linalg.vector_norm(weight.float()).item()), EPS)
        rtn = symmetric_rtn_quantize(weight, args.bits)
        rotated = rotated_rtn_quantize(weight, args.bits)
        for method, quantized, basis in [
            ("rtn", rtn, weight.float()),
            ("rotated_rtn", rotated, rotated_basis),
        ]:
            delta = quantized - weight
            rows.append(
                {
                    "layer": name,
                    "layer_short": short_layer_name(name),
                    "q_method": method,
                    "bits": args.bits,
                    "relative_weight_error": float(torch.linalg.vector_norm(delta.float()).item() / base_norm),
                    "hessian_self_cost": 0.5 * hessian_inner_matrix(delta, delta, covariances[name]),
                    "input_channel_max_over_median_before": max_over_median(weight.float().pow(2).sum(dim=0).sqrt()),
                    "input_channel_max_over_median_quant_basis": max_over_median(basis.float().pow(2).sum(dim=0).sqrt()),
                    "padded_input_dim": padded_cols,
                    "original_input_dim": cols,
                }
            )
    return rows


def plot_rotation_quantization(figures_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    methods = sorted({str(row["q_method"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for method in methods:
        group = [row for row in rows if str(row["q_method"]) == method]
        x = np.arange(len(group))
        axes[0].plot(x, [float(row["relative_weight_error"]) for row in group], marker="o", linewidth=1.2, label=method)
        axes[1].plot(x, [float(row["hessian_self_cost"]) for row in group], marker="o", linewidth=1.2, label=method)
    axes[0].set_ylabel("relative weight error")
    axes[1].set_ylabel("Hessian self cost")
    for ax in axes:
        ax.set_xlabel("selected module index")
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "rotation_quantization_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "rotation_quantization_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lossless_frontier(figures_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        return
    labels = [str(row["family"]) for row in summary_rows]
    memory = [float(row["nominal_memory_ratio"]) for row in summary_rows]
    drops = [float(row["benchmark_drop_percent"]) for row in summary_rows]
    colors = ["#2563eb" if bool(row["lossless_pass"]) else "#94a3b8" for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(labels, memory, color=colors)
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("nominal memory ratio")
    axes[0].set_title("Selected lossless frontier candidates")
    axes[0].grid(True, axis="y", alpha=0.22)
    for index, value in enumerate(memory):
        axes[0].text(index, value + 0.025, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(labels, drops, color=colors)
    threshold = float(summary_rows[0].get("lossless_threshold_percent", 1.0))
    axes[1].axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.0, label="threshold")
    for index, value in enumerate(drops):
        axes[1].text(index, value + 0.025 * max(threshold, 1.0), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("benchmark drop (%)")
    axes[1].set_title("Lossless criterion")
    axes[1].set_ylim(0.0, max(threshold * 1.15, max(drops + [0.0]) * 1.15, 0.1))
    axes[1].grid(True, axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    fig.savefig(figures_dir / "lossless_frontier_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "lossless_frontier_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fair_benchmark(figures_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    compressed = [row for row in rows if str(row["strategy"]) != "baseline"]
    if not compressed:
        return
    labels = [str(row["strategy"]) for row in compressed]
    ppl = [float(row["signed_ppl_delta_percent"]) for row in compressed]
    zero = [float(row["zero_shot_accuracy_delta"]) if math.isfinite(float(row["zero_shot_accuracy_delta"])) else float("nan") for row in compressed]
    memory = [float(row["nominal_memory_ratio"]) for row in compressed]
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.2), sharex=True)
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in ppl]
    axes[0].bar(labels, ppl, color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("signed PPL delta (%)")
    axes[0].set_title("Fair benchmark: signed metrics, no result-based selection")
    axes[0].grid(True, axis="y", alpha=0.22)
    colors = ["#dc2626" if math.isfinite(value) and value < 0 else "#2563eb" for value in zero]
    axes[1].bar(labels, zero, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("zero-shot acc. delta")
    axes[1].grid(True, axis="y", alpha=0.22)
    axes[2].bar(labels, memory, color="#0f766e")
    axes[2].axhline(1.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("nominal memory ratio")
    axes[2].grid(True, axis="y", alpha=0.22)
    axes[2].tick_params(axis="x", labelrotation=28)
    fig.tight_layout()
    fig.savefig(figures_dir / "fair_benchmark_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "fair_benchmark_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def hessian_heatmap_rows(
    deltas: dict[str, dict[str, torch.Tensor]],
    covariances: dict[str, torch.Tensor],
    figures_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matrix = np.zeros((len(deltas), len(PAIR_LABELS)), dtype=np.float64)
    names = list(deltas)
    for row_index, name in enumerate(names):
        for col_index, (left, right) in enumerate(PAIR_LABELS):
            rho = hessian_cosine_matrix(deltas[name][left], deltas[name][right], covariances[name])
            matrix[row_index, col_index] = rho
            rows.append(
                {
                    "layer": name,
                    "layer_short": short_layer_name(name),
                    "pair": left + right,
                    "rho_h": rho,
                    "abs_rho_h": abs(rho),
                }
            )

    fig_height = max(4.0, 0.45 * len(names) + 1.4)
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(PAIR_LABELS)), [left + "/" + right for left, right in PAIR_LABELS])
    ax.set_yticks(range(len(names)), [short_layer_name(name) for name in names])
    ax.set_xlabel("compression pair")
    ax.set_ylabel("layer/module")
    ax.set_title("Layer-wise Hessian cosine rho_H")
    fig.colorbar(im, ax=ax, label="rho_H")
    fig.tight_layout()
    fig.savefig(figures_dir / "hessian_cosine_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "hessian_cosine_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


def additivity_rows(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    deltas: dict[str, dict[str, torch.Tensor]],
    covariances: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if args.zero_shot_additivity_limit != args.zero_shot_strategy_limit:
        raise ValueError(
            "zero-shot degradation correlations require matched additivity and strategy limits; "
            f"got additivity={args.zero_shot_additivity_limit}, strategy={args.zero_shot_strategy_limit}"
        )
    rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        single_metrics: dict[str, dict[str, float | int]] = {}
        for op in ("q", "s", "r"):
            metrics, _, _ = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                {name: weight + deltas[name][op]},
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=[],
                zero_shot_limit=0,
            )
            single_metrics[op] = metrics

        for left, right in PAIR_LABELS:
            pair_weight = weight + deltas[name][left] + deltas[name][right]
            metrics, zero_mean, pair_zero_rows = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                {name: pair_weight},
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=args.zero_shot_tasks,
                zero_shot_limit=args.zero_shot_additivity_limit,
            )
            for zero_row in pair_zero_rows:
                zero_rows.append({"scope": "additivity_pair", "layer": name, "pair": left + right, **zero_row})
            delta_left = deltas[name][left]
            delta_right = deltas[name][right]
            rho = hessian_cosine_matrix(delta_left, delta_right, covariances[name])
            cross = hessian_inner_matrix(delta_left, delta_right, covariances[name])
            self_left = 0.5 * hessian_inner_matrix(delta_left, delta_left, covariances[name])
            self_right = 0.5 * hessian_inner_matrix(delta_right, delta_right, covariances[name])
            pred = self_left + self_right + cross
            add_error = empirical_additivity_error(
                float(baseline_metrics["nll"]),
                float(single_metrics[left]["nll"]),
                float(single_metrics[right]["nll"]),
                float(metrics["nll"]),
            )
            activation_reconstruction_sum = activation_self_cost(delta_left, covariances[name]) + activation_self_cost(delta_right, covariances[name])
            accuracy_degradation = float("nan") if not math.isfinite(baseline_zero_shot) or not math.isfinite(zero_mean) else baseline_zero_shot - zero_mean
            rows.append(
                {
                    "layer": name,
                    "layer_short": short_layer_name(name),
                    "pair": left + right,
                    "pair_evaluation": "linearized_delta_sum",
                    "rho_h": rho,
                    "abs_rho_h": abs(rho),
                    "hessian_cross_cost": cross,
                    "self_cost_left": self_left,
                    "self_cost_right": self_right,
                    "taylor_predicted_loss_delta": pred,
                    "frobenius_delta_sum": float(torch.linalg.vector_norm(delta_left.float()).item() + torch.linalg.vector_norm(delta_right.float()).item()),
                    "parameter_cosine": parameter_cosine(delta_left.float().cpu().numpy(), delta_right.float().cpu().numpy()),
                    "abs_parameter_cosine": abs(parameter_cosine(delta_left.float().cpu().numpy(), delta_right.float().cpu().numpy())),
                    "activation_reconstruction_sum": activation_reconstruction_sum,
                    "trace_only_cost": trace_only_cost(delta_left, delta_right, covariances[name]),
                    "loss_left": float(single_metrics[left]["nll"]),
                    "loss_right": float(single_metrics[right]["nll"]),
                    "loss_pair": float(metrics["nll"]),
                    "loss_degradation_pair": float(metrics["nll"]) - float(baseline_metrics["nll"]),
                    "ppl_pair": float(metrics["perplexity"]),
                    "ppl_degradation_pair": float(metrics["perplexity"]) - float(baseline_metrics["perplexity"]),
                    "zero_shot_pair": zero_mean,
                    "zero_shot_accuracy_degradation_pair": accuracy_degradation,
                    "additivity_error": add_error,
                    "abs_additivity_error": abs(add_error),
                }
            )
    return rows, zero_rows


def order_gap_rows(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    methods: dict[str, str],
    covariances: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    order_pairs = [(("r", "q"), ("q", "r")), (("r", "s"), ("s", "r"))]
    for name, weight in baseline_weights.items():
        base_spec = spectrum_summary(weight.float().cpu().numpy())
        for left_order, right_order in order_pairs:
            left_weight = apply_order(
                weight,
                covariances[name],
                left_order,
                methods,
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            right_weight = apply_order(
                weight,
                covariances[name],
                right_order,
                methods,
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            left_metrics, _, _ = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                {name: left_weight},
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=[],
                zero_shot_limit=0,
            )
            right_metrics, _, _ = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                {name: right_weight},
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=[],
                zero_shot_limit=0,
            )
            first_left = compress_weight(
                weight,
                covariances[name],
                left_order[0],
                methods[left_order[0]],
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            first_right = compress_weight(
                weight,
                covariances[name],
                right_order[0],
                methods[right_order[0]],
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            left_cond_rho = hessian_cosine_matrix(first_left - weight, left_weight - first_left, covariances[name])
            right_cond_rho = hessian_cosine_matrix(first_right - weight, right_weight - first_right, covariances[name])
            r_first_cond_rho = left_cond_rho if left_order[0] == "r" else right_cond_rho
            non_r_first_cond_rho = right_cond_rho if left_order[0] == "r" else left_cond_rho
            left_first_spec = spectrum_summary(first_left.float().cpu().numpy())
            right_first_spec = spectrum_summary(first_right.float().cpu().numpy())
            rows.append(
                {
                    "layer": name,
                    "layer_short": short_layer_name(name),
                    "left_order": "".join(left_order),
                    "right_order": "".join(right_order),
                    "loss_left": float(left_metrics["nll"]),
                    "loss_right": float(right_metrics["nll"]),
                    "loss_gap_left_minus_right": float(left_metrics["nll"]) - float(right_metrics["nll"]),
                    "abs_loss_gap": abs(float(left_metrics["nll"]) - float(right_metrics["nll"])),
                    "ppl_left": float(left_metrics["perplexity"]),
                    "ppl_right": float(right_metrics["perplexity"]),
                    "abs_ppl_gap": abs(float(left_metrics["perplexity"]) - float(right_metrics["perplexity"])),
                    "left_conditional_hessian_overlap": left_cond_rho,
                    "right_conditional_hessian_overlap": right_cond_rho,
                    "r_first_conditional_hessian_overlap": r_first_cond_rho,
                    "abs_r_first_conditional_hessian_overlap": abs(r_first_cond_rho),
                    "non_r_first_conditional_hessian_overlap": non_r_first_cond_rho,
                    "abs_non_r_first_conditional_hessian_overlap": abs(non_r_first_cond_rho),
                    "max_abs_conditional_hessian_overlap": max(abs(left_cond_rho), abs(right_cond_rho)),
                    "mean_abs_conditional_hessian_overlap": 0.5 * (abs(left_cond_rho) + abs(right_cond_rho)),
                    "base_spectral_entropy": base_spec["spectral_entropy"],
                    "left_first_spectral_entropy": left_first_spec["spectral_entropy"],
                    "right_first_spectral_entropy": right_first_spec["spectral_entropy"],
                    "abs_first_spectral_entropy_delta": abs(float(left_first_spec["spectral_entropy"]) - float(right_first_spec["spectral_entropy"])),
                    "abs_first_top1_energy_delta": abs(float(left_first_spec["top1_energy"]) - float(right_first_spec["top1_energy"])),
                    "abs_first_stable_rank_delta": abs(float(left_first_spec["stable_rank"]) - float(right_first_spec["stable_rank"])),
                    "final_weight_disagreement": float(torch.linalg.vector_norm((left_weight - right_weight).float()).item() / max(torch.linalg.vector_norm(weight.float()).item(), EPS)),
                    "loss_degradation_left": float(left_metrics["nll"]) - float(baseline_metrics["nll"]),
                    "loss_degradation_right": float(right_metrics["nll"]) - float(baseline_metrics["nll"]),
                }
            )
    return rows


def corr_row(family: str, x_name: str, y_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    x = [float(row.get(x_name, float("nan"))) for row in rows]
    y = [float(row.get(y_name, float("nan"))) for row in rows]
    rho, n = spearmanr(x, y)
    return {"family": family, "x": x_name, "y": y_name, "spearman_rho": rho, "n": n}


def correlation_rows(additivity: list[dict[str, object]], order_gap: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [
        corr_row("additivity", "abs_rho_h", "abs_additivity_error", additivity),
        corr_row("additivity_signed", "rho_h", "additivity_error", additivity),
        corr_row("real_ppl", "abs_rho_h", "ppl_degradation_pair", additivity),
        corr_row("real_zero_shot", "abs_rho_h", "zero_shot_accuracy_degradation_pair", additivity),
        corr_row("taylor", "taylor_predicted_loss_delta", "loss_degradation_pair", additivity),
        corr_row("frobenius_baseline", "frobenius_delta_sum", "loss_degradation_pair", additivity),
        corr_row("parameter_cosine_baseline", "abs_parameter_cosine", "abs_additivity_error", additivity),
        corr_row("activation_reconstruction_baseline", "activation_reconstruction_sum", "loss_degradation_pair", additivity),
        corr_row("trace_only_baseline", "trace_only_cost", "loss_degradation_pair", additivity),
    ]
    if order_gap:
        rows.extend(
            [
                corr_row("order_gap", "max_abs_conditional_hessian_overlap", "abs_loss_gap", order_gap),
                corr_row("order_gap_r_first_overlap", "abs_r_first_conditional_hessian_overlap", "abs_loss_gap", order_gap),
                corr_row("order_gap_non_r_first_overlap", "abs_non_r_first_conditional_hessian_overlap", "abs_loss_gap", order_gap),
                corr_row("spectrum_order_entropy", "abs_first_spectral_entropy_delta", "abs_loss_gap", order_gap),
                corr_row("spectrum_order_top1", "abs_first_top1_energy_delta", "abs_loss_gap", order_gap),
                corr_row("spectrum_order_stable_rank", "abs_first_stable_rank_delta", "abs_loss_gap", order_gap),
                corr_row("order_disagreement", "final_weight_disagreement", "abs_loss_gap", order_gap),
            ]
        )
    return rows


def choose_hessian_layerwise_budget(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    selection_family: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    replacements: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    orders = list(itertools.permutations(("q", "s", "r")))
    for name, weight in baseline_weights.items():
        best: tuple[float, str, dict[str, str], torch.Tensor] | None = None
        for q_method, s_method, r_method in itertools.product(q_methods, s_methods, r_methods):
            methods = {"q": q_method, "s": s_method, "r": r_method}
            for order in orders:
                final = apply_order(
                    weight,
                    covariances[name],
                    order,
                    methods,
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    svd_device=args.svd_device,
                )
                delta = final - weight
                cost = 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
                current = (cost, "".join(order), methods, final)
                if best is None or (current[0], current[1], str(current[2])) < (best[0], best[1], str(best[2])):
                    best = current
        assert best is not None
        cost, order_name, methods, final = best
        replacements[name] = final
        rows.append(
            {
                "selection_family": selection_family,
                "layer": name,
                "layer_short": short_layer_name(name),
                "selected_order": order_name,
                "selected_q_method": methods["q"],
                "selected_s_method": methods["s"],
                "selected_r_method": methods["r"],
                "selected_nominal_bits": bits,
                "selected_nominal_keep_fraction": keep_fraction,
                "selected_nominal_rank_fraction": rank_fraction,
                "selected_predicted_hessian_cost": cost,
            }
        )
    return replacements, rows


def choose_hessian_layerwise(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    return choose_hessian_layerwise_budget(
        baseline_weights,
        covariances,
        q_methods,
        s_methods,
        r_methods,
        args,
        bits=args.bits,
        keep_fraction=args.keep_fraction,
        rank_fraction=args.rank_fraction,
        selection_family="hessian_layerwise",
    )


def strategy_replacements(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    order: tuple[str, ...],
    methods: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return {
        name: apply_order(
            weight,
            covariances[name],
            order,
            methods,
            bits=args.bits,
            keep_fraction=args.keep_fraction,
            rank_fraction=args.rank_fraction,
            svd_device=args.svd_device,
        )
        for name, weight in baseline_weights.items()
    }


def benchmark_drop_fraction(
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    metrics: dict[str, float | int],
    zero_mean: float,
    requested_metric: str,
) -> tuple[float, str]:
    metric = requested_metric
    if metric == "zero_shot" and (not math.isfinite(baseline_zero_shot) or not math.isfinite(zero_mean)):
        metric = "ppl"
    if metric == "zero_shot":
        drop = max(float(baseline_zero_shot) - float(zero_mean), 0.0)
        return drop / max(abs(float(baseline_zero_shot)), EPS), metric
    if metric == "loss":
        drop = max(float(metrics["nll"]) - float(baseline_metrics["nll"]), 0.0)
        return drop / max(abs(float(baseline_metrics["nll"])), EPS), metric
    drop = max(float(metrics["perplexity"]) - float(baseline_metrics["perplexity"]), 0.0)
    return drop / max(abs(float(baseline_metrics["perplexity"])), EPS), "ppl"


def parse_order_candidates(value: str | None) -> list[tuple[str, ...]]:
    raw = parse_csv(value, ["qsr", "qrs", "sqr", "srq", "rqs", "rsq"])
    out: list[tuple[str, ...]] = []
    for item in raw:
        order = tuple(ch for ch in item.strip().lower() if ch in {"q", "s", "r"})
        if sorted(order) != ["q", "r", "s"]:
            raise ValueError(f"low-loss triple orders must contain q, s, and r exactly once, got {item!r}")
        if order not in out:
            out.append(order)
    return out


def low_loss_candidate_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    bits = parse_int_csv(args.low_loss_bits_list) or [8, 6, 4]
    keep_values = parse_float_csv(args.low_loss_keep_list, [0.995, 0.99, 0.98, 0.95])
    rank_values = parse_float_csv(args.low_loss_rank_list, [0.995, 0.99, 0.98, 0.95])
    q_methods = parse_csv(args.low_loss_q_methods, ["rotated_rtn", "rtn"])
    s_methods = parse_csv(args.low_loss_s_methods, ["wanda"])
    r_methods = parse_csv(args.low_loss_r_methods, ["whitened_svd", "svd"])
    orders = parse_order_candidates(args.low_loss_orders)
    specs: list[dict[str, object]] = []
    for bit, keep, rank, q_method, s_method, r_method, order in itertools.product(bits, keep_values, rank_values, q_methods, s_methods, r_methods, orders):
        specs.append(
            {
                "bits": int(bit),
                "keep_fraction": float(keep),
                "rank_fraction": float(rank),
                "q_method": str(q_method),
                "s_method": str(s_method),
                "r_method": str(r_method),
                "order": "".join(order),
                "order_tuple": order,
            }
        )
    q_priority = {"rotated_rtn": 0, "rtn": 1}
    r_priority = {"whitened_svd": 0, "svd": 1}
    specs.sort(
        key=lambda row: (
            -int(row["bits"]),
            -float(row["keep_fraction"]),
            -float(row["rank_fraction"]),
            q_priority.get(str(row["q_method"]), 10),
            r_priority.get(str(row["r_method"]), 10),
            str(row["order"]),
        )
    )
    return specs[: max(int(args.low_loss_max_candidates), 1)]


def low_loss_replacements_and_cost(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    spec: dict[str, object],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float]:
    order = tuple(spec["order_tuple"])  # type: ignore[arg-type]
    methods = {"q": str(spec["q_method"]), "s": str(spec["s_method"]), "r": str(spec["r_method"])}
    replacements: dict[str, torch.Tensor] = {}
    total_cost = 0.0
    for name, weight in baseline_weights.items():
        final = apply_order(
            weight,
            covariances[name],
            order,
            methods,
            bits=int(spec["bits"]),
            keep_fraction=float(spec["keep_fraction"]),
            rank_fraction=float(spec["rank_fraction"]),
            svd_device=args.svd_device,
        )
        replacements[name] = final
        delta = final - weight
        total_cost += 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
    return replacements, total_cost


def selected_param_count(baseline_weights: dict[str, torch.Tensor]) -> int:
    return int(sum(weight.numel() for weight in baseline_weights.values()))


def weighted_factorized_rank_ratio(baseline_weights: dict[str, torch.Tensor], rank_fraction: float) -> float:
    total = max(selected_param_count(baseline_weights), 1)
    factor_params = 0
    for weight in baseline_weights.values():
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        rank = max(1, min(rows, cols, int(round(float(rank_fraction) * min(rows, cols)))))
        factor_params += rank * (rows + cols)
    return float(factor_params) / float(total)


def factorized_rank_ratio_for_weight(weight: torch.Tensor, rank_fraction: float) -> float:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    rank = max(1, min(rows, cols, int(round(float(rank_fraction) * min(rows, cols)))))
    return float(rank * (rows + cols)) / float(max(weight.numel(), 1))


def nominal_memory_ratio_for_order(order: str, weight: torch.Tensor, *, bits: int, keep_fraction: float, rank_fraction: float) -> float:
    bit_ratio = float(bits) / 16.0 if "q" in str(order) else 1.0
    keep_ratio = float(keep_fraction) if "s" in str(order) else 1.0
    rank_ratio = factorized_rank_ratio_for_weight(weight, rank_fraction) if "r" in str(order) else 1.0
    return bit_ratio * keep_ratio * rank_ratio


def weighted_layerwise_nominal_memory_ratio(
    baseline_weights: dict[str, torch.Tensor],
    orders_by_layer: dict[str, str],
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
) -> float:
    total = max(selected_param_count(baseline_weights), 1)
    weighted = 0.0
    for name, weight in baseline_weights.items():
        order = orders_by_layer.get(name, "")
        weighted += float(weight.numel()) * nominal_memory_ratio_for_order(order, weight, bits=bits, keep_fraction=keep_fraction, rank_fraction=rank_fraction)
    return weighted / float(total)


def nominal_memory_ratio_for_spec(spec: dict[str, object], baseline_weights: dict[str, torch.Tensor]) -> float:
    bit_ratio = float(spec.get("bits", 16)) / 16.0 if "q" in str(spec["order"]) else 1.0
    keep_ratio = float(spec.get("keep_fraction", 1.0)) if "s" in str(spec["order"]) else 1.0
    rank_ratio = 1.0
    if "r" in str(spec["order"]):
        rank_ratio = weighted_factorized_rank_ratio(baseline_weights, float(spec.get("rank_fraction", 1.0)))
    return bit_ratio * keep_ratio * rank_ratio


def lossless_frontier_candidate_specs(args: argparse.Namespace, baseline_weights: dict[str, torch.Tensor]) -> list[dict[str, object]]:
    bits = parse_int_csv(args.frontier_bits_list) or [8, 6, 4, 3]
    keep_values = parse_float_csv(args.frontier_keep_list, [0.995, 0.99, 0.98, 0.95, 0.9, 0.8])
    rank_values = parse_float_csv(args.frontier_rank_list, [0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5])
    q_methods = parse_csv(args.frontier_q_methods, ["rotated_rtn", "rtn"])
    s_methods = parse_csv(args.frontier_s_methods, ["wanda", "magnitude"])
    r_methods = parse_csv(args.frontier_r_methods, ["whitened_svd", "svd"])
    orders = parse_order_candidates(args.frontier_orders)
    specs: list[dict[str, object]] = []
    for bit, q_method in itertools.product(bits, q_methods):
        specs.append(
            {
                "family": "q_only",
                "order": "q",
                "order_tuple": ("q",),
                "bits": int(bit),
                "keep_fraction": 1.0,
                "rank_fraction": 1.0,
                "q_method": str(q_method),
                "s_method": "",
                "r_method": "",
            }
        )
    for keep, s_method in itertools.product(keep_values, s_methods):
        specs.append(
            {
                "family": "s_only",
                "order": "s",
                "order_tuple": ("s",),
                "bits": 16,
                "keep_fraction": float(keep),
                "rank_fraction": 1.0,
                "q_method": "",
                "s_method": str(s_method),
                "r_method": "",
            }
        )
    for rank, r_method in itertools.product(rank_values, r_methods):
        specs.append(
            {
                "family": "r_only",
                "order": "r",
                "order_tuple": ("r",),
                "bits": 16,
                "keep_fraction": 1.0,
                "rank_fraction": float(rank),
                "q_method": "",
                "s_method": "",
                "r_method": str(r_method),
            }
        )
    triple_specs: list[dict[str, object]] = []
    for bit, keep, rank, q_method, s_method, r_method, order in itertools.product(bits, keep_values, rank_values, q_methods, s_methods, r_methods, orders):
        triple_specs.append(
            {
                "family": "qsr_stack",
                "order": "".join(order),
                "order_tuple": order,
                "bits": int(bit),
                "keep_fraction": float(keep),
                "rank_fraction": float(rank),
                "q_method": str(q_method),
                "s_method": str(s_method),
                "r_method": str(r_method),
            }
        )
    triple_full_grid_count = len(triple_specs)
    for spec in triple_specs:
        spec["nominal_memory_ratio"] = nominal_memory_ratio_for_spec(spec, baseline_weights)
    triple_specs.sort(
        key=lambda row: (
            float(row["nominal_memory_ratio"]),
            -int(row["bits"]),
            -float(row["keep_fraction"]),
            -float(row["rank_fraction"]),
            str(row["q_method"]),
            str(row["s_method"]),
            str(row["r_method"]),
            str(row["order"]),
        )
    )
    max_triples = max(int(args.frontier_max_triple_candidates), 1)
    selected_triples = triple_specs[:max_triples]
    for spec in specs:
        spec["frontier_candidate_scope"] = "single_method_full_grid"
        spec["frontier_triple_full_grid_count"] = triple_full_grid_count
        spec["frontier_triple_evaluated_count"] = len(selected_triples)
        spec["frontier_triple_candidate_cap"] = max_triples
    for spec in selected_triples:
        spec["frontier_candidate_scope"] = "qsr_lowest_nominal_memory_capped"
        spec["frontier_triple_full_grid_count"] = triple_full_grid_count
        spec["frontier_triple_evaluated_count"] = len(selected_triples)
        spec["frontier_triple_candidate_cap"] = max_triples
    return specs + selected_triples


def frontier_replacements_and_cost(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    spec: dict[str, object],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float]:
    order = tuple(spec["order_tuple"])  # type: ignore[arg-type]
    methods = {
        "q": str(spec.get("q_method") or args.q_method),
        "s": str(spec.get("s_method") or args.s_method),
        "r": str(spec.get("r_method") or args.r_method),
    }
    replacements: dict[str, torch.Tensor] = {}
    total_cost = 0.0
    for name, weight in baseline_weights.items():
        final = apply_order(
            weight,
            covariances[name],
            order,
            methods,
            bits=int(spec["bits"]),
            keep_fraction=float(spec["keep_fraction"]),
            rank_fraction=float(spec["rank_fraction"]),
            svd_device=args.svd_device,
        )
        replacements[name] = final
        delta = final - weight
        total_cost += 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
    return replacements, total_cost


def frontier_compression_criterion(row: dict[str, object]) -> float:
    family = str(row["family"])
    if family == "q_only":
        return float(row["nominal_bits"])
    if family == "s_only":
        return float(row["nominal_keep_fraction"])
    if family == "r_only":
        return float(row["nominal_rank_fraction"])
    return float(row["nominal_memory_ratio"])


def frontier_passing_row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        frontier_compression_criterion(row),
        float(row["benchmark_drop_fraction"]),
        float(row["predicted_hessian_cost"]),
        str(row["order"]),
        str(row["q_method"]),
        str(row["s_method"]),
        str(row["r_method"]),
    )


def frontier_failed_row_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        float(row["benchmark_drop_fraction"]),
        float(row["perplexity"]),
        float(row["predicted_hessian_cost"]),
        frontier_compression_criterion(row),
        str(row["order"]),
        str(row["q_method"]),
        str(row["s_method"]),
        str(row["r_method"]),
    )


def summarize_lossless_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for family in ("q_only", "s_only", "r_only", "qsr_stack"):
        group = [row for row in rows if str(row["family"]) == family]
        if not group:
            continue
        passing = [row for row in group if bool(row["lossless_pass"])]
        best = min(passing, key=frontier_passing_row_key) if passing else min(group, key=frontier_failed_row_key)
        selected = dict(best)
        selected["selection_rule"] = {
            "q_only": "lowest passing bit",
            "s_only": "lowest passing keep fraction",
            "r_only": "lowest passing rank fraction",
            "qsr_stack": "lowest passing nominal memory ratio",
        }[family]
        selected["frontier_status"] = "pass" if bool(best["lossless_pass"]) else "no_pass_best_drop"
        summary.append(selected)
    pass_single = [row for row in summary if str(row["family"]) in {"q_only", "s_only", "r_only"} and bool(row["lossless_pass"])]
    qsr = next((row for row in summary if str(row["family"]) == "qsr_stack"), None)
    if qsr is not None:
        best_single_ratio = min((float(row["nominal_memory_ratio"]) for row in pass_single), default=float("nan"))
        qsr_ratio = float(qsr["nominal_memory_ratio"])
        qsr["best_single_nominal_memory_ratio"] = best_single_ratio
        qsr["beats_best_single_nominal"] = bool(math.isfinite(best_single_ratio) and bool(qsr["lossless_pass"]) and qsr_ratio < best_single_ratio)
        for row in summary:
            if str(row["family"]) != "qsr_stack":
                row["best_single_nominal_memory_ratio"] = best_single_ratio
                row["beats_best_single_nominal"] = ""
    return summary


def evaluate_lossless_frontier(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    candidate_rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    specs = lossless_frontier_candidate_specs(args, baseline_weights)
    for index, spec in enumerate(specs, start=1):
        replacements, predicted_cost = frontier_replacements_and_cost(baseline_weights, covariances, spec, args)
        metrics, zero_mean, raw_zero_rows = evaluate_replacements(
            model,
            tokenizer,
            modules,
            baseline_weights,
            replacements,
            texts=args.eval_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            eval_limit=args.eval_limit,
            zero_shot_tasks=args.zero_shot_tasks,
            zero_shot_limit=args.zero_shot_strategy_limit,
        )
        drop, effective_metric = benchmark_drop_fraction(baseline_metrics, baseline_zero_shot, metrics, zero_mean, args.lossless_benchmark_metric)
        pass_threshold = drop < float(args.lossless_benchmark_drop_threshold)
        family = str(spec["family"])
        for zero_row in raw_zero_rows:
            zero_rows.append({"scope": "strategy", "strategy": "lossless_frontier", "family": family, "candidate_index": index, **zero_row})
        nominal_memory = nominal_memory_ratio_for_spec(spec, baseline_weights)
        row = {
            "candidate_index": index,
            "family": family,
            "strategy": "lossless_frontier",
            "frontier_candidate_scope": spec.get("frontier_candidate_scope", ""),
            "frontier_triple_full_grid_count": spec.get("frontier_triple_full_grid_count", ""),
            "frontier_triple_evaluated_count": spec.get("frontier_triple_evaluated_count", ""),
            "frontier_triple_candidate_cap": spec.get("frontier_triple_candidate_cap", ""),
            "order": spec["order"],
            "q_method": spec["q_method"],
            "s_method": spec["s_method"],
            "r_method": spec["r_method"],
            "nominal_bits": spec["bits"],
            "nominal_keep_fraction": spec["keep_fraction"],
            "nominal_rank_fraction": spec["rank_fraction"],
            "nominal_memory_ratio": nominal_memory,
            "nominal_memory_saving": 1.0 - nominal_memory,
            "selected_parameter_count": selected_param_count(baseline_weights),
            "factorized_rank_memory_ratio": weighted_factorized_rank_ratio(baseline_weights, float(spec["rank_fraction"])) if "r" in str(spec["order"]) else "",
            "predicted_hessian_cost": predicted_cost,
            "nll": float(metrics["nll"]),
            "perplexity": float(metrics["perplexity"]),
            "tokens": int(metrics["tokens"]),
            "ppl_degradation": float(metrics["perplexity"]) - float(baseline_metrics["perplexity"]),
            "loss_degradation": float(metrics["nll"]) - float(baseline_metrics["nll"]),
            "zero_shot_accuracy": zero_mean,
            "zero_shot_accuracy_degradation": float("nan") if not math.isfinite(zero_mean) or not math.isfinite(baseline_zero_shot) else baseline_zero_shot - zero_mean,
            "benchmark_metric_requested": args.lossless_benchmark_metric,
            "benchmark_metric_effective": effective_metric,
            "benchmark_drop_fraction": drop,
            "benchmark_drop_percent": 100.0 * drop,
            "lossless_threshold_percent": 100.0 * float(args.lossless_benchmark_drop_threshold),
            "lossless_pass": pass_threshold,
        }
        candidate_rows.append(row)
    return candidate_rows, summarize_lossless_frontier(candidate_rows), zero_rows


def fair_benchmark_specs() -> list[dict[str, object]]:
    return [
        {"strategy": "q_only_rtn_4bit", "family": "q_only", "order": "q", "q_method": "rtn", "s_method": "", "r_method": "", "bits": 4, "keep_fraction": 1.0, "rank_fraction": 1.0},
        {"strategy": "q_only_rotated_4bit", "family": "q_only", "order": "q", "q_method": "rotated_rtn", "s_method": "", "r_method": "", "bits": 4, "keep_fraction": 1.0, "rank_fraction": 1.0},
        {"strategy": "s_only_magnitude_keep0p8", "family": "s_only", "order": "s", "q_method": "", "s_method": "magnitude", "r_method": "", "bits": 16, "keep_fraction": 0.8, "rank_fraction": 1.0},
        {"strategy": "s_only_wanda_keep0p8", "family": "s_only", "order": "s", "q_method": "", "s_method": "wanda", "r_method": "", "bits": 16, "keep_fraction": 0.8, "rank_fraction": 1.0},
        {"strategy": "r_only_svd_rank0p5", "family": "r_only", "order": "r", "q_method": "", "s_method": "", "r_method": "svd", "bits": 16, "keep_fraction": 1.0, "rank_fraction": 0.5},
        {"strategy": "r_only_whitened_rank0p5", "family": "r_only", "order": "r", "q_method": "", "s_method": "", "r_method": "whitened_svd", "bits": 16, "keep_fraction": 1.0, "rank_fraction": 0.5},
        {"strategy": "qsr_naive_rtn_magnitude_svd", "family": "qsr_stack", "order": "qsr", "q_method": "rtn", "s_method": "magnitude", "r_method": "svd", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5},
        {"strategy": "qsr_rotated_wanda_whitened", "family": "qsr_stack", "order": "qsr", "q_method": "rotated_rtn", "s_method": "wanda", "r_method": "whitened_svd", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5},
        {"strategy": "rqs_rotated_wanda_whitened", "family": "qsr_stack", "order": "rqs", "q_method": "rotated_rtn", "s_method": "wanda", "r_method": "whitened_svd", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5},
        {"strategy": "hessian_guided_qsr_budget", "family": "hessian_guided_stack", "order": "layerwise", "q_method": "selected", "s_method": "selected", "r_method": "selected", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5},
    ]


def fair_benchmark_extended_recipe_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {"strategy": "slim_like_srq_proxy", "family": "slim_proxy", "order": "srq", "q_method": "rtn", "s_method": "wanda", "r_method": "whitened_svd", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5, "recovery": "none"},
        {"strategy": "spq_like_rsq_no_lora", "family": "spq_like", "order": "spq_layer_typed_rq_sq", "q_method": "rtn", "s_method": args.spq_s_method, "r_method": args.spq_r_method, "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5, "recovery": "none"},
        {"strategy": "hessian_guided_spq_no_lora", "family": "hessian_guided_spq", "order": "spq_layerwise", "q_method": "selected", "s_method": "selected", "r_method": "selected", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5, "recovery": "none"},
    ]
    if bool(getattr(args, "include_orthofilter_spq_refine", False)):
        specs.append(
            {
                "strategy": "orthofilter_spq_refine_no_lora",
                "family": "orthofilter_spq",
                "order": "spq_conditional_filter",
                "q_method": "selected",
                "s_method": "selected",
                "r_method": "selected",
                "bits": 4,
                "keep_fraction": 0.8,
                "rank_fraction": 0.5,
                "recovery": "none",
                "include_residual": False,
            }
        )
        if bool(getattr(args, "orthofilter_include_residual_candidates", False)):
            specs.append(
                {
                    "strategy": "orthofilter_spq_residual_refine_no_lora",
                    "family": "orthofilter_spq",
                    "order": "spq_conditional_filter_residual",
                    "q_method": "selected",
                    "s_method": "selected",
                    "r_method": "selected",
                    "bits": 4,
                    "keep_fraction": 0.8,
                    "rank_fraction": 0.5,
                    "recovery": "none",
                    "include_residual": True,
                }
            )
    if int(args.spq_lora_steps) > 0:
        specs.extend(
            [
                {"strategy": "spq_like_rsq_lora", "family": "spq_like", "order": "spq_layer_typed_rq_sq", "q_method": "rtn", "s_method": args.spq_s_method, "r_method": args.spq_r_method, "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5, "recovery": "lora"},
                {"strategy": "hessian_guided_spq_lora", "family": "hessian_guided_spq", "order": "spq_layerwise", "q_method": "selected", "s_method": "selected", "r_method": "selected", "bits": 4, "keep_fraction": 0.8, "rank_fraction": 0.5, "recovery": "lora"},
            ]
        )
        if bool(getattr(args, "include_orthofilter_spq_refine", False)):
            specs.append(
                {
                    "strategy": "orthofilter_spq_refine_lora",
                    "family": "orthofilter_spq",
                    "order": "spq_conditional_filter",
                    "q_method": "selected",
                    "s_method": "selected",
                    "r_method": "selected",
                    "bits": 4,
                    "keep_fraction": 0.8,
                    "rank_fraction": 0.5,
                    "recovery": "lora",
                    "include_residual": False,
                }
            )
            if bool(getattr(args, "orthofilter_include_residual_candidates", False)):
                specs.append(
                    {
                        "strategy": "orthofilter_spq_residual_refine_lora",
                        "family": "orthofilter_spq",
                        "order": "spq_conditional_filter_residual",
                        "q_method": "selected",
                        "s_method": "selected",
                        "r_method": "selected",
                        "bits": 4,
                        "keep_fraction": 0.8,
                        "rank_fraction": 0.5,
                        "recovery": "lora",
                        "include_residual": True,
                    }
                )
    return specs


def predicted_hessian_cost_for_replacements(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
) -> float:
    total = 0.0
    for name, final in replacements.items():
        delta = final - baseline_weights[name]
        total += 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
    return total


def fair_benchmark_replacements_and_cost(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    spec: dict[str, object],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], float]:
    order = tuple(str(spec["order"]))
    methods = {
        "q": str(spec.get("q_method") or args.q_method),
        "s": str(spec.get("s_method") or args.s_method),
        "r": str(spec.get("r_method") or args.r_method),
    }
    replacements: dict[str, torch.Tensor] = {}
    total_cost = 0.0
    for name, weight in baseline_weights.items():
        final = apply_order(
            weight,
            covariances[name],
            order,
            methods,
            bits=int(spec["bits"]),
            keep_fraction=float(spec["keep_fraction"]),
            rank_fraction=float(spec["rank_fraction"]),
            svd_device=args.svd_device,
        )
        replacements[name] = final
        delta = final - weight
        total_cost += 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
    return replacements, total_cost


def evaluate_fair_benchmark(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    activation_samples: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    dam_selection_rows: list[dict[str, object]] = []
    fair_zero_limit = int(args.fair_benchmark_zero_shot_limit) if args.fair_benchmark_zero_shot_limit is not None else int(args.zero_shot_strategy_limit)
    baseline_zero_shot, baseline_zero_rows = evaluate_zero_shot_mean(
        model,
        tokenizer,
        tasks=args.zero_shot_tasks,
        limit=fair_zero_limit,
        device=args.device,
    )
    baseline_by_task = {str(row.get("task")): row for row in baseline_zero_rows if row.get("status") == "ok"}
    rows.append(
        {
            "strategy": "baseline",
            "family": "baseline",
            "order": "none",
            "q_method": "",
            "s_method": "",
            "r_method": "",
            "nominal_bits": 16,
            "nominal_keep_fraction": 1.0,
            "nominal_rank_fraction": 1.0,
            "nominal_memory_ratio": 1.0,
            "nominal_memory_saving": 0.0,
            "predicted_hessian_cost": 0.0,
            "nll": float(baseline_metrics["nll"]),
            "perplexity": float(baseline_metrics["perplexity"]),
            "tokens": int(baseline_metrics["tokens"]),
            "signed_nll_delta": 0.0,
            "signed_nll_delta_percent": 0.0,
            "signed_ppl_delta": 0.0,
            "signed_ppl_delta_percent": 0.0,
            "clipped_ppl_drop_percent": 0.0,
            "zero_shot_accuracy": baseline_zero_shot,
            "zero_shot_accuracy_delta": 0.0 if math.isfinite(baseline_zero_shot) else float("nan"),
            "zero_shot_accuracy_drop": 0.0 if math.isfinite(baseline_zero_shot) else float("nan"),
            "recovery": "none",
            "lora_steps_completed": 0,
            "lora_params": 0,
            "benchmark_selection_rule": "fixed_configs_no_metric_selection",
        }
    )
    for row in baseline_zero_rows:
        zero_rows.append({"scope": "fair_benchmark", "strategy": "baseline", "accuracy_delta": 0.0, "accuracy_drop": 0.0, **row})

    specs = fair_benchmark_specs()
    if bool(getattr(args, "include_fair_extended_recipes", False)) or bool(getattr(args, "include_orthofilter_spq_refine", False)):
        specs.extend(fair_benchmark_extended_recipe_specs(args))
    for spec in specs:
        lora_info: dict[str, object] = {"lora_steps_completed": 0, "lora_params": 0}
        selected_rows: list[dict[str, object]] = []
        if str(spec["family"]) == "hessian_guided_stack":
            replacements, selected_rows = choose_hessian_layerwise_budget(
                baseline_weights,
                covariances,
                q_methods=parse_csv(args.fair_benchmark_guided_q_methods, ["rtn", "rotated_rtn"]),
                s_methods=parse_csv(args.fair_benchmark_guided_s_methods, ["magnitude", "wanda"]),
                r_methods=parse_csv(args.fair_benchmark_guided_r_methods, ["svd", "whitened_svd"]),
                args=args,
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
                selection_family="fair_hessian_guided_qsr_budget",
            )
            predicted_cost = sum(float(row["selected_predicted_hessian_cost"]) for row in selected_rows)
            for selected in selected_rows:
                selection_rows.append({"strategy": spec["strategy"], **selected})
        elif str(spec["family"]) == "spq_like":
            methods = {"q": str(spec["q_method"]), "s": str(spec["s_method"]), "r": str(spec["r_method"])}
            replacements = spq_like_replacements_budget(
                baseline_weights,
                covariances,
                methods,
                args,
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
            )
            predicted_cost = predicted_hessian_cost_for_replacements(baseline_weights, covariances, replacements)
        elif str(spec["family"]) == "hessian_guided_spq":
            replacements, selected_rows = choose_hessian_guided_spq_budget(
                baseline_weights,
                covariances,
                q_methods=parse_csv(args.spq_guided_q_methods, ["rtn"]),
                s_methods=parse_csv(args.spq_guided_s_methods, [args.spq_s_method]),
                r_methods=parse_csv(args.spq_guided_r_methods, [args.spq_r_method]),
                args=args,
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
            )
            predicted_cost = sum(float(row["selected_predicted_hessian_cost"]) for row in selected_rows)
            for selected in selected_rows:
                selected_copy = dict(selected)
                selected_copy["selection_family"] = "fair_hessian_guided_spq"
                selection_rows.append({"strategy": spec["strategy"], **selected_copy})
        elif str(spec["family"]) == "orthofilter_spq":
            replacements, selected_rows = choose_orthofilter_spq_budget(
                baseline_weights,
                covariances,
                activation_samples,
                q_methods=parse_csv(args.spq_guided_q_methods, ["rtn"]),
                s_methods=parse_csv(args.spq_guided_s_methods, [args.spq_s_method]),
                r_methods=parse_csv(args.spq_guided_r_methods, [args.spq_r_method]),
                args=args,
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
                include_residual=bool(spec.get("include_residual", False)),
            )
            predicted_cost = predicted_hessian_cost_for_replacements(baseline_weights, covariances, replacements)
            for selected in selected_rows:
                selected_copy = dict(selected)
                selected_copy["selection_family"] = "fair_orthofilter_spq"
                selection_rows.append({"strategy": spec["strategy"], **selected_copy})
        else:
            replacements, predicted_cost = fair_benchmark_replacements_and_cost(baseline_weights, covariances, spec, args)
        if str(spec.get("recovery", "none")) == "lora":
            metrics, zero_mean, raw_zero_rows, lora_info = evaluate_recovered_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                recovery_texts=args.recovery_texts,
                eval_texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=args.zero_shot_tasks,
                zero_shot_limit=fair_zero_limit,
                lora_steps=args.spq_lora_steps,
                lora_rank=args.spq_lora_rank,
                lora_alpha=args.spq_lora_alpha,
                lora_lr=args.spq_lora_lr,
                lora_train_limit=args.spq_lora_train_limit,
            )
        else:
            metrics, zero_mean, raw_zero_rows = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=args.zero_shot_tasks,
                zero_shot_limit=fair_zero_limit,
            )
        for zero_row in raw_zero_rows:
            base_task = baseline_by_task.get(str(zero_row.get("task")))
            base_acc = float(base_task["accuracy"]) if base_task is not None else float("nan")
            acc = float(zero_row.get("accuracy", float("nan")))
            zero_rows.append(
                {
                    "scope": "fair_benchmark",
                    "strategy": spec["strategy"],
                    "family": spec["family"],
                    "accuracy_delta": float("nan") if not math.isfinite(base_acc) or not math.isfinite(acc) else acc - base_acc,
                    "accuracy_drop": float("nan") if not math.isfinite(base_acc) or not math.isfinite(acc) else max(base_acc - acc, 0.0),
                    **zero_row,
                }
            )
        signed_nll_delta = float(metrics["nll"]) - float(baseline_metrics["nll"])
        signed_ppl_delta = float(metrics["perplexity"]) - float(baseline_metrics["perplexity"])
        if str(spec["family"]) == "spq_like":
            memory_ratio = weighted_layerwise_nominal_memory_ratio(
                baseline_weights,
                {name: "".join(spq_ops_for_layer(name)) for name in baseline_weights},
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
            )
        elif str(spec["family"]) == "hessian_guided_spq":
            memory_ratio = weighted_layerwise_nominal_memory_ratio(
                baseline_weights,
                {str(row["layer"]): str(row["selected_order"]) for row in selected_rows},
                bits=int(spec["bits"]),
                keep_fraction=float(spec["keep_fraction"]),
                rank_fraction=float(spec["rank_fraction"]),
            )
        elif str(spec["family"]) == "orthofilter_spq":
            selected_by_layer = {str(row["layer"]): str(row["candidate_order"]) for row in selected_rows if bool(row.get("selected", False))}
            total_params = max(selected_param_count(baseline_weights), 1)
            memory_ratio = sum(
                float(weight.numel())
                * candidate_memory_ratio(
                    selected_by_layer.get(name, ""),
                    weight,
                    bits=int(spec["bits"]),
                    keep_fraction=float(spec["keep_fraction"]),
                    rank_fraction=float(spec["rank_fraction"]),
                )
                for name, weight in baseline_weights.items()
            ) / float(total_params)
        else:
            spec_for_memory = {
                "family": spec["family"],
                "order": "qsr" if str(spec["family"]) == "hessian_guided_stack" else spec["order"],
                "bits": spec["bits"],
                "keep_fraction": spec["keep_fraction"],
                "rank_fraction": spec["rank_fraction"],
            }
            memory_ratio = nominal_memory_ratio_for_spec(spec_for_memory, baseline_weights)
        zero_delta = float("nan") if not math.isfinite(zero_mean) or not math.isfinite(baseline_zero_shot) else zero_mean - baseline_zero_shot
        if str(spec["family"]) in {"hessian_guided_stack", "hessian_guided_spq"}:
            selection_rule = "calibration_hessian_cost_fixed_budget_no_metric_selection"
        elif str(spec["family"]) == "orthofilter_spq":
            selection_rule = "calibration_conditional_hessian_filter_then_activation_worst_token_proxy_score"
        elif str(spec["family"]) in {"slim_proxy", "spq_like"}:
            selection_rule = "fixed_recipe_no_metric_selection"
        else:
            selection_rule = "fixed_configs_no_metric_selection"
        rows.append(
            {
                "strategy": spec["strategy"],
                "family": spec["family"],
                "order": spec["order"],
                "q_method": spec["q_method"],
                "s_method": spec["s_method"],
                "r_method": spec["r_method"],
                "nominal_bits": spec["bits"],
                "nominal_keep_fraction": spec["keep_fraction"],
                "nominal_rank_fraction": spec["rank_fraction"],
                "nominal_memory_ratio": memory_ratio,
                "nominal_memory_saving": 1.0 - memory_ratio,
                "predicted_hessian_cost": predicted_cost,
                "nll": float(metrics["nll"]),
                "perplexity": float(metrics["perplexity"]),
                "tokens": int(metrics["tokens"]),
                "signed_nll_delta": signed_nll_delta,
                "signed_nll_delta_percent": 100.0 * signed_nll_delta / max(abs(float(baseline_metrics["nll"])), EPS),
                "signed_ppl_delta": signed_ppl_delta,
                "signed_ppl_delta_percent": 100.0 * signed_ppl_delta / max(abs(float(baseline_metrics["perplexity"])), EPS),
                "clipped_ppl_drop_percent": 100.0 * max(signed_ppl_delta, 0.0) / max(abs(float(baseline_metrics["perplexity"])), EPS),
                "zero_shot_accuracy": zero_mean,
                "zero_shot_accuracy_delta": zero_delta,
                "zero_shot_accuracy_drop": float("nan") if not math.isfinite(zero_delta) else max(-zero_delta, 0.0),
                "recovery": spec.get("recovery", "none"),
                "lora_steps_completed": lora_info.get("lora_steps_completed", 0),
                "lora_params": lora_info.get("lora_params", 0),
                "benchmark_selection_rule": selection_rule,
            }
        )
    return rows, zero_rows, selection_rows


def evaluate_low_loss_triple(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, list[dict[str, object]], list[dict[str, object]]]:
    best_row: dict[str, object] | None = None
    candidate_rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    for index, spec in enumerate(low_loss_candidate_specs(args), start=1):
        replacements, predicted_cost = low_loss_replacements_and_cost(baseline_weights, covariances, spec, args)
        metrics, zero_mean, raw_zero_rows = evaluate_replacements(
            model,
            tokenizer,
            modules,
            baseline_weights,
            replacements,
            texts=args.eval_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            eval_limit=args.eval_limit,
            zero_shot_tasks=args.zero_shot_tasks,
            zero_shot_limit=args.zero_shot_strategy_limit,
        )
        drop, effective_metric = benchmark_drop_fraction(baseline_metrics, baseline_zero_shot, metrics, zero_mean, args.lossless_benchmark_metric)
        pass_threshold = drop < float(args.lossless_benchmark_drop_threshold)
        for zero_row in raw_zero_rows:
            zero_rows.append({"scope": "strategy", "strategy": "low_loss_triple_stack", "candidate_index": index, **zero_row})
        row = {
            "candidate_index": index,
            "strategy": "low_loss_triple_stack",
            "order": spec["order"],
            "q_method": spec["q_method"],
            "s_method": spec["s_method"],
            "r_method": spec["r_method"],
            "nominal_bits": spec["bits"],
            "nominal_keep_fraction": spec["keep_fraction"],
            "nominal_rank_fraction": spec["rank_fraction"],
            "predicted_hessian_cost": predicted_cost,
            "nll": float(metrics["nll"]),
            "perplexity": float(metrics["perplexity"]),
            "tokens": int(metrics["tokens"]),
            "ppl_degradation": float(metrics["perplexity"]) - float(baseline_metrics["perplexity"]),
            "loss_degradation": float(metrics["nll"]) - float(baseline_metrics["nll"]),
            "zero_shot_accuracy": zero_mean,
            "zero_shot_accuracy_degradation": float("nan") if not math.isfinite(zero_mean) or not math.isfinite(baseline_zero_shot) else baseline_zero_shot - zero_mean,
            "benchmark_metric_requested": args.lossless_benchmark_metric,
            "benchmark_metric_effective": effective_metric,
            "benchmark_drop_fraction": drop,
            "benchmark_drop_percent": 100.0 * drop,
            "lossless_threshold_percent": 100.0 * float(args.lossless_benchmark_drop_threshold),
            "lossless_pass": pass_threshold,
        }
        candidate_rows.append(row)
        if best_row is None:
            best_row = row
            continue
        current_key = (not bool(row["lossless_pass"]), float(row["benchmark_drop_fraction"]), float(row["perplexity"]), float(row["predicted_hessian_cost"]))
        best_key = (not bool(best_row["lossless_pass"]), float(best_row["benchmark_drop_fraction"]), float(best_row["perplexity"]), float(best_row["predicted_hessian_cost"]))
        if current_key < best_key:
            best_row = row
    if best_row is None:
        return None, candidate_rows, zero_rows
    best_strategy = {
        key: value
        for key, value in best_row.items()
        if key
        not in {
            "candidate_index",
            "predicted_hessian_cost",
            "benchmark_metric_requested",
            "benchmark_metric_effective",
            "benchmark_drop_fraction",
            "benchmark_drop_percent",
            "lossless_threshold_percent",
            "lossless_pass",
        }
    }
    best_strategy.update(
        {
            "strategy": "low_loss_triple_stack",
            "layer_policy": "global_low_loss_triple_search",
            "recovery": "none",
            "lora_steps_completed": 0,
            "lora_params": 0,
            "all_three_ops": True,
            "candidate_index": best_row["candidate_index"],
            "predicted_hessian_cost": best_row["predicted_hessian_cost"],
            "benchmark_metric_requested": best_row["benchmark_metric_requested"],
            "benchmark_metric_effective": best_row["benchmark_metric_effective"],
            "benchmark_drop_fraction": best_row["benchmark_drop_fraction"],
            "benchmark_drop_percent": best_row["benchmark_drop_percent"],
            "lossless_threshold_percent": best_row["lossless_threshold_percent"],
            "lossless_pass": best_row["lossless_pass"],
        }
    )
    return best_strategy, candidate_rows, zero_rows


def spq_like_replacements(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    methods: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return spq_like_replacements_budget(
        baseline_weights,
        covariances,
        methods,
        args,
        bits=args.bits,
        keep_fraction=args.keep_fraction,
        rank_fraction=args.rank_fraction,
    )


def spq_like_replacements_budget(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    methods: dict[str, str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
) -> dict[str, torch.Tensor]:
    replacements: dict[str, torch.Tensor] = {}
    for name, weight in baseline_weights.items():
        replacements[name] = apply_order(
            weight,
            covariances[name],
            spq_ops_for_layer(name),
            methods,
            bits=bits,
            keep_fraction=keep_fraction,
            rank_fraction=rank_fraction,
            svd_device=args.svd_device,
        )
    return replacements


def choose_hessian_guided_spq(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    return choose_hessian_guided_spq_budget(
        baseline_weights,
        covariances,
        q_methods,
        s_methods,
        r_methods,
        args,
        bits=args.bits,
        keep_fraction=args.keep_fraction,
        rank_fraction=args.rank_fraction,
    )


def choose_hessian_guided_spq_budget(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    replacements: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        ops = spq_ops_for_layer(name)
        q_space = q_methods if "q" in ops else [args.q_method]
        s_space = s_methods if "s" in ops else [args.s_method]
        r_space = r_methods if "r" in ops else [args.r_method]
        best: tuple[float, str, dict[str, str], torch.Tensor] | None = None
        for q_method, s_method, r_method in itertools.product(q_space, s_space, r_space):
            methods = {"q": q_method, "s": s_method, "r": r_method}
            for order in itertools.permutations(ops):
                final = apply_order(
                    weight,
                    covariances[name],
                    order,
                    methods,
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    svd_device=args.svd_device,
                )
                delta = final - weight
                cost = 0.5 * hessian_inner_matrix(delta, delta, covariances[name])
                current = (cost, "".join(order), methods, final)
                if best is None or (current[0], current[1], str(current[2])) < (best[0], best[1], str(best[2])):
                    best = current
        assert best is not None
        cost, order_name, methods, final = best
        replacements[name] = final
        rows.append(
            {
                "selection_family": "hessian_guided_spq",
                "layer": name,
                "layer_short": short_layer_name(name),
                "layer_family": layer_family(name),
                "selected_order": order_name,
                "selected_q_method": methods["q"] if "q" in ops else "",
                "selected_s_method": methods["s"] if "s" in ops else "",
                "selected_r_method": methods["r"] if "r" in ops else "",
                "selected_nominal_bits": bits,
                "selected_nominal_keep_fraction": keep_fraction,
                "selected_nominal_rank_fraction": rank_fraction,
                "selected_predicted_hessian_cost": cost,
            }
        )
    return replacements, rows


def orthofilter_candidate_score(row: dict[str, object], args: argparse.Namespace) -> float:
    return (
        float(args.orthofilter_hessian_weight) * float(row["normalized_hessian_cost"])
        + float(args.orthofilter_activation_weight) * float(row["activation_reconstruction_error"])
        + float(args.orthofilter_worst_token_weight) * float(row["worst_token_risk"])
        + float(args.orthofilter_conflict_weight) * float(row["positive_conditional_rho"])
        + float(args.orthofilter_memory_weight) * float(row["candidate_memory_ratio"])
        + float(args.orthofilter_zero_shot_proxy_weight) * float(row["zero_shot_proxy_risk"])
    )


def make_orthofilter_candidate(
    *,
    name: str,
    weight: torch.Tensor,
    cov: torch.Tensor,
    samples: torch.Tensor | None,
    order_name: str,
    final: torch.Tensor,
    first_weight: torch.Tensor | None,
    methods: dict[str, str],
    args: argparse.Namespace,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    candidate_kind: str,
) -> dict[str, object]:
    if first_weight is None:
        conditional_rho = 0.0
        conditional_cross = 0.0
    else:
        first_delta = first_weight - weight
        incremental_delta = final - first_weight
        conditional_rho = hessian_cosine_matrix(first_delta, incremental_delta, cov)
        conditional_cross = hessian_inner_matrix(first_delta, incremental_delta, cov)
    metrics = activation_candidate_metrics(weight, final, cov, samples)
    text_source_used = str(getattr(args, "text_source_used", "unknown"))
    has_activation_samples = samples is not None and samples.numel() > 0
    zero_shot_proxy_risk = float(metrics["token_risk_p95"]) if text_source_used.startswith("zero_shot_backup") and has_activation_samples else 0.0
    row: dict[str, object] = {
        "selection_family": "orthofilter_spq_refine",
        "layer": name,
        "layer_short": short_layer_name(name),
        "layer_family": layer_family(name),
        "candidate_kind": candidate_kind,
        "candidate_order": order_name,
        "candidate_q_method": methods.get("q", ""),
        "candidate_s_method": methods.get("s", ""),
        "candidate_r_method": methods.get("r", ""),
        "candidate_bits": bits,
        "candidate_keep_fraction": keep_fraction,
        "candidate_rank_fraction": rank_fraction,
        "candidate_memory_ratio": candidate_memory_ratio(order_name, weight, bits=bits, keep_fraction=keep_fraction, rank_fraction=rank_fraction),
        "conditional_rho": conditional_rho,
        "positive_conditional_rho": max(conditional_rho, 0.0),
        "abs_conditional_rho": abs(conditional_rho),
        "conditional_hessian_cross": conditional_cross,
        "filter_rho_threshold": float(args.orthofilter_rho_threshold),
        "filter_pass": max(conditional_rho, 0.0) <= float(args.orthofilter_rho_threshold),
        "zero_shot_proxy_source": "choice_text_token_risk_p95" if zero_shot_proxy_risk > 0.0 else "not_used_without_zero_shot_backup_text_or_samples",
        "zero_shot_proxy_risk": zero_shot_proxy_risk,
        "uses_residual_decomposition": "_res" in order_name,
        **metrics,
    }
    row["selector_score"] = orthofilter_candidate_score(row, args)
    row["_final"] = final
    return row


def orthofilter_spq_candidates_for_layer(
    name: str,
    weight: torch.Tensor,
    cov: torch.Tensor,
    samples: torch.Tensor | None,
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    include_residual: bool,
) -> list[dict[str, object]]:
    ops = spq_ops_for_layer(name)
    q_space = q_methods if "q" in ops else [args.q_method]
    s_space = s_methods if "s" in ops else [args.s_method]
    r_space = r_methods if "r" in ops else [args.r_method]
    candidates: list[dict[str, object]] = []
    for q_method, s_method, r_method in itertools.product(q_space, s_space, r_space):
        methods = {"q": q_method, "s": s_method, "r": r_method}
        for order in itertools.permutations(ops):
            final = apply_order(
                weight,
                cov,
                order,
                methods,
                bits=bits,
                keep_fraction=keep_fraction,
                rank_fraction=rank_fraction,
                svd_device=args.svd_device,
            )
            first = None
            if len(order) > 1:
                first = compress_weight(
                    weight,
                    cov,
                    order[0],
                    methods[order[0]],
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    svd_device=args.svd_device,
                )
            candidates.append(
                make_orthofilter_candidate(
                    name=name,
                    weight=weight,
                    cov=cov,
                    samples=samples,
                    order_name="".join(order),
                    final=final,
                    first_weight=first,
                    methods=methods,
                    args=args,
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    candidate_kind="sequential",
                )
            )
        if include_residual and ops == ("r", "q"):
            final, base, _component = residual_compensated_weight(
                weight,
                cov,
                "q",
                "r",
                methods,
                bits=bits,
                keep_fraction=keep_fraction,
                rank_fraction=rank_fraction,
                svd_device=args.svd_device,
            )
            candidates.append(
                make_orthofilter_candidate(
                    name=name,
                    weight=weight,
                    cov=cov,
                    samples=samples,
                    order_name="q+r_res",
                    final=final,
                    first_weight=base,
                    methods=methods,
                    args=args,
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    candidate_kind="residual_low_rank_after_q",
                )
            )
        if include_residual and ops == ("s", "q"):
            final, base, _component = residual_compensated_weight(
                weight,
                cov,
                "q",
                "s",
                methods,
                bits=bits,
                keep_fraction=keep_fraction,
                rank_fraction=rank_fraction,
                svd_device=args.svd_device,
            )
            candidates.append(
                make_orthofilter_candidate(
                    name=name,
                    weight=weight,
                    cov=cov,
                    samples=samples,
                    order_name="q+s_res",
                    final=final,
                    first_weight=base,
                    methods=methods,
                    args=args,
                    bits=bits,
                    keep_fraction=keep_fraction,
                    rank_fraction=rank_fraction,
                    candidate_kind="residual_sparse_after_q",
                )
            )
    return candidates


def select_orthofilter_candidate(candidates: list[dict[str, object]]) -> tuple[dict[str, object], bool]:
    feasible = [row for row in candidates if bool(row["filter_pass"])]
    pool = feasible if feasible else candidates
    best = min(
        pool,
        key=lambda row: (
            float(row["selector_score"]),
            float(row["positive_conditional_rho"]),
            float(row["candidate_memory_ratio"]),
            str(row["candidate_order"]),
            str(row["candidate_q_method"]),
            str(row["candidate_s_method"]),
            str(row["candidate_r_method"]),
        ),
    )
    return best, not bool(feasible)


def choose_orthofilter_spq_budget(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    activation_samples: dict[str, torch.Tensor],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
    include_residual: bool,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    replacements: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        candidates = orthofilter_spq_candidates_for_layer(
            name,
            weight,
            covariances[name],
            activation_samples.get(name),
            q_methods,
            s_methods,
            r_methods,
            args,
            bits=bits,
            keep_fraction=keep_fraction,
            rank_fraction=rank_fraction,
            include_residual=include_residual,
        )
        if not candidates:
            continue
        best, fallback_used = select_orthofilter_candidate(candidates)
        replacements[name] = best["_final"]  # type: ignore[assignment]
        for row in candidates:
            public_row = {key: value for key, value in row.items() if key != "_final"}
            public_row["selected"] = row is best
            public_row["filter_fallback_used"] = fallback_used
            public_row["selected_order"] = best["candidate_order"] if row is best else ""
            public_row["selected_q_method"] = best["candidate_q_method"] if row is best else ""
            public_row["selected_s_method"] = best["candidate_s_method"] if row is best else ""
            public_row["selected_r_method"] = best["candidate_r_method"] if row is best else ""
            public_row["selected_nominal_bits"] = bits if row is best else ""
            public_row["selected_nominal_keep_fraction"] = keep_fraction if row is best else ""
            public_row["selected_nominal_rank_fraction"] = rank_fraction if row is best else ""
            public_row["selected_predicted_hessian_cost"] = best["predicted_hessian_cost"] if row is best else ""
            rows.append(public_row)
    return replacements, rows


def lowrank_memory_ratio_for_rank(weight: torch.Tensor, rank: int) -> float:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    bounded = max(0, min(int(rank), min(rows, cols)))
    return float(bounded * (rows + cols)) / float(max(weight.numel(), 1))


def rank_from_residual_memory_ratio(weight: torch.Tensor, memory_ratio: float) -> int:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    unit = float(rows + cols) / float(max(weight.numel(), 1))
    if memory_ratio <= 0.0 or unit <= 0.0:
        return 0
    return max(0, min(min(rows, cols), int(math.floor(float(memory_ratio) / unit + 1e-12))))


def residual_sparse_project(weight: torch.Tensor, cov: torch.Tensor, keep_fraction: float, method: str) -> torch.Tensor:
    keep_fraction = float(max(0.0, min(1.0, keep_fraction)))
    if keep_fraction <= 0.0:
        return torch.zeros_like(weight)
    if keep_fraction >= 1.0:
        return weight.clone()
    keep = int(math.floor(keep_fraction * weight.numel() + 1e-12))
    if keep <= 0:
        return torch.zeros_like(weight)
    keep = min(keep, weight.numel())
    work = weight.float()
    if method == "magnitude":
        score = torch.abs(work)
    elif method == "wanda":
        diag = torch.clamp(torch.diag(cov).to(device=weight.device, dtype=torch.float32), min=0.0).sqrt()
        score = torch.abs(work) * diag.reshape(1, -1)
    else:
        raise ValueError(f"unsupported residual sparse method: {method}")
    flat_score = score.reshape(-1)
    top_indices = torch.topk(flat_score, keep, largest=True).indices
    mask = torch.zeros_like(flat_score, dtype=torch.bool)
    mask[top_indices] = True
    return torch.where(mask.reshape_as(work), work, torch.zeros_like(work)).to(dtype=weight.dtype)


def sparse_keep_fraction_under_budget(weight: torch.Tensor, budget_ratio: float) -> float:
    budget_ratio = float(max(0.0, min(1.0, budget_ratio)))
    if budget_ratio >= 1.0:
        return 1.0
    keep = int(math.floor(budget_ratio * weight.numel() + 1e-12))
    return float(max(0, min(keep, weight.numel()))) / float(max(weight.numel(), 1))


def lowrank_project_rank(weight: torch.Tensor, cov: torch.Tensor, rank: int, method: str, *, svd_device: str) -> torch.Tensor:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    rank = max(0, min(int(rank), min(rows, cols)))
    if rank <= 0:
        return torch.zeros_like(weight)
    if rank >= min(rows, cols):
        return weight.clone()
    if method == "svd":
        work = weight.float()
        u, s, vh = torch.linalg.svd(work, full_matrices=False)
        return ((u[:, :rank] * s[:rank]) @ vh[:rank, :]).to(dtype=weight.dtype)
    if method == "whitened_svd":
        device = svd_device
        work = weight.float().to(device)
        h = cov.float().to(device)
        evals, evecs = torch.linalg.eigh(h)
        floor = torch.clamp(evals.max() * 1e-5, min=torch.tensor(1e-8, device=device, dtype=evals.dtype))
        evals = torch.clamp(evals, min=floor)
        sqrt_h = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.transpose(0, 1)
        inv_sqrt_h = evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.transpose(0, 1)
        transformed = work @ sqrt_h
        u, s, vh = torch.linalg.svd(transformed, full_matrices=False)
        low_transformed = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        approx = low_transformed @ inv_sqrt_h
        return approx.to(device=weight.device, dtype=weight.dtype)
    raise ValueError(f"unsupported residual low-rank method: {method}")


def residual_spectrum_stats(matrix: torch.Tensor) -> dict[str, float]:
    work = matrix.float().detach().cpu()
    denom = float(torch.linalg.norm(work).pow(2).item())
    if denom <= EPS:
        return {"residual_stable_rank": 0.0, "residual_effective_rank": 0.0}
    vals = torch.linalg.svdvals(work)
    if vals.numel() == 0:
        return {"residual_stable_rank": 0.0, "residual_effective_rank": 0.0}
    stable = denom / max(float(vals[0].pow(2).item()), EPS)
    energy = vals.pow(2)
    probs = energy / energy.sum().clamp_min(EPS)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(EPS)))
    return {"residual_stable_rank": stable, "residual_effective_rank": float(torch.exp(entropy).item())}


def factor_quant_rank_for_memory(weight: torch.Tensor, target: float, bits: int) -> int:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    unit = (float(bits) / 16.0) * float(rows + cols) / float(max(weight.numel(), 1))
    if unit <= 0.0:
        return 0
    return max(0, min(min(rows, cols), int(math.floor(float(target) / unit + 1e-12))))


def factor_quant_memory_ratio_for_rank(weight: torch.Tensor, rank: int, bits: int) -> float:
    return (float(bits) / 16.0) * lowrank_memory_ratio_for_rank(weight, rank)


def dam_scale_from_singular_values(singular_values: torch.Tensor, rows: int, cols: int, alpha: float) -> torch.Tensor:
    sigma = torch.clamp(singular_values.float(), min=1e-12)
    shape_balance = (float(cols) / max(float(rows), 1.0)) ** 0.25
    return torch.clamp(sigma.pow(float(alpha)) * shape_balance, min=1e-6)


def factor_quantized_svd_weight(
    weight: torch.Tensor,
    cov: torch.Tensor,
    samples: torch.Tensor | None,
    *,
    rank: int,
    bits: int,
    scale_mode: str,
    svd_device: str,
    alpha_grid: list[float],
) -> tuple[torch.Tensor, dict[str, object]]:
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    rank = max(0, min(int(rank), min(rows, cols)))
    if rank <= 0:
        return torch.zeros_like(weight), {"dam_alpha": 0.0, "factor_rank": 0}
    work = weight.float().to(svd_device)
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    u = u[:, :rank]
    s = s[:rank]
    vh = vh[:rank, :]

    def reconstruct(alpha: float) -> torch.Tensor:
        if scale_mode == "plain_lq":
            scale = torch.ones_like(s)
        else:
            scale = dam_scale_from_singular_values(s, rows, cols, alpha).to(device=work.device, dtype=work.dtype)
        left = u * scale.reshape(1, -1)
        right = (s.reshape(-1, 1) / scale.reshape(-1, 1)) * vh
        q_left = symmetric_rtn_quantize(left, bits).float()
        q_right = symmetric_rtn_quantize(right, bits).float()
        return (q_left @ q_right).to(device=weight.device, dtype=weight.dtype)

    if scale_mode == "plain_lq":
        final = reconstruct(0.0)
        return final, {"dam_alpha": "", "factor_rank": rank, "dam_scale_mode": scale_mode}
    if scale_mode == "dam_closed":
        final = reconstruct(0.5)
        return final, {"dam_alpha": 0.5, "factor_rank": rank, "dam_scale_mode": scale_mode}
    if scale_mode != "dam_activation_grid":
        raise ValueError(f"unsupported DAM scale mode: {scale_mode}")

    best: tuple[float, float, torch.Tensor] | None = None
    for alpha in alpha_grid:
        final = reconstruct(float(alpha))
        metrics = activation_candidate_metrics(weight, final, cov, samples)
        score = float(metrics["activation_reconstruction_error"]) + 0.5 * float(metrics["token_risk_p95"]) + 0.2 * float(metrics["normalized_hessian_cost"])
        current = (score, float(alpha), final)
        if best is None or (current[0], current[1]) < (best[0], best[1]):
            best = current
    assert best is not None
    return best[2], {"dam_alpha": best[1], "factor_rank": rank, "dam_scale_mode": scale_mode}


def residual_stack_score(row: dict[str, object], args: argparse.Namespace) -> float:
    return (
        float(args.residual_stack_activation_weight) * float(row["activation_reconstruction_error"])
        + float(args.residual_stack_worst_token_weight) * float(row["token_risk_p95"])
        + float(args.residual_stack_hessian_weight) * float(row["normalized_hessian_cost"])
    )


def safe_hessian_cosine(delta_a: torch.Tensor, delta_b: torch.Tensor, cov: torch.Tensor) -> float:
    if float(torch.sum(delta_a.float().pow(2)).detach().cpu().item()) <= EPS:
        return 0.0
    if float(torch.sum(delta_b.float().pow(2)).detach().cpu().item()) <= EPS:
        return 0.0
    return hessian_cosine_matrix(delta_a, delta_b, cov)


def make_residual_stack_candidate(
    *,
    name: str,
    weight: torch.Tensor,
    cov: torch.Tensor,
    samples: torch.Tensor | None,
    q_weight: torch.Tensor,
    s_res: torch.Tensor,
    l_res: torch.Tensor,
    target_memory_ratio: float,
    q_method: str,
    s_method: str,
    r_method: str,
    bits: int,
    candidate_kind: str,
    stack_order: str,
    sparse_keep_fraction: float,
    lowrank_rank: int,
    residual_stats: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, object]:
    final = q_weight + s_res + l_res
    q_error = q_weight - weight
    q_plus_s_error = q_weight + s_res - weight
    q_plus_l_error = q_weight + l_res - weight
    rho_q_s = safe_hessian_cosine(q_error, s_res, cov)
    rho_q_l = safe_hessian_cosine(q_error, l_res, cov)
    rho_qs_l = safe_hessian_cosine(q_plus_s_error, l_res, cov)
    rho_ql_s = safe_hessian_cosine(q_plus_l_error, s_res, cov)
    rho_s_l = safe_hessian_cosine(s_res, l_res, cov)
    conditional_terms = {
        "q_only": [],
        "q_s": [rho_q_s],
        "q_l": [rho_q_l],
        "q_s_l": [rho_q_s, rho_qs_l, rho_s_l],
        "q_l_s": [rho_q_l, rho_ql_s, rho_s_l],
    }.get(candidate_kind, [rho_q_s, rho_q_l, rho_qs_l, rho_s_l])
    finite_terms = [value for value in conditional_terms if math.isfinite(float(value))]
    positive_rho = max([0.0] + [float(value) for value in finite_terms])
    q_memory = float(bits) / 16.0
    memory_ratio = q_memory + float(max(0.0, sparse_keep_fraction)) + lowrank_memory_ratio_for_rank(weight, lowrank_rank)
    residual = weight - q_weight
    residual_norm = float(torch.linalg.norm(residual.float()).detach().cpu().item())
    stack_residual_error = residual - s_res - l_res
    sparse_error = residual - s_res
    lowrank_error = residual - l_res
    metrics = activation_candidate_metrics(weight, final, cov, samples)
    weight_norm = float(torch.linalg.norm(weight.float()).detach().cpu().item())
    row: dict[str, object] = {
        "selection_family": "residual_stack_validate",
        "layer": name,
        "layer_short": short_layer_name(name),
        "layer_family": layer_family(name),
        "candidate_kind": candidate_kind,
        "stack_order": stack_order,
        "target_memory_ratio": float(target_memory_ratio),
        "candidate_memory_ratio": memory_ratio,
        "unused_memory_ratio": float(target_memory_ratio) - memory_ratio,
        "budget_feasible": memory_ratio <= float(target_memory_ratio) + 1e-9,
        "layer_parameter_count": int(weight.numel()),
        "q_method": q_method,
        "s_method": s_method,
        "r_method": r_method,
        "bits": int(bits),
        "sparse_keep_fraction": float(sparse_keep_fraction),
        "lowrank_rank": int(lowrank_rank),
        "lowrank_memory_ratio": lowrank_memory_ratio_for_rank(weight, lowrank_rank),
        "q_memory_ratio": q_memory,
        "residual_memory_ratio": memory_ratio - q_memory,
        "rho_q_error_s_res": rho_q_s,
        "rho_q_error_l_res": rho_q_l,
        "rho_q_plus_s_error_l_res": rho_qs_l,
        "rho_q_plus_l_error_s_res": rho_ql_s,
        "rho_s_res_l_res": rho_s_l,
        "positive_conditional_rho": positive_rho,
        "filter_rho_threshold": float(args.residual_stack_rho_threshold),
        "filter_pass": positive_rho <= float(args.residual_stack_rho_threshold),
        "weight_error": float(torch.linalg.norm((final - weight).float()).detach().cpu().item()) / max(weight_norm, EPS),
        "q_residual_norm": residual_norm,
        "stack_residual_error": float(torch.linalg.norm(stack_residual_error.float()).detach().cpu().item()) / max(residual_norm, EPS),
        "sparse_projection_error": float(torch.linalg.norm(sparse_error.float()).detach().cpu().item()) / max(residual_norm, EPS),
        "lowrank_projection_error": float(torch.linalg.norm(lowrank_error.float()).detach().cpu().item()) / max(residual_norm, EPS),
        "selected_by_fixed_kind": False,
        "selected_by_greedy": False,
        **residual_stats,
        **metrics,
    }
    row["selector_score"] = residual_stack_score(row, args)
    row["_final"] = final
    return row


def residual_stack_candidates_for_layer(
    name: str,
    weight: torch.Tensor,
    cov: torch.Tensor,
    samples: torch.Tensor | None,
    args: argparse.Namespace,
    *,
    targets: list[float],
    splits: list[float],
    q_methods: list[str],
    s_methods: list[str],
    r_methods: list[str],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for q_method in q_methods:
        q_weight = compress_weight(
            weight,
            cov,
            "q",
            q_method,
            bits=args.bits,
            keep_fraction=1.0,
            rank_fraction=1.0,
            svd_device=args.svd_device,
        )
        residual = weight - q_weight
        residual_stats = residual_spectrum_stats(residual)
        q_memory = float(args.bits) / 16.0
        zero = torch.zeros_like(weight)
        for target in targets:
            candidates.append(
                make_residual_stack_candidate(
                    name=name,
                    weight=weight,
                    cov=cov,
                    samples=samples,
                    q_weight=q_weight,
                    s_res=zero,
                    l_res=zero,
                    target_memory_ratio=target,
                    q_method=q_method,
                    s_method="",
                    r_method="",
                    bits=args.bits,
                    candidate_kind="q_only",
                    stack_order="q",
                    sparse_keep_fraction=0.0,
                    lowrank_rank=0,
                    residual_stats=residual_stats,
                    args=args,
                )
            )
            residual_budget = max(0.0, float(target) - q_memory)
            if residual_budget <= 0.0:
                continue
            for s_method in s_methods:
                keep = sparse_keep_fraction_under_budget(residual, residual_budget)
                s_res = residual_sparse_project(residual, cov, keep, s_method)
                candidates.append(
                    make_residual_stack_candidate(
                        name=name,
                        weight=weight,
                        cov=cov,
                        samples=samples,
                        q_weight=q_weight,
                        s_res=s_res,
                        l_res=zero,
                        target_memory_ratio=target,
                        q_method=q_method,
                        s_method=s_method,
                        r_method="",
                        bits=args.bits,
                        candidate_kind="q_s",
                        stack_order="q+s_res",
                        sparse_keep_fraction=keep,
                        lowrank_rank=0,
                        residual_stats=residual_stats,
                        args=args,
                    )
                )
            for r_method in r_methods:
                rank = rank_from_residual_memory_ratio(weight, residual_budget)
                l_res = lowrank_project_rank(residual, cov, rank, r_method, svd_device=args.svd_device)
                candidates.append(
                    make_residual_stack_candidate(
                        name=name,
                        weight=weight,
                        cov=cov,
                        samples=samples,
                        q_weight=q_weight,
                        s_res=zero,
                        l_res=l_res,
                        target_memory_ratio=target,
                        q_method=q_method,
                        s_method="",
                        r_method=r_method,
                        bits=args.bits,
                        candidate_kind="q_l",
                        stack_order="q+l_res",
                        sparse_keep_fraction=0.0,
                        lowrank_rank=rank,
                        residual_stats=residual_stats,
                        args=args,
                    )
                )
            for s_method, r_method, split in itertools.product(s_methods, r_methods, splits):
                split = float(max(0.0, min(1.0, split)))
                sparse_budget = residual_budget * split
                keep = sparse_keep_fraction_under_budget(residual, sparse_budget)
                lowrank_budget = max(0.0, residual_budget - keep)
                rank = rank_from_residual_memory_ratio(weight, lowrank_budget)
                s_res = residual_sparse_project(residual, cov, keep, s_method)
                l_res = lowrank_project_rank(residual - s_res, cov, rank, r_method, svd_device=args.svd_device)
                candidates.append(
                    make_residual_stack_candidate(
                        name=name,
                        weight=weight,
                        cov=cov,
                        samples=samples,
                        q_weight=q_weight,
                        s_res=s_res,
                        l_res=l_res,
                        target_memory_ratio=target,
                        q_method=q_method,
                        s_method=s_method,
                        r_method=r_method,
                        bits=args.bits,
                        candidate_kind="q_s_l",
                        stack_order="q+s_res+l_res",
                        sparse_keep_fraction=keep,
                        lowrank_rank=rank,
                        residual_stats=residual_stats,
                        args=args,
                    )
                )
                if bool(args.residual_stack_include_order_gap):
                    l_first = lowrank_project_rank(residual, cov, rank, r_method, svd_device=args.svd_device)
                    s_after_l = residual_sparse_project(residual - l_first, cov, keep, s_method)
                    candidates.append(
                        make_residual_stack_candidate(
                            name=name,
                            weight=weight,
                            cov=cov,
                            samples=samples,
                            q_weight=q_weight,
                            s_res=s_after_l,
                            l_res=l_first,
                            target_memory_ratio=target,
                            q_method=q_method,
                            s_method=s_method,
                            r_method=r_method,
                            bits=args.bits,
                            candidate_kind="q_l_s",
                            stack_order="q+l_res+s_res",
                            sparse_keep_fraction=keep,
                            lowrank_rank=rank,
                            residual_stats=residual_stats,
                            args=args,
                        )
                    )
    return candidates


def annotate_residual_stack_gains(rows: list[dict[str, object]]) -> None:
    q_only: dict[tuple[str, float], dict[str, object]] = {}
    for row in rows:
        if str(row["candidate_kind"]) != "q_only":
            continue
        key = (str(row["layer"]), float(row["target_memory_ratio"]))
        current = q_only.get(key)
        if current is None or float(row["activation_reconstruction_error"]) < float(current["activation_reconstruction_error"]):
            q_only[key] = row
    for row in rows:
        base = q_only.get((str(row["layer"]), float(row["target_memory_ratio"])))
        if base is None:
            row["activation_gain_vs_q_only"] = float("nan")
            row["hessian_gain_vs_q_only"] = float("nan")
            continue
        row["activation_gain_vs_q_only"] = float(base["activation_reconstruction_error"]) - float(row["activation_reconstruction_error"])
        row["hessian_gain_vs_q_only"] = float(base["normalized_hessian_cost"]) - float(row["normalized_hessian_cost"])


def generate_residual_stack_candidates(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    activation_samples: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    targets = parse_float_csv(args.residual_stack_memory_targets, [0.258, 0.300, 0.350])
    splits = parse_float_csv(args.residual_stack_splits, [0.25, 0.50, 0.75])
    q_methods = parse_csv(args.residual_stack_q_methods, ["rtn"])
    s_methods = parse_csv(args.residual_stack_s_methods, ["wanda", "magnitude"])
    r_methods = parse_csv(args.residual_stack_r_methods, ["whitened_svd", "svd"])
    rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        rows.extend(
            residual_stack_candidates_for_layer(
                name,
                weight,
                covariances[name],
                activation_samples.get(name),
                args,
                targets=targets,
                splits=splits,
                q_methods=q_methods,
                s_methods=s_methods,
                r_methods=r_methods,
            )
        )
    annotate_residual_stack_gains(rows)
    return rows


def candidate_rows_for_target(rows: list[dict[str, object]], target: float) -> list[dict[str, object]]:
    return [row for row in rows if math.isclose(float(row["target_memory_ratio"]), float(target), rel_tol=0.0, abs_tol=1e-9)]


def choose_best_residual_rows_by_kind(rows: list[dict[str, object]], target: float, kind: str, *, require_filter: bool) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    selected: list[dict[str, object]] = []
    by_layer: dict[str, list[dict[str, object]]] = {}
    for row in candidate_rows_for_target(rows, target):
        if str(row["candidate_kind"]) != kind:
            continue
        by_layer.setdefault(str(row["layer"]), []).append(row)
    for layer_rows in by_layer.values():
        feasible = [row for row in layer_rows if bool(row["budget_feasible"])]
        if require_filter:
            filtered = [row for row in feasible if bool(row["filter_pass"])]
            pool = filtered if filtered else feasible
        else:
            pool = feasible
        if not pool:
            pool = layer_rows
        best = min(
            pool,
            key=lambda row: (
                float(row["selector_score"]),
                float(row["positive_conditional_rho"]),
                float(row["candidate_memory_ratio"]),
                str(row["q_method"]),
                str(row["s_method"]),
                str(row["r_method"]),
            ),
        )
        best["selected_by_fixed_kind"] = True
        selected.append(best)
    replacements = {str(row["layer"]): row["_final"] for row in selected}  # type: ignore[dict-item]
    return replacements, selected


def select_residual_stack_greedy(rows: list[dict[str, object]], target: float) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], dict[str, object]]:
    target_rows = candidate_rows_for_target(rows, target)
    by_layer: dict[str, list[dict[str, object]]] = {}
    for row in target_rows:
        by_layer.setdefault(str(row["layer"]), []).append(row)
    selected_by_layer: dict[str, dict[str, object]] = {}
    total_params = max(sum(int(layer_rows[0]["layer_parameter_count"]) for layer_rows in by_layer.values()), 1)
    total_memory_params = 0.0
    fallback_layers = 0
    for layer, layer_rows in by_layer.items():
        q_pool = [row for row in layer_rows if str(row["candidate_kind"]) == "q_only"]
        if not q_pool:
            q_pool = layer_rows
        best_q = min(q_pool, key=lambda row: (float(row["selector_score"]), float(row["candidate_memory_ratio"]), str(row["q_method"])))
        selected_by_layer[layer] = best_q
        total_memory_params += float(best_q["candidate_memory_ratio"]) * int(best_q["layer_parameter_count"])

    for layer, layer_rows in by_layer.items():
        current = selected_by_layer[layer]
        feasible = [row for row in layer_rows if bool(row["budget_feasible"])]
        filtered = [row for row in feasible if bool(row["filter_pass"])]
        pool = filtered if filtered else feasible
        if not filtered:
            fallback_layers += 1
        free_better = [
            row
            for row in pool
            if float(row["candidate_memory_ratio"]) <= float(current["candidate_memory_ratio"]) + 1e-12
            and float(row["selector_score"]) < float(current["selector_score"])
        ]
        if free_better:
            best = min(free_better, key=lambda row: (float(row["selector_score"]), float(row["candidate_memory_ratio"])))
            total_memory_params += (float(best["candidate_memory_ratio"]) - float(current["candidate_memory_ratio"])) * int(best["layer_parameter_count"])
            selected_by_layer[layer] = best

    upgrades: list[tuple[float, float, str, dict[str, object]]] = []
    for layer, layer_rows in by_layer.items():
        current = selected_by_layer[layer]
        feasible = [row for row in layer_rows if bool(row["budget_feasible"])]
        filtered = [row for row in feasible if bool(row["filter_pass"])]
        pool = filtered if filtered else feasible
        for row in pool:
            if row is current:
                continue
            benefit = float(current["selector_score"]) - float(row["selector_score"])
            delta_memory = float(row["candidate_memory_ratio"]) - float(current["candidate_memory_ratio"])
            if benefit <= 0.0 or delta_memory <= 1e-12:
                continue
            weighted_cost = delta_memory * int(row["layer_parameter_count"])
            upgrades.append((benefit / max(weighted_cost, EPS), benefit, layer, row))
    budget_params = float(target) * float(total_params)
    for _efficiency, _benefit, layer, row in sorted(upgrades, key=lambda item: (-item[0], -item[1], str(item[3]["candidate_kind"]))):
        current = selected_by_layer[layer]
        if current is row:
            continue
        if float(row["selector_score"]) >= float(current["selector_score"]) - 1e-12:
            continue
        weighted_cost = (float(row["candidate_memory_ratio"]) - float(current["candidate_memory_ratio"])) * int(row["layer_parameter_count"])
        if weighted_cost <= 0.0 or total_memory_params + weighted_cost <= budget_params + 1e-9:
            total_memory_params += weighted_cost
            selected_by_layer[layer] = row

    selected = list(selected_by_layer.values())
    for row in selected:
        row["selected_by_greedy"] = True
    replacements = {str(row["layer"]): row["_final"] for row in selected}  # type: ignore[dict-item]
    summary = {
        "target_memory_ratio": float(target),
        "selected_memory_ratio": total_memory_params / float(total_params),
        "global_budget_feasible": total_memory_params <= budget_params + 1e-9,
        "filter_fallback_layer_count": int(fallback_layers),
        "selected_layer_count": len(selected),
    }
    return replacements, selected, summary


def weighted_memory_for_selected_rows(rows: list[dict[str, object]], baseline_weights: dict[str, torch.Tensor]) -> float:
    total = max(selected_param_count(baseline_weights), 1)
    return sum(float(row["candidate_memory_ratio"]) * int(row["layer_parameter_count"]) for row in rows) / float(total)


def strategy_replacements_budget(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    order: tuple[str, ...],
    methods: dict[str, str],
    args: argparse.Namespace,
    *,
    bits: int,
    keep_fraction: float,
    rank_fraction: float,
) -> dict[str, torch.Tensor]:
    return {
        name: apply_order(
            weight,
            covariances[name],
            order,
            methods,
            bits=bits,
            keep_fraction=keep_fraction,
            rank_fraction=rank_fraction,
            svd_device=args.svd_device,
        )
        for name, weight in baseline_weights.items()
    }


def residual_stack_capacity_grid_values(args: argparse.Namespace) -> list[float]:
    values = {float(args.keep_fraction), float(args.rank_fraction), 1.0}
    for value in np.linspace(0.05, 1.0, 96):
        values.add(float(value))
    return sorted(value for value in values if 0.0 < value <= 1.0)


def choose_keep_rank_for_memory(
    baseline_weights: dict[str, torch.Tensor],
    target: float,
    memory_fn,
    args: argparse.Namespace,
) -> dict[str, object]:
    best: dict[str, object] | None = None
    for keep_fraction, rank_fraction in itertools.product(residual_stack_capacity_grid_values(args), repeat=2):
        memory = float(memory_fn(float(keep_fraction), float(rank_fraction)))
        feasible = memory <= float(target) + 1e-9
        gap = abs(float(target) - memory)
        row = {
            "bits": int(args.bits),
            "keep_fraction": float(keep_fraction),
            "rank_fraction": float(rank_fraction),
            "nominal_memory_ratio": memory,
            "target_memory_ratio": float(target),
            "memory_gap_abs": gap,
            "under_target": feasible,
        }
        if best is None:
            best = row
            continue
        current_key = (
            not feasible,
            gap if feasible else memory - float(target),
            -float(keep_fraction),
            -float(rank_fraction),
        )
        best_feasible = bool(best["under_target"])
        best_key = (
            not best_feasible,
            float(best["memory_gap_abs"]) if best_feasible else float(best["nominal_memory_ratio"]) - float(target),
            -float(best["keep_fraction"]),
            -float(best["rank_fraction"]),
        )
        if current_key < best_key:
            best = row
    assert best is not None
    return best


def choose_sequential_qsr_matched_budget(baseline_weights: dict[str, torch.Tensor], target: float, args: argparse.Namespace) -> dict[str, object]:
    return choose_keep_rank_for_memory(
        baseline_weights,
        target,
        lambda keep, rank: nominal_memory_ratio_for_spec(
            {"order": "qsr", "bits": args.bits, "keep_fraction": keep, "rank_fraction": rank},
            baseline_weights,
        ),
        args,
    )


def choose_spq_matched_budget(baseline_weights: dict[str, torch.Tensor], target: float, args: argparse.Namespace) -> dict[str, object]:
    orders_by_layer = {name: "".join(spq_ops_for_layer(name)) for name in baseline_weights}
    return choose_keep_rank_for_memory(
        baseline_weights,
        target,
        lambda keep, rank: weighted_layerwise_nominal_memory_ratio(
            baseline_weights,
            orders_by_layer,
            bits=args.bits,
            keep_fraction=keep,
            rank_fraction=rank,
        ),
        args,
    )


def dam_factor_replacements(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    activation_samples: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    target: float,
    scale_mode: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], float]:
    replacements: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    total_params = max(selected_param_count(baseline_weights), 1)
    weighted_memory = 0.0
    alpha_grid = parse_float_csv(args.dam_alpha_grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    for name, weight in baseline_weights.items():
        rank = factor_quant_rank_for_memory(weight, target, args.bits)
        final, info = factor_quantized_svd_weight(
            weight,
            covariances[name],
            activation_samples.get(name),
            rank=rank,
            bits=args.bits,
            scale_mode=scale_mode,
            svd_device=args.svd_device,
            alpha_grid=alpha_grid,
        )
        replacements[name] = final
        memory_ratio = factor_quant_memory_ratio_for_rank(weight, rank, args.bits)
        weighted_memory += memory_ratio * int(weight.numel())
        metrics = activation_candidate_metrics(weight, final, covariances[name], activation_samples.get(name))
        rows.append(
            {
                "selection_family": "dam_factor_quant",
                "layer": name,
                "layer_short": short_layer_name(name),
                "layer_family": layer_family(name),
                "candidate_kind": scale_mode,
                "target_memory_ratio": float(target),
                "candidate_memory_ratio": memory_ratio,
                "bits": int(args.bits),
                "factor_rank": rank,
                "dam_alpha": info.get("dam_alpha", ""),
                "dam_alpha_grid": args.dam_alpha_grid if scale_mode == "dam_activation_grid" else "",
                "same_budget_feasible": memory_ratio <= float(target) + 1e-9,
                **metrics,
            }
        )
    return replacements, rows, weighted_memory / float(total_params)


def public_residual_stack_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if not str(key).startswith("_")} for row in rows]


def spq_recipe_diagnostic_rows(
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    deltas: dict[str, dict[str, torch.Tensor]],
    fixed_replacements: dict[str, torch.Tensor],
    guided_selection_rows: list[dict[str, object]],
    methods: dict[str, str],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    guided_by_layer = {str(row["layer"]): row for row in guided_selection_rows}
    rows: list[dict[str, object]] = []
    for name, weight in baseline_weights.items():
        ops = spq_ops_for_layer(name)
        fixed_order = "".join(ops)
        fixed_delta = fixed_replacements[name] - weight
        fixed_cost = 0.5 * hessian_inner_matrix(fixed_delta, fixed_delta, covariances[name])
        reversed_order = "".join(tuple(reversed(ops)))
        reversed_cost = fixed_cost
        if len(ops) > 1:
            reversed_final = apply_order(
                weight,
                covariances[name],
                tuple(reversed(ops)),
                methods,
                bits=args.bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
                svd_device=args.svd_device,
            )
            reversed_delta = reversed_final - weight
            reversed_cost = 0.5 * hessian_inner_matrix(reversed_delta, reversed_delta, covariances[name])
        pair = "".join(sorted(ops)) if len(ops) > 1 else ops[0]
        if len(ops) > 1:
            left, right = sorted(ops)
            rho_h = hessian_cosine_matrix(deltas[name][left], deltas[name][right], covariances[name])
        else:
            rho_h = float("nan")
        guided = guided_by_layer.get(name, {})
        rows.append(
            {
                "layer": name,
                "layer_short": short_layer_name(name),
                "layer_family": layer_family(name),
                "spq_ops": fixed_order,
                "spq_pair": pair,
                "spq_pair_rho_h": rho_h,
                "fixed_spq_order": fixed_order,
                "fixed_spq_predicted_hessian_cost": fixed_cost,
                "reversed_spq_order": reversed_order,
                "reversed_spq_predicted_hessian_cost": reversed_cost,
                "predicted_order_cost_gap_fixed_minus_reversed": fixed_cost - reversed_cost,
                "guided_spq_order": guided.get("selected_order", ""),
                "guided_spq_predicted_hessian_cost": guided.get("selected_predicted_hessian_cost", ""),
                "guided_q_method": guided.get("selected_q_method", ""),
                "guided_s_method": guided.get("selected_s_method", ""),
                "guided_r_method": guided.get("selected_r_method", ""),
            }
        )
    return rows


def evaluate_strategies(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    naive_methods = {"q": "rtn", "s": "magnitude", "r": "svd"}
    slim_proxy_methods = {"q": "rtn", "s": "wanda", "r": "whitened_svd"}
    default_methods = {"q": args.q_method, "s": args.s_method, "r": args.r_method}
    spq_methods = {"q": "rtn", "s": args.spq_s_method, "r": args.spq_r_method}
    hessian_repl, selection_rows = choose_hessian_layerwise(
        baseline_weights,
        covariances,
        q_methods=parse_csv(args.guided_q_methods, ["rtn"]),
        s_methods=["magnitude", "wanda"],
        r_methods=["svd", "whitened_svd"],
        args=args,
    )
    for row in selection_rows:
        row.setdefault("selection_family", "hessian_layerwise")
        row.setdefault("layer_family", layer_family(str(row["layer"])))
    spq_fixed_repl: dict[str, torch.Tensor] = {}
    spq_guided_repl: dict[str, torch.Tensor] = {}
    spq_selection_rows: list[dict[str, object]] = []
    if args.include_spq_strategies:
        spq_fixed_repl = spq_like_replacements(baseline_weights, covariances, spq_methods, args)
        spq_guided_repl, spq_selection_rows = choose_hessian_guided_spq(
            baseline_weights,
            covariances,
            q_methods=parse_csv(args.spq_guided_q_methods, ["rtn"]),
            s_methods=parse_csv(args.spq_guided_s_methods, [args.spq_s_method]),
            r_methods=parse_csv(args.spq_guided_r_methods, [args.spq_r_method]),
            args=args,
        )
        selection_rows.extend(spq_selection_rows)
    strategies = [
        ("baseline", {}, "none", {}),
        ("fixed_qsr_naive", strategy_replacements(baseline_weights, covariances, ("q", "s", "r"), naive_methods, args), "qsr", naive_methods),
        ("fixed_qsr_default", strategy_replacements(baseline_weights, covariances, ("q", "s", "r"), default_methods, args), "qsr", default_methods),
        ("slim_like_srq_proxy", strategy_replacements(baseline_weights, covariances, ("s", "r", "q"), slim_proxy_methods, args), "srq", slim_proxy_methods),
        ("hessian_layerwise", hessian_repl, "layerwise", {"q": "selected", "s": "selected", "r": "selected"}),
    ]
    if args.include_rotation_analysis:
        rotated_methods = {"q": "rotated_rtn", "s": args.s_method, "r": args.r_method}
        strategies.append(
            (
                "fixed_qsr_rotated_q",
                strategy_replacements(baseline_weights, covariances, ("q", "s", "r"), rotated_methods, args),
                "qsr",
                rotated_methods,
            )
        )
    if args.include_spq_strategies:
        strategies.extend(
            [
                ("spq_like_rsq_no_lora", spq_fixed_repl, "spq_layer_typed_rq_sq", spq_methods),
                ("hessian_guided_spq_no_lora", spq_guided_repl, "hessian_guided_spq", {"q": "rtn", "s": "selected", "r": "selected"}),
            ]
        )
        if args.spq_lora_steps > 0:
            strategies.extend(
                [
                    ("spq_like_rsq_lora", spq_fixed_repl, "spq_layer_typed_rq_sq+lora", spq_methods),
                    ("hessian_guided_spq_lora", spq_guided_repl, "hessian_guided_spq+lora", {"q": "rtn", "s": "selected", "r": "selected"}),
                ]
            )
    perf_rows: list[dict[str, object]] = []
    zero_rows: list[dict[str, object]] = []
    low_loss_rows: list[dict[str, object]] = []
    for strategy, replacements, order_name, methods in strategies:
        is_lora_strategy = strategy.endswith("_lora") and not strategy.endswith("_no_lora")
        lora_info: dict[str, object] = {"lora_steps_completed": 0, "lora_params": 0}
        if strategy == "baseline":
            metrics = baseline_metrics
            zero_mean = baseline_zero_shot
            raw_zero_rows: list[dict[str, object]] = []
        elif is_lora_strategy:
            metrics, zero_mean, raw_zero_rows, lora_info = evaluate_recovered_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                recovery_texts=args.recovery_texts,
                eval_texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=args.zero_shot_tasks,
                zero_shot_limit=args.zero_shot_strategy_limit,
                lora_steps=args.spq_lora_steps,
                lora_rank=args.spq_lora_rank,
                lora_alpha=args.spq_lora_alpha,
                lora_lr=args.spq_lora_lr,
                lora_train_limit=args.spq_lora_train_limit,
            )
        else:
            metrics, zero_mean, raw_zero_rows = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
                zero_shot_tasks=args.zero_shot_tasks,
                zero_shot_limit=args.zero_shot_strategy_limit,
            )
        for row in raw_zero_rows:
            zero_rows.append({"scope": "strategy", "strategy": strategy, **row})
        perf_rows.append(
            {
                "strategy": strategy,
                "order": order_name,
                "q_method": methods.get("q", ""),
                "s_method": methods.get("s", ""),
                "r_method": methods.get("r", ""),
                "nll": float(metrics["nll"]),
                "perplexity": float(metrics["perplexity"]),
                "tokens": int(metrics["tokens"]),
                "ppl_degradation": float(metrics["perplexity"]) - float(baseline_metrics["perplexity"]),
                "loss_degradation": float(metrics["nll"]) - float(baseline_metrics["nll"]),
                "zero_shot_accuracy": zero_mean,
                "zero_shot_accuracy_degradation": float("nan") if not math.isfinite(zero_mean) or not math.isfinite(baseline_zero_shot) else baseline_zero_shot - zero_mean,
                "layer_policy": "spq_layer_typed" if strategy.startswith("spq_like") else ("hessian_guided_spq" if strategy.startswith("hessian_guided_spq") else ""),
                "recovery": "lora" if is_lora_strategy else "none",
                "nominal_bits": args.bits,
                "nominal_keep_fraction": args.keep_fraction,
                "nominal_rank_fraction": args.rank_fraction,
                **lora_info,
            }
        )
    if args.include_low_loss_triple:
        low_loss_strategy, low_loss_rows, low_loss_zero_rows = evaluate_low_loss_triple(
            model,
            tokenizer,
            modules,
            baseline_weights,
            covariances,
            baseline_metrics,
            baseline_zero_shot,
            args,
        )
        zero_rows.extend(low_loss_zero_rows)
        if low_loss_strategy is not None:
            perf_rows.append(low_loss_strategy)
    return perf_rows, selection_rows, zero_rows, low_loss_rows


def residual_stack_eval_budget(args: argparse.Namespace) -> float:
    targets = parse_float_csv(args.residual_stack_memory_targets, [0.258, 0.300, 0.350])
    if float(args.residual_stack_eval_budget) > 0.0:
        return float(args.residual_stack_eval_budget)
    q_memory = float(args.bits) / 16.0
    feasible = [target for target in targets if target >= q_memory]
    return float(feasible[0] if feasible else targets[0])


def residual_stack_selected_strategy_row(
    *,
    strategy: str,
    family: str,
    order: str,
    target: float,
    memory_ratio: float,
    predicted_cost: float,
    metrics: dict[str, float | int],
    zero_mean: float,
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    selection_rule: str,
    q_method: str,
    s_method: str,
    r_method: str,
    bits: int | str = "",
    keep_fraction: float | str = "",
    rank_fraction: float | str = "",
    memory_match_rule: str = "",
) -> dict[str, object]:
    signed_ppl_delta = float(metrics["perplexity"]) - float(baseline_metrics["perplexity"])
    signed_nll_delta = float(metrics["nll"]) - float(baseline_metrics["nll"])
    zero_delta = float("nan") if not math.isfinite(zero_mean) or not math.isfinite(baseline_zero_shot) else zero_mean - baseline_zero_shot
    return {
        "strategy": strategy,
        "family": family,
        "order": order,
        "target_memory_ratio": float(target),
        "nominal_memory_ratio": float(memory_ratio),
        "memory_gap_vs_target": float(memory_ratio) - float(target),
        "same_budget_feasible": float(memory_ratio) <= float(target) + 1e-9,
        "q_method": q_method,
        "s_method": s_method,
        "r_method": r_method,
        "predicted_hessian_cost": predicted_cost,
        "nll": float(metrics["nll"]),
        "perplexity": float(metrics["perplexity"]),
        "tokens": int(metrics["tokens"]),
        "signed_nll_delta": signed_nll_delta,
        "signed_nll_delta_percent": 100.0 * signed_nll_delta / max(abs(float(baseline_metrics["nll"])), EPS),
        "signed_ppl_delta": signed_ppl_delta,
        "signed_ppl_delta_percent": 100.0 * signed_ppl_delta / max(abs(float(baseline_metrics["perplexity"])), EPS),
        "clipped_ppl_drop_percent": 100.0 * max(signed_ppl_delta, 0.0) / max(abs(float(baseline_metrics["perplexity"])), EPS),
        "zero_shot_accuracy": zero_mean,
        "zero_shot_accuracy_delta": zero_delta,
        "zero_shot_accuracy_drop": float("nan") if not math.isfinite(zero_delta) else max(-zero_delta, 0.0),
        "selection_rule": selection_rule,
        "bits": bits,
        "keep_fraction": keep_fraction,
        "rank_fraction": rank_fraction,
        "memory_match_rule": memory_match_rule,
    }


def evaluate_residual_stack_replacements(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    zero_shot_limit: int,
) -> tuple[dict[str, float | int], float, list[dict[str, object]]]:
    return evaluate_replacements(
        model,
        tokenizer,
        modules,
        baseline_weights,
        replacements,
        texts=args.eval_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=args.device,
        eval_limit=args.eval_limit,
        zero_shot_tasks=args.zero_shot_tasks,
        zero_shot_limit=zero_shot_limit,
    )


def append_residual_stack_eval_row(
    rows: list[dict[str, object]],
    zero_rows: list[dict[str, object]],
    *,
    strategy: str,
    family: str,
    order: str,
    target: float,
    memory_ratio: float,
    predicted_cost: float,
    replacements: dict[str, torch.Tensor],
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    args: argparse.Namespace,
    zero_shot_limit: int,
    selection_rule: str,
    q_method: str,
    s_method: str,
    r_method: str,
    bits: int | str = "",
    keep_fraction: float | str = "",
    rank_fraction: float | str = "",
    memory_match_rule: str = "",
) -> None:
    metrics, zero_mean, raw_zero_rows = evaluate_residual_stack_replacements(
        model,
        tokenizer,
        modules,
        baseline_weights,
        replacements,
        args=args,
        zero_shot_limit=zero_shot_limit,
    )
    for zero_row in raw_zero_rows:
        zero_rows.append({"scope": "residual_stack_validate", "strategy": strategy, **zero_row})
    rows.append(
        residual_stack_selected_strategy_row(
            strategy=strategy,
            family=family,
            order=order,
            target=target,
            memory_ratio=memory_ratio,
            predicted_cost=predicted_cost,
            metrics=metrics,
            zero_mean=zero_mean,
            baseline_metrics=baseline_metrics,
            baseline_zero_shot=baseline_zero_shot,
            selection_rule=selection_rule,
            q_method=q_method,
            s_method=s_method,
            r_method=r_method,
            bits=bits,
            keep_fraction=keep_fraction,
            rank_fraction=rank_fraction,
            memory_match_rule=memory_match_rule,
        )
    )


def plot_residual_stack_frontier(figures_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    compressed = [row for row in rows if str(row["strategy"]) != "baseline"]
    x = [float(row["nominal_memory_ratio"]) for row in compressed]
    y = [float(row["perplexity"]) for row in compressed]
    colors = ["#0f766e" if str(row["family"]).startswith("residual") else "#64748b" for row in compressed]
    ax.scatter(x, y, s=70, c=colors, edgecolors="white", linewidth=0.8)
    baseline = next((row for row in rows if str(row["strategy"]) == "baseline"), None)
    if baseline:
        ax.axhline(float(baseline["perplexity"]), color="black", linewidth=1.0, linestyle="--", label="dense baseline")
    for row, xi, yi in zip(compressed, x, y):
        ax.annotate(str(row["strategy"]), (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("nominal memory ratio")
    ax.set_ylabel("PPL")
    ax.set_title("Residual-stack same-budget frontier")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures_dir / "memory_ppl_frontier.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "memory_ppl_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residual_stack_activation_by_layer(figures_dir: Path, rows: list[dict[str, object]], target: float) -> None:
    target_rows = [
        row
        for row in candidate_rows_for_target(rows, target)
        if bool(row.get("budget_feasible", False)) and str(row["candidate_kind"]) in {"q_only", "q_l", "q_s", "q_s_l"}
    ]
    if not target_rows:
        return
    best: dict[tuple[str, str], dict[str, object]] = {}
    for row in target_rows:
        key = (str(row["layer_short"]), str(row["candidate_kind"]))
        current = best.get(key)
        if current is None or float(row["activation_reconstruction_error"]) < float(current["activation_reconstruction_error"]):
            best[key] = row
    layers = sorted({key[0] for key in best})
    kinds = ["q_only", "q_l", "q_s", "q_s_l"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(7.5, 1.2 * len(layers)), 4.8))
    positions = np.arange(len(layers))
    palette = {"q_only": "#94a3b8", "q_l": "#2563eb", "q_s": "#f97316", "q_s_l": "#0f766e"}
    for idx, kind in enumerate(kinds):
        values = [float(best[(layer, kind)]["activation_reconstruction_error"]) if (layer, kind) in best else np.nan for layer in layers]
        ax.bar(positions + (idx - 1.5) * width, values, width=width, label=kind, color=palette[kind])
    ax.set_xticks(positions)
    ax.set_xticklabels(layers, rotation=25, ha="right")
    ax.set_ylabel("activation reconstruction error")
    ax.set_title(f"Candidate activation error by layer at memory {target:.3f}")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "candidate_activation_error_by_layer.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "candidate_activation_error_by_layer.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residual_stack_overlap_heatmap(figures_dir: Path, selected_rows: list[dict[str, object]]) -> None:
    if not selected_rows:
        return
    columns = ["rho_q_error_s_res", "rho_q_error_l_res", "rho_q_plus_s_error_l_res", "rho_s_res_l_res"]
    data = np.asarray([[float(row.get(col, 0.0)) for col in columns] for row in selected_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.38 * len(selected_rows) + 1.6)))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(selected_rows)))
    ax.set_yticklabels([str(row["layer_short"]) for row in selected_rows], fontsize=8)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(["rho(Qerr,S)", "rho(Qerr,L)", "rho(Q+S err,L)", "rho(S,L)"], rotation=25, ha="right")
    ax.set_title("Conditional Hessian overlap for selected residual recipe")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.tight_layout()
    fig.savefig(figures_dir / "conditional_overlap_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "conditional_overlap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residual_structure_scatter(figures_dir: Path, rows: list[dict[str, object]], target: float) -> None:
    target_rows = [row for row in candidate_rows_for_target(rows, target) if str(row["candidate_kind"]) != "q_only" and bool(row.get("budget_feasible", False))]
    if not target_rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    palette = {"q_l": "#2563eb", "q_s": "#f97316", "q_s_l": "#0f766e", "q_l_s": "#a855f7"}
    for kind in sorted({str(row["candidate_kind"]) for row in target_rows}):
        group = [row for row in target_rows if str(row["candidate_kind"]) == kind]
        ax.scatter(
            [float(row["positive_conditional_rho"]) for row in group],
            [float(row["activation_gain_vs_q_only"]) for row in group],
            s=[34 if bool(row["filter_pass"]) else 18 for row in group],
            alpha=0.78,
            label=kind,
            color=palette.get(kind, "#64748b"),
            edgecolors="white",
            linewidth=0.5,
        )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("positive conditional Hessian rho")
    ax.set_ylabel("activation-error gain vs Q-only")
    ax.set_title("Residual structure benefit vs overlap")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "residual_structure_scatter.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "residual_structure_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def write_residual_stack_summary(
    root: Path,
    args: argparse.Namespace,
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    strategy_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
    selector_summary: dict[str, object],
    target: float,
) -> None:
    by_strategy = {str(row["strategy"]): row for row in strategy_rows}
    ql = by_strategy.get("residual_q_l_same_budget")
    qsl = by_strategy.get("residual_q_s_l_same_budget")
    selector = by_strategy.get("residual_stack_selector")
    fixed_spq = by_strategy.get("spq_like_rsq_no_lora_matched_budget")
    guided_spq = by_strategy.get("hessian_guided_spq_no_lora_matched_budget")
    sequential = by_strategy.get("sequential_qsr_matched_budget")
    plain_lq = by_strategy.get("paper_lq_factor_quant_matched_budget")
    dam_closed = by_strategy.get("paper_dam_closed_matched_budget")
    dam_grid = by_strategy.get("paper_dam_activation_grid_matched_budget")
    target_candidates = [row for row in candidate_rows_for_target(candidate_rows, target) if str(row["candidate_kind"]) != "q_only"]
    rho_values = [float(row["positive_conditional_rho"]) for row in target_candidates]
    gains = [float(row["activation_gain_vs_q_only"]) for row in target_candidates]
    rho_gain, rho_gain_n = spearmanr(rho_values, gains) if len(rho_values) >= 2 else (float("nan"), len(rho_values))
    selected_kinds = {str(row["candidate_kind"]): 0 for row in selected_rows}
    for row in selected_rows:
        selected_kinds[str(row["candidate_kind"])] = selected_kinds.get(str(row["candidate_kind"]), 0) + 1

    def row_line(label: str, row: dict[str, object] | None) -> str:
        if row is None:
            return f"| {label} | unavailable | | | | | |"
        return (
            f"| {label} | {float(row['nominal_memory_ratio']):.4f} | {float(row['perplexity']):.4f} | "
            f"{float(row['signed_ppl_delta']):+.4f} | {float(row['zero_shot_accuracy']):.4f} | "
            f"{float(row['zero_shot_accuracy_delta']):+.4f} | {row['same_budget_feasible']} |"
        )

    verdict = "not yet supported"
    if ql is not None and qsl is not None:
        if bool(qsl["same_budget_feasible"]) and float(qsl["perplexity"]) < float(ql["perplexity"]):
            verdict = "positive on PPL for Q+S+L vs Q+L at the recorded budget"
        else:
            verdict = "not positive against Q+L at the recorded budget"
    lines = [
        "# Residual-Stack Validation",
        "",
        f"- Model: `{args.model}`",
        f"- Selected modules: {len(args.selected_layers)} (`{', '.join(args.selected_layers)}`)",
        f"- Mode: `residual_stack_validate`; target memory ratio: {target:.4f}; q base: {args.bits}-bit.",
        f"- Text source: `{args.text_source_used}`; calib={len(args.calib_texts)} texts, eval={len(args.eval_texts)} texts.",
        f"- Dense baseline PPL: {float(baseline_metrics['perplexity']):.4f}; NLL: {float(baseline_metrics['nll']):.4f}; zero-shot mean: {baseline_zero_shot:.4f}.",
        "",
        "## Strategy Results",
        "",
        "| strategy | memory | PPL | signed PPL delta | zero-shot | zero-shot delta | <= target |",
        "|---|---:|---:|---:|---:|---:|---|",
        row_line("Q only", by_strategy.get("residual_q_only")),
        row_line("Q+L same budget", ql),
        row_line("Q+S same budget", by_strategy.get("residual_q_s_same_budget")),
        row_line("Q+S+L same budget", qsl),
        row_line("Residual-stack selector", selector),
        row_line("Sequential QSR matched", sequential),
        row_line("Fixed SPQ-like matched", fixed_spq),
        row_line("Hessian-guided SPQ matched", guided_spq),
        row_line("Paper L->Q factor matched", plain_lq),
        row_line("Paper DAM closed matched", dam_closed),
        row_line("Paper DAM activation-grid matched", dam_grid),
        "",
        "## Evidence Notes",
        "",
        f"- Same-budget rule: a residual-stack win is counted only when `nominal_memory_ratio <= target_memory_ratio`; residual rows use additive component accounting and baseline rows use a memory-only keep/rank grid matched under the same target.",
        f"- Selector memory summary: selected={float(selector_summary.get('selected_memory_ratio', float('nan'))):.4f}, global feasible={selector_summary.get('global_budget_feasible')}, filter fallback layers={selector_summary.get('filter_fallback_layer_count')}.",
        f"- Selected candidate mix: {json.dumps(selected_kinds, sort_keys=True)}.",
        f"- Conditional-overlap diagnostic: Spearman(positive rho, activation gain vs Q-only) = {rho_gain:.4f} (n={rho_gain_n}); negative values mean lower conflict tends to give larger activation gain.",
        f"- DAM comparison rows are DAM-like proxies implemented from the paper equations, not an official repository: `dam_closed` uses Eq.21-style balancing, and `dam_activation_grid` selects the diagonal exponent using calibration activation reconstruction.",
        f"- Conservative verdict for this run: {verdict}.",
        "",
        "## Artifacts",
        "",
        "- `metrics/residual_stack_candidates.csv`",
        "- `metrics/residual_stack_selection.csv`",
        "- `metrics/dam_factor_selection.csv` when `--include-dam-comparison` is enabled",
        "- `metrics/residual_stack_strategy.csv`",
        "- `metrics/residual_stack_zero_shot.csv`",
        "- `selected_recipe.json`",
        "- `figures/memory_ppl_frontier.png`",
        "- `figures/candidate_activation_error_by_layer.png`",
        "- `figures/conditional_overlap_heatmap.png`",
        "- `figures/residual_structure_scatter.png`",
        "",
        "This experiment validates a framework hypothesis only; it is not a SOTA claim.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_residual_stack_validate(
    *,
    root: Path,
    metrics_dir: Path,
    figures_dir: Path,
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    activation_samples: dict[str, torch.Tensor],
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    args: argparse.Namespace,
) -> None:
    target = residual_stack_eval_budget(args)
    zero_shot_limit = int(args.residual_stack_zero_shot_limit) if args.residual_stack_zero_shot_limit is not None else int(args.zero_shot_strategy_limit)
    candidate_rows = generate_residual_stack_candidates(baseline_weights, covariances, activation_samples, args)
    strategy_rows: list[dict[str, object]] = [
        residual_stack_selected_strategy_row(
            strategy="baseline",
            family="baseline",
            order="none",
            target=target,
            memory_ratio=1.0,
            predicted_cost=0.0,
            metrics=baseline_metrics,
            zero_mean=baseline_zero_shot,
            baseline_metrics=baseline_metrics,
            baseline_zero_shot=baseline_zero_shot,
            selection_rule="dense_uncompressed",
            q_method="",
            s_method="",
            r_method="",
        )
    ]
    zero_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    dam_selection_rows: list[dict[str, object]] = []

    residual_specs = [
        ("residual_q_only", "q_only", "q", "Q base only", False),
        ("residual_q_l_same_budget", "q_l", "q+l_res", "best Q+L candidate under the target memory", False),
        ("residual_q_s_same_budget", "q_s", "q+s_res", "best Q+S candidate under the target memory", False),
        ("residual_q_s_l_same_budget", "q_s_l", "q+s_res+l_res", "best Q+S+L split under the target memory", False),
    ]
    for strategy, kind, order, rule, require_filter in residual_specs:
        replacements, selected = choose_best_residual_rows_by_kind(candidate_rows, target, kind, require_filter=require_filter)
        if not replacements:
            continue
        selection_rows.extend({"strategy": strategy, **public} for public in public_residual_stack_rows(selected))
        append_residual_stack_eval_row(
            strategy_rows,
            zero_rows,
            strategy=strategy,
            family="residual_stack",
            order=order,
            target=target,
            memory_ratio=weighted_memory_for_selected_rows(selected, baseline_weights),
            predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, replacements),
            replacements=replacements,
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            baseline_weights=baseline_weights,
            baseline_metrics=baseline_metrics,
            baseline_zero_shot=baseline_zero_shot,
            args=args,
            zero_shot_limit=zero_shot_limit,
            selection_rule=rule,
            q_method="selected",
            s_method="selected",
            r_method="selected",
        )

    selector_replacements, selector_rows, selector_summary = select_residual_stack_greedy(candidate_rows, target)
    selection_rows.extend({"strategy": "residual_stack_selector", **public} for public in public_residual_stack_rows(selector_rows))
    append_residual_stack_eval_row(
        strategy_rows,
        zero_rows,
        strategy="residual_stack_selector",
        family="residual_stack_selector",
        order="greedy_layerwise",
        target=target,
        memory_ratio=float(selector_summary["selected_memory_ratio"]),
        predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, selector_replacements),
        replacements=selector_replacements,
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        baseline_metrics=baseline_metrics,
        baseline_zero_shot=baseline_zero_shot,
        args=args,
        zero_shot_limit=zero_shot_limit,
        selection_rule="filter_positive_conditional_rho_then_greedy_activation_capacity",
        q_method="selected",
        s_method="selected",
        r_method="selected",
    )

    default_methods = {"q": args.q_method, "s": args.s_method, "r": args.r_method}
    sequential_budget = choose_sequential_qsr_matched_budget(baseline_weights, target, args)
    sequential_replacements = strategy_replacements_budget(
        baseline_weights,
        covariances,
        ("q", "s", "r"),
        default_methods,
        args,
        bits=int(sequential_budget["bits"]),
        keep_fraction=float(sequential_budget["keep_fraction"]),
        rank_fraction=float(sequential_budget["rank_fraction"]),
    )
    append_residual_stack_eval_row(
        strategy_rows,
        zero_rows,
        strategy="sequential_qsr_matched_budget",
        family="sequential_qsr",
        order="qsr",
        target=target,
        memory_ratio=float(sequential_budget["nominal_memory_ratio"]),
        predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, sequential_replacements),
        replacements=sequential_replacements,
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        baseline_metrics=baseline_metrics,
        baseline_zero_shot=baseline_zero_shot,
        args=args,
        zero_shot_limit=zero_shot_limit,
        selection_rule="memory_matched_fixed_sequential_qsr_no_metric_selection",
        q_method=args.q_method,
        s_method=args.s_method,
        r_method=args.r_method,
        bits=int(sequential_budget["bits"]),
        keep_fraction=float(sequential_budget["keep_fraction"]),
        rank_fraction=float(sequential_budget["rank_fraction"]),
        memory_match_rule="grid_closest_under_target",
    )

    spq_methods = {"q": "rtn", "s": args.spq_s_method, "r": args.spq_r_method}
    spq_budget = choose_spq_matched_budget(baseline_weights, target, args)
    spq_replacements = spq_like_replacements_budget(
        baseline_weights,
        covariances,
        spq_methods,
        args,
        bits=int(spq_budget["bits"]),
        keep_fraction=float(spq_budget["keep_fraction"]),
        rank_fraction=float(spq_budget["rank_fraction"]),
    )
    append_residual_stack_eval_row(
        strategy_rows,
        zero_rows,
        strategy="spq_like_rsq_no_lora_matched_budget",
        family="spq_like",
        order="spq_layer_typed_rq_sq",
        target=target,
        memory_ratio=float(spq_budget["nominal_memory_ratio"]),
        predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, spq_replacements),
        replacements=spq_replacements,
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        baseline_metrics=baseline_metrics,
        baseline_zero_shot=baseline_zero_shot,
        args=args,
        zero_shot_limit=zero_shot_limit,
        selection_rule="memory_matched_fixed_spq_like_recipe_no_metric_selection",
        q_method="rtn",
        s_method=args.spq_s_method,
        r_method=args.spq_r_method,
        bits=int(spq_budget["bits"]),
        keep_fraction=float(spq_budget["keep_fraction"]),
        rank_fraction=float(spq_budget["rank_fraction"]),
        memory_match_rule="grid_closest_under_target",
    )

    guided_spq_replacements, guided_spq_selection = choose_hessian_guided_spq_budget(
        baseline_weights,
        covariances,
        q_methods=parse_csv(args.spq_guided_q_methods, ["rtn"]),
        s_methods=parse_csv(args.spq_guided_s_methods, [args.spq_s_method]),
        r_methods=parse_csv(args.spq_guided_r_methods, [args.spq_r_method]),
        args=args,
        bits=int(spq_budget["bits"]),
        keep_fraction=float(spq_budget["keep_fraction"]),
        rank_fraction=float(spq_budget["rank_fraction"]),
    )
    selection_rows.extend({"strategy": "hessian_guided_spq_no_lora_matched_budget", **row} for row in guided_spq_selection)
    guided_spq_memory = weighted_layerwise_nominal_memory_ratio(
        baseline_weights,
        {str(row["layer"]): str(row["selected_order"]) for row in guided_spq_selection},
        bits=int(spq_budget["bits"]),
        keep_fraction=float(spq_budget["keep_fraction"]),
        rank_fraction=float(spq_budget["rank_fraction"]),
    )
    append_residual_stack_eval_row(
        strategy_rows,
        zero_rows,
        strategy="hessian_guided_spq_no_lora_matched_budget",
        family="hessian_guided_spq",
        order="spq_layerwise",
        target=target,
        memory_ratio=guided_spq_memory,
        predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, guided_spq_replacements),
        replacements=guided_spq_replacements,
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        baseline_metrics=baseline_metrics,
        baseline_zero_shot=baseline_zero_shot,
        args=args,
        zero_shot_limit=zero_shot_limit,
        selection_rule="memory_matched_hessian_guided_spq_fixed_budget_no_metric_selection",
        q_method="selected",
        s_method="selected",
        r_method="selected",
        bits=int(spq_budget["bits"]),
        keep_fraction=float(spq_budget["keep_fraction"]),
        rank_fraction=float(spq_budget["rank_fraction"]),
        memory_match_rule="grid_closest_under_target",
    )

    if bool(args.include_dam_comparison):
        for strategy, scale_mode, label in [
            ("paper_lq_factor_quant_matched_budget", "plain_lq", "plain low-rank factor quantization before reconstruction"),
            ("paper_dam_closed_matched_budget", "dam_closed", "DAM Eq.21-style closed-form diagonal scaling"),
            ("paper_dam_activation_grid_matched_budget", "dam_activation_grid", "DAM calibration activation-grid diagonal scaling"),
        ]:
            dam_replacements, dam_rows, dam_memory = dam_factor_replacements(
                baseline_weights,
                covariances,
                activation_samples,
                args,
                target=target,
                scale_mode=scale_mode,
            )
            dam_selection_rows.extend({"strategy": strategy, **row} for row in dam_rows)
            append_residual_stack_eval_row(
                strategy_rows,
                zero_rows,
                strategy=strategy,
                family="paper_dam_like",
                order="lowrank_then_quantized_factors",
                target=target,
                memory_ratio=dam_memory,
                predicted_cost=predicted_hessian_cost_for_replacements(baseline_weights, covariances, dam_replacements),
                replacements=dam_replacements,
                model=model,
                tokenizer=tokenizer,
                modules=modules,
                baseline_weights=baseline_weights,
                baseline_metrics=baseline_metrics,
                baseline_zero_shot=baseline_zero_shot,
                args=args,
                zero_shot_limit=zero_shot_limit,
                selection_rule=label,
                q_method="factor_rtn",
                s_method="",
                r_method=scale_mode,
                bits=args.bits,
                keep_fraction="",
                rank_fraction="per_layer_memory_matched_rank",
                memory_match_rule="factor_rank_floor_under_target",
            )

    candidate_public = public_residual_stack_rows(candidate_rows)
    selector_public = public_residual_stack_rows(selector_rows)
    write_csv(metrics_dir / "residual_stack_candidates.csv", candidate_public)
    write_csv(metrics_dir / "residual_stack_selection.csv", selection_rows)
    write_csv(metrics_dir / "dam_factor_selection.csv", dam_selection_rows)
    write_csv(metrics_dir / "residual_stack_strategy.csv", strategy_rows)
    write_csv(metrics_dir / "residual_stack_zero_shot.csv", zero_rows)
    write_json(
        root / "selected_recipe.json",
        {
            "mode": "residual_stack_validate",
            "target_memory_ratio": target,
            "selector_summary": selector_summary,
            "selected_layers": selector_public,
        },
    )
    plot_residual_stack_frontier(figures_dir, strategy_rows)
    plot_residual_stack_activation_by_layer(figures_dir, candidate_rows, target)
    plot_residual_stack_overlap_heatmap(figures_dir, selector_rows)
    plot_residual_structure_scatter(figures_dir, candidate_rows, target)
    write_residual_stack_summary(root, args, baseline_metrics, baseline_zero_shot, strategy_rows, candidate_rows, selector_rows, selector_summary, target)
    restore_weights(modules, baseline_weights)


def plot_dashboard(
    figures_dir: Path,
    additivity: list[dict[str, object]],
    order_gap: list[dict[str, object]],
    correlations: list[dict[str, object]],
    strategy: list[dict[str, object]],
) -> None:
    corr_lookup = {(row["family"], row["x"], row["y"]): float(row["spearman_rho"]) for row in correlations}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    names = [
        ("|rho_H| vs |A_ij|", ("additivity", "abs_rho_h", "abs_additivity_error")),
        ("|rho_H| vs PPL deg.", ("real_ppl", "abs_rho_h", "ppl_degradation_pair")),
        ("Taylor vs loss deg.", ("taylor", "taylor_predicted_loss_delta", "loss_degradation_pair")),
        ("Frobenius vs loss deg.", ("frobenius_baseline", "frobenius_delta_sum", "loss_degradation_pair")),
        ("Trace-only vs loss deg.", ("trace_only_baseline", "trace_only_cost", "loss_degradation_pair")),
    ]
    values = [corr_lookup.get(key, float("nan")) for _, key in names]
    ax = axes[0, 0]
    ax.barh([name for name, _ in names][::-1], values[::-1], color=["#64748b", "#64748b", "#0f766e", "#2563eb", "#2563eb"][::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Prediction and baseline correlations")
    ax.grid(True, axis="x", alpha=0.22)

    x = [float(row["abs_rho_h"]) for row in additivity]
    y = [float(row["abs_additivity_error"]) for row in additivity]
    ax = axes[0, 1]
    ax.scatter(x, y, s=38, alpha=0.82, color="#ef4444", edgecolors="white", linewidth=0.5)
    if len(x) >= 2:
        coef = np.polyfit(np.asarray(x), np.asarray(y), 1)
        xx = np.linspace(min(x), max(x), 100)
        ax.plot(xx, coef[0] * xx + coef[1], color="black", linewidth=1)
    ax.set_xlabel("|rho_H|")
    ax.set_ylabel("|additivity error|")
    ax.set_title("Hessian overlap vs additivity")
    ax.text(0.03, 0.95, f"Spearman={corr_lookup.get(('additivity', 'abs_rho_h', 'abs_additivity_error'), float('nan')):.3f}", transform=ax.transAxes, va="top")
    ax.grid(True, alpha=0.22)

    labels = [str(row["strategy"]) for row in strategy]
    ppl_deg = [float(row["ppl_degradation"]) for row in strategy]
    ax = axes[1, 0]
    palette = ["#94a3b8", "#64748b", "#64748b", "#0f766e", "#2563eb", "#f97316", "#b45309", "#a855f7", "#7e22ce"]
    ax.bar(labels, ppl_deg, color=[palette[i % len(palette)] for i in range(len(labels))])
    ax.tick_params(axis="x", labelrotation=25)
    ax.set_ylabel("PPL degradation")
    ax.set_title("Strategy comparison")
    ax.grid(True, axis="y", alpha=0.22)

    order_names = [
        ("R-first overlap", ("order_gap_r_first_overlap", "abs_r_first_conditional_hessian_overlap", "abs_loss_gap")),
        ("Spectrum entropy", ("spectrum_order_entropy", "abs_first_spectral_entropy_delta", "abs_loss_gap")),
        ("Weight disagreement", ("order_disagreement", "final_weight_disagreement", "abs_loss_gap")),
        ("Symmetric overlap", ("order_gap", "max_abs_conditional_hessian_overlap", "abs_loss_gap")),
    ]
    order_values = [corr_lookup.get(key, float("nan")) for _, key in order_names]
    ax = axes[1, 1]
    ax.barh([name for name, _ in order_names][::-1], order_values[::-1], color=["#64748b", "#0f766e", "#0f766e", "#2563eb"][::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho with |order loss gap|")
    ax.set_title("Order-gap explanations")
    ax.grid(True, axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures_dir / "pretrained_goal_dashboard.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "pretrained_goal_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_singular_spectrum(
    figures_dir: Path,
    order_gap: list[dict[str, object]],
    baseline_weights: dict[str, torch.Tensor],
    covariances: dict[str, torch.Tensor],
    methods: dict[str, str],
    args: argparse.Namespace,
) -> None:
    if not order_gap:
        return
    row = max(order_gap, key=lambda item: float(item["abs_loss_gap"]))
    name = str(row["layer"])
    weight = baseline_weights[name]
    left_order = tuple(str(row["left_order"]))
    right_order = tuple(str(row["right_order"]))
    first_left = compress_weight(weight, covariances[name], left_order[0], methods[left_order[0]], bits=args.bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction, svd_device=args.svd_device)
    first_right = compress_weight(weight, covariances[name], right_order[0], methods[right_order[0]], bits=args.bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction, svd_device=args.svd_device)
    final_left = apply_order(weight, covariances[name], left_order, methods, bits=args.bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction, svd_device=args.svd_device)
    final_right = apply_order(weight, covariances[name], right_order, methods, bits=args.bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction, svd_device=args.svd_device)
    series = [
        ("W", weight),
        (f"after {left_order[0]}", first_left),
        (f"after {right_order[0]}", first_right),
        ("final " + "".join(left_order), final_left),
        ("final " + "".join(right_order), final_right),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for label, mat in series:
        vals = np.linalg.svd(mat.float().cpu().numpy(), compute_uv=False)
        vals = vals / max(float(vals[0]) if vals.size else 0.0, EPS)
        ax.plot(np.arange(1, min(vals.size, 80) + 1), vals[:80], label=label, linewidth=1.4)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("normalized singular value")
    ax.set_title(f"Singular spectrum for largest order gap: {short_layer_name(name)}")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "largest_order_gap_singular_spectrum.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "largest_order_gap_singular_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    root: Path,
    args: argparse.Namespace,
    baseline_metrics: dict[str, float | int],
    baseline_zero_shot: float,
    additivity: list[dict[str, object]],
    order_gap: list[dict[str, object]],
    correlations: list[dict[str, object]],
    strategy: list[dict[str, object]],
    method_status: list[dict[str, object]],
) -> None:
    corr = {(row["family"], row["x"], row["y"]): row for row in correlations}

    def c(family: str, x: str, y: str) -> str:
        row = corr.get((family, x, y))
        if row is None:
            return "nan"
        return f"{float(row['spearman_rho']):.4f} (n={int(row['n'])})"

    high = max(additivity, key=lambda row: float(row["abs_rho_h"])) if additivity else None
    largest_gap = max(order_gap, key=lambda row: float(row["abs_loss_gap"])) if order_gap else None
    best_strategy = min([row for row in strategy if row["strategy"] != "baseline"], key=lambda row: float(row["perplexity"])) if len(strategy) > 1 else None
    unavailable = [row for row in method_status if str(row["status"]) == "unavailable"]
    lines = [
        "# Pretrained Small-LLM Compression Orthogonality",
        "",
        f"- Model: `{args.model}`",
        f"- Target modules: `{', '.join(args.module_types)}`; selected count: {len(args.selected_layers)}",
        f"- Text source: `{args.text_source_used}`; split policy: `{args.text_split_policy}` "
        f"(calib={len(args.calib_texts)}, eval={len(args.eval_texts)}, recovery={len(args.recovery_texts)} texts).",
        f"- Compression settings: q={args.q_method}/bits{args.bits}, s={args.s_method}/keep{args.keep_fraction}, r={args.r_method}/rank{args.rank_fraction}",
        f"- Baseline PPL: {float(baseline_metrics['perplexity']):.4f}; NLL: {float(baseline_metrics['nll']):.4f}; zero-shot mean: {baseline_zero_shot:.4f}",
        "",
        "## Goal-Criterion Evidence",
        "",
        f"- Hessian/Gauss-Newton proxy cosine heatmap generated for q/s/r over {len(args.selected_layers)} pretrained model modules; the proxy is local activation covariance `X^T X`, not the full model Hessian.",
        f"- Higher overlap vs linearized perturbation additivity: Spearman(|rho_H|, |A_ij|) = {c('additivity', 'abs_rho_h', 'abs_additivity_error')}. Additivity rows use `W + Delta_i + Delta_j`, while executable order effects are reported separately in `order_gap.csv`.",
        f"- Real degradation: Spearman(|rho_H|, PPL degradation) = {c('real_ppl', 'abs_rho_h', 'ppl_degradation_pair')}; zero-shot degradation = {c('real_zero_shot', 'abs_rho_h', 'zero_shot_accuracy_degradation_pair')}.",
        f"- Taylor/cross-term prediction vs actual loss degradation = {c('taylor', 'taylor_predicted_loss_delta', 'loss_degradation_pair')}; Frobenius baseline = {c('frobenius_baseline', 'frobenius_delta_sum', 'loss_degradation_pair')}; trace-only baseline = {c('trace_only_baseline', 'trace_only_cost', 'loss_degradation_pair')}.",
        f"- Order gap explanation: R-first conditional overlap = {c('order_gap_r_first_overlap', 'abs_r_first_conditional_hessian_overlap', 'abs_loss_gap')}; singular entropy shift = {c('spectrum_order_entropy', 'abs_first_spectral_entropy_delta', 'abs_loss_gap')}; symmetric overlap = {c('order_gap', 'max_abs_conditional_hessian_overlap', 'abs_loss_gap')}.",
    ]
    if high:
        lines.append(
            f"- Highest |rho_H| row: {high['layer_short']} pair={high['pair']} |rho_H|={float(high['abs_rho_h']):.4f}, |A_ij|={float(high['abs_additivity_error']):.4f}."
        )
    if largest_gap:
        lines.append(
            f"- Largest order gap: {largest_gap['layer_short']} {largest_gap['left_order']} vs {largest_gap['right_order']} abs loss gap={float(largest_gap['abs_loss_gap']):.4f}."
        )
    if best_strategy:
        baseline = next(row for row in strategy if row["strategy"] == "baseline")
        lines.append(
            f"- Best compressed strategy by PPL: {best_strategy['strategy']} PPL={float(best_strategy['perplexity']):.4f}, degradation={float(best_strategy['ppl_degradation']):.4f}; baseline PPL={float(baseline['perplexity']):.4f}."
        )
    by_strategy = {str(row["strategy"]): row for row in strategy}
    if "fixed_qsr_rotated_q" in by_strategy:
        rotated = by_strategy["fixed_qsr_rotated_q"]
        default = by_strategy.get("fixed_qsr_default")
        lines.extend(["", "## Rotation-Quantization Evidence", ""])
        lines.append(
            f"- Hadamard rotated RTN is evaluated as `fixed_qsr_rotated_q` with q=rotated_rtn, s={args.s_method}, r={args.r_method}; "
            f"PPL={float(rotated['perplexity']):.4f}, degradation={float(rotated['ppl_degradation']):.4f}."
        )
        if default:
            lines.append(
                f"- Compared with `fixed_qsr_default`, rotated-Q delta PPL={float(rotated['perplexity']) - float(default['perplexity']):.4f} under the same bits/keep/rank settings."
            )
        lines.append(
            "- `metrics/rotation_quantization.csv` records RTN vs rotated RTN relative weight error, Hessian self cost, and input-channel max/median outlier ratios."
        )
    if "low_loss_triple_stack" in by_strategy:
        low_loss = by_strategy["low_loss_triple_stack"]
        lines.extend(["", "## Low-Loss Triple-Stack Evidence", ""])
        lines.append(
            f"- `low_loss_triple_stack` applies all three operations with order={low_loss['order']}, q={low_loss['q_method']}, "
            f"s={low_loss['s_method']}, r={low_loss['r_method']}, bits={low_loss['nominal_bits']}, "
            f"keep={float(low_loss['nominal_keep_fraction']):.4f}, rank={float(low_loss['nominal_rank_fraction']):.4f}."
        )
        lines.append(
            f"- Benchmark-drop criterion: metric={low_loss['benchmark_metric_effective']} "
            f"(requested={low_loss['benchmark_metric_requested']}), drop={float(low_loss['benchmark_drop_percent']):.4f}%, "
            f"threshold={float(low_loss['lossless_threshold_percent']):.4f}%, pass={low_loss['lossless_pass']}."
        )
        lines.append(
            f"- Result: PPL={float(low_loss['perplexity']):.4f}, PPL degradation={float(low_loss['ppl_degradation']):.4f}, "
            f"zero-shot={float(low_loss['zero_shot_accuracy']):.4f}."
        )
        lines.append("- `metrics/low_loss_triple_candidates.csv` records every evaluated conservative Q+S+R candidate.")
    if "spq_like_rsq_no_lora" in by_strategy:
        fixed = by_strategy["spq_like_rsq_no_lora"]
        guided = by_strategy.get("hessian_guided_spq_no_lora")
        lines.extend(["", "## SPQ-Like Recipe Evidence", ""])
        lines.append(
            f"- Fixed SPQ-like no-LoRA uses attention R+Q, MLP S+Q, and Q-only for other selected linear modules with q=rtn, s={args.spq_s_method}, r={args.spq_r_method}."
        )
        if guided:
            lines.append(
                f"- No-LoRA comparison: fixed SPQ-like PPL={float(fixed['perplexity']):.4f}, Hessian-guided-SPQ PPL={float(guided['perplexity']):.4f}; "
                f"delta guided-fixed={float(guided['perplexity']) - float(fixed['perplexity']):.4f}. Both use the same nominal bits/keep/rank budget."
            )
        fixed_lora = by_strategy.get("spq_like_rsq_lora")
        guided_lora = by_strategy.get("hessian_guided_spq_lora")
        if fixed_lora and guided_lora:
            lines.append(
                f"- LoRA-recovered comparison ({args.spq_lora_steps} steps, rank {args.spq_lora_rank}): fixed SPQ-like PPL={float(fixed_lora['perplexity']):.4f}, "
                f"Hessian-guided-SPQ PPL={float(guided_lora['perplexity']):.4f}; delta guided-fixed={float(guided_lora['perplexity']) - float(fixed_lora['perplexity']):.4f}."
            )
        lines.append(
            "- `metrics/spq_recipe_diagnostics.csv` records the SPQ-applicable pair rho_H, fixed/reversed predicted Hessian costs, and Hessian-guided order/method choices per layer."
        )
    lines.extend(
        [
            "",
            "## Method-Coverage Notes",
            "",
            "This run is a pretrained-LLM framework experiment, not a claim that the native script reimplements every external baseline.",
            "Text provenance is recorded in `metrics/text_source_metadata.csv`; zero-shot additivity and strategy evaluations use the same per-task example limit so degradation correlations are comparable.",
            "When `--include-fair-benchmark` is enabled, fair benchmark zero-shot scores are in `metrics/fair_benchmark_zero_shot.csv` even if top-level strategy zero-shot was disabled.",
        ]
    )
    if unavailable:
        lines.append("Unavailable external baselines in this environment:")
        for row in unavailable:
            lines.append(f"- {row['component']}/{row['method']}: {row['reason']}")
    lines.extend(
        [
            "",
            "Native baselines included: RTN quantization, Hadamard rotated RTN proxy, magnitude pruning, Wanda-style activation-aware pruning, vanilla SVD, and activation-whitened SVD proxy.",
            "The `slim_like_srq_proxy` row is a fixed triple-compression recipe proxy; it is not the official SLiM implementation.",
            "",
            "## Artifacts",
            "",
            "- `metrics/hessian_cosine.csv` and `figures/hessian_cosine_heatmap.png`",
            "- `metrics/additivity.csv`, `metrics/order_gap.csv`, `metrics/correlations.csv`",
            "- `metrics/strategy_performance.csv`, `metrics/layerwise_selection.csv`, `metrics/method_status.csv`",
            "- `metrics/spq_recipe_diagnostics.csv` when `--include-spq-strategies` is enabled",
            "- `metrics/rotation_quantization.csv` and `figures/rotation_quantization_summary.png` when `--include-rotation-analysis` is enabled",
            "- `metrics/low_loss_triple_candidates.csv` when `--include-low-loss-triple` is enabled",
            "- `metrics/lossless_frontier_candidates.csv`, `metrics/lossless_frontier_summary.csv`, and `figures/lossless_frontier_summary.png` when `--include-lossless-frontier` is enabled",
            "- `metrics/fair_benchmark.csv`, `metrics/fair_benchmark_zero_shot.csv`, `metrics/fair_benchmark_selection.csv`, and `figures/fair_benchmark_summary.png` when `--include-fair-benchmark` is enabled",
            "- `figures/pretrained_goal_dashboard.png`",
            "- `figures/largest_order_gap_singular_spectrum.png`",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compression-orthogonality metrics on a pretrained small LLM.")
    parser.add_argument("--mode", choices=["full", "residual_stack_validate"], default="full")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--module-types", default="dense_h_to_4h,dense_4h_to_h")
    parser.add_argument("--layer-positions", default="first,middle,last")
    parser.add_argument("--layers", default="")
    parser.add_argument("--max-modules", type=int, default=6)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--rank-fraction", type=float, default=0.5)
    parser.add_argument("--q-method", default="rtn")
    parser.add_argument("--s-method", default="wanda")
    parser.add_argument("--r-method", default="whitened_svd")
    parser.add_argument("--guided-q-methods", default="rtn", help="Q-method candidates for Hessian-guided layer-wise selection.")
    parser.add_argument("--svd-device", default="auto")
    parser.add_argument("--calib-limit", type=int, default=8)
    parser.add_argument("--eval-limit", type=int, default=8)
    parser.add_argument("--disjoint-text-splits", action="store_true", help="Use non-overlapping text windows for Hessian calibration, PPL evaluation, and LoRA recovery training.")
    parser.add_argument("--texts-per-batch-window", type=int, default=8, help="Text rows reserved per requested token batch when --disjoint-text-splits is enabled.")
    parser.add_argument("--zero-shot-tasks", default="arc_easy,hellaswag")
    parser.add_argument("--zero-shot-additivity-limit", type=int)
    parser.add_argument("--zero-shot-strategy-limit", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--subset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--backup-name", default="wikitext_2_raw")
    parser.add_argument("--text-source", choices=["auto", "dataset", "zero_shot_backup", "fallback"], default="auto")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--include-rotation-analysis", action="store_true", help="Evaluate Hadamard rotated RTN diagnostics and a fixed_qsr_rotated_q strategy.")
    parser.add_argument("--include-spq-strategies", action="store_true", help="Evaluate SPQ-like fixed and Hessian-guided-SPQ strategies.")
    parser.add_argument("--spq-s-method", default="wanda", choices=["magnitude", "wanda"], help="Structured-pruning method for fixed SPQ-like MLP modules.")
    parser.add_argument("--spq-r-method", default="svd", choices=["svd", "whitened_svd"], help="Low-rank method for fixed SPQ-like attention modules.")
    parser.add_argument("--spq-guided-q-methods", default="rtn", help="Candidate Q methods for Hessian-guided-SPQ.")
    parser.add_argument("--spq-guided-s-methods", default="magnitude,wanda", help="Candidate S methods for Hessian-guided-SPQ under the same keep-fraction budget.")
    parser.add_argument("--spq-guided-r-methods", default="svd,whitened_svd", help="Candidate R methods for Hessian-guided-SPQ under the same rank-fraction budget.")
    parser.add_argument("--spq-lora-steps", type=int, default=0, help="If >0, evaluate LoRA-recovered fixed SPQ-like and Hessian-guided-SPQ strategies.")
    parser.add_argument("--spq-lora-rank", type=int, default=8)
    parser.add_argument("--spq-lora-alpha", type=float, default=32.0)
    parser.add_argument("--spq-lora-lr", type=float, default=5e-5)
    parser.add_argument("--spq-lora-train-limit", type=int, default=8, help="Number of token batches per LoRA recovery epoch over the run text source.")
    parser.add_argument("--include-low-loss-triple", action="store_true", help="Evaluate a conservative Q+S+R stack targeting benchmark drop below the configured threshold.")
    parser.add_argument("--low-loss-bits-list", default="8,6,4")
    parser.add_argument("--low-loss-keep-list", default="0.995,0.99,0.98,0.95")
    parser.add_argument("--low-loss-rank-list", default="0.995,0.99,0.98,0.95")
    parser.add_argument("--low-loss-q-methods", default="rotated_rtn,rtn")
    parser.add_argument("--low-loss-s-methods", default="wanda")
    parser.add_argument("--low-loss-r-methods", default="whitened_svd,svd")
    parser.add_argument("--low-loss-orders", default="qsr,qrs,sqr,srq,rqs,rsq")
    parser.add_argument("--low-loss-max-candidates", type=int, default=24)
    parser.add_argument("--include-lossless-frontier", action="store_true", help="Evaluate matched Q-only/S-only/R-only/Q+S+R lossless frontier candidates.")
    parser.add_argument("--frontier-bits-list", default="8,6,4,3")
    parser.add_argument("--frontier-keep-list", default="0.995,0.99,0.98,0.95,0.9,0.8")
    parser.add_argument("--frontier-rank-list", default="0.995,0.99,0.98,0.95,0.9,0.8,0.5")
    parser.add_argument("--frontier-q-methods", default="rotated_rtn,rtn")
    parser.add_argument("--frontier-s-methods", default="wanda,magnitude")
    parser.add_argument("--frontier-r-methods", default="whitened_svd,svd")
    parser.add_argument("--frontier-orders", default="qsr,rqs")
    parser.add_argument("--frontier-max-triple-candidates", type=int, default=48)
    parser.add_argument("--include-fair-benchmark", action="store_true", help="Evaluate fixed non-selected Q/S/R/QSR benchmark configs with signed PPL and zero-shot metrics.")
    parser.add_argument("--include-fair-extended-recipes", action="store_true", help="Add SLiM-like and SPQ-like fixed/guided recipes to --include-fair-benchmark; LoRA rows require --spq-lora-steps > 0.")
    parser.add_argument("--fair-benchmark-zero-shot-limit", type=int, help="Per-task zero-shot example limit for --include-fair-benchmark; defaults to --zero-shot-strategy-limit.")
    parser.add_argument("--fair-benchmark-guided-q-methods", default="rtn,rotated_rtn")
    parser.add_argument("--fair-benchmark-guided-s-methods", default="magnitude,wanda")
    parser.add_argument("--fair-benchmark-guided-r-methods", default="svd,whitened_svd")
    parser.add_argument("--include-orthofilter-spq-refine", action="store_true", help="Add conditional-Hessian-filtered SPQ-prior refinement rows to --include-fair-benchmark; also includes the fixed SPQ/SLiM extended recipe comparison rows.")
    parser.add_argument("--orthofilter-include-residual-candidates", action="store_true", help="Also let orthofilter-SPQ choose residual low-rank/sparse compensation candidates; reported memory uses additive component accounting.")
    parser.add_argument("--selector-activation-sample-rows", type=int, default=512, help="Max calibration activation rows per selected module for selector activation and worst-token risk.")
    parser.add_argument("--orthofilter-rho-threshold", type=float, default=0.25, help="Reject candidates whose positive conditional Hessian rho exceeds this threshold unless every candidate is rejected.")
    parser.add_argument("--orthofilter-hessian-weight", type=float, default=0.25)
    parser.add_argument("--orthofilter-activation-weight", type=float, default=1.0)
    parser.add_argument("--orthofilter-worst-token-weight", type=float, default=0.25)
    parser.add_argument("--orthofilter-conflict-weight", type=float, default=0.5)
    parser.add_argument("--orthofilter-memory-weight", type=float, default=0.05)
    parser.add_argument("--orthofilter-zero-shot-proxy-weight", type=float, default=0.2)
    parser.add_argument("--residual-stack-memory-targets", default="0.258,0.300,0.350", help="Nominal additive-memory budgets for residual-space validation.")
    parser.add_argument("--residual-stack-eval-budget", type=float, default=0.0, help="Budget evaluated with PPL/zero-shot; default uses the first feasible target.")
    parser.add_argument("--residual-stack-splits", default="0.25,0.50,0.75", help="Sparse share of residual budget for Q+S_res+L_res candidates.")
    parser.add_argument("--residual-stack-q-methods", default="rtn", help="Q bases for residual-stack candidates: rtn, rotated_rtn, sinq_like.")
    parser.add_argument("--residual-stack-s-methods", default="wanda,magnitude")
    parser.add_argument("--residual-stack-r-methods", default="whitened_svd,svd")
    parser.add_argument("--residual-stack-rho-threshold", type=float, default=0.30, help="Positive conditional Hessian rho filter threshold.")
    parser.add_argument("--residual-stack-activation-weight", type=float, default=1.0)
    parser.add_argument("--residual-stack-worst-token-weight", type=float, default=0.5)
    parser.add_argument("--residual-stack-hessian-weight", type=float, default=0.2)
    parser.add_argument("--residual-stack-include-order-gap", action="store_true", help="Also generate Q+L_res+S_res candidates for residual order-gap diagnostics.")
    parser.add_argument("--residual-stack-zero-shot-limit", type=int, help="Per-task zero-shot limit for residual_stack_validate; defaults to --zero-shot-strategy-limit.")
    parser.add_argument("--include-dam-comparison", action="store_true", help="Add paper-style low-rank-before-quantization and DAM-like factor quantization baselines to residual_stack_validate.")
    parser.add_argument("--dam-alpha-grid", default="0.0,0.25,0.5,0.75,1.0", help="Diagonal scaling exponents evaluated by the activation-grid DAM proxy.")
    parser.add_argument("--lossless-benchmark-metric", choices=["zero_shot", "ppl", "loss"], default="zero_shot")
    parser.add_argument("--lossless-benchmark-drop-threshold", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    args.module_types = parse_csv(args.module_types, [])
    args.layer_positions = parse_csv(args.layer_positions, ["first", "middle", "last"])
    args.layers = parse_int_csv(args.layers)
    args.zero_shot_tasks = parse_csv(args.zero_shot_tasks, [])
    if args.mode == "residual_stack_validate" and args.residual_stack_zero_shot_limit is not None:
        args.zero_shot_strategy_limit = int(args.residual_stack_zero_shot_limit)
    if args.zero_shot_additivity_limit is None:
        args.zero_shot_additivity_limit = args.zero_shot_strategy_limit
    args.data_cfg = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "backup_name": args.backup_name,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "allow_fallback": args.allow_fallback,
    }
    if args.disjoint_text_splits:
        recovery_batches = max(int(args.spq_lora_train_limit), int(args.spq_lora_steps), 1)
        text_pool_limit = max(int(args.calib_limit) + int(args.eval_limit) + recovery_batches, 1) * int(args.texts_per_batch_window)
    else:
        text_pool_limit = max(args.eval_limit, args.calib_limit) * int(args.texts_per_batch_window)
    text_pool, args.text_source_used, args.text_source_metadata = load_eval_texts(args, limit=text_pool_limit)
    args.text_pool_count = len(text_pool)
    split_text_windows(args, text_pool)
    model_id = args.model.replace("/", "_").replace(":", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_prefix = "residual_stack_validate" if args.mode == "residual_stack_validate" else "pretrained_orthogonality"
    root = Path(args.output_dir) if args.output_dir else Path("results") / f"{run_prefix}_{model_id}_{timestamp}"
    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = {
        "model": args.model,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    model, tokenizer, device = load_model_and_tokenizer_from_config(model_cfg)
    args.device = device
    if args.svd_device == "auto":
        args.svd_device = "cuda" if torch.cuda.is_available() else "cpu"

    modules = discover_target_linears(
        model,
        module_types=args.module_types,
        layer_positions=args.layer_positions,
        layers=args.layers,
        max_modules=args.max_modules,
    )
    args.selected_layers = list(modules)
    baseline_weights = clone_weights(modules)
    model_parameter_count = int(sum(param.numel() for param in model.parameters()))
    trainable_parameter_count = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    write_json(
        metrics_dir / "run_config.json",
        {
            "model": args.model,
            "mode": args.mode,
            "model_parameter_count": model_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "device": args.device,
            "svd_device": args.svd_device,
            "selected_layers": args.selected_layers,
            "runtime_args": args_snapshot(args),
            "versions": {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "datasets": package_version("datasets"),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "data": args.data_cfg,
            "text_source_requested": args.text_source,
            "text_source_used": args.text_source_used,
            "text_source_metadata": args.text_source_metadata,
            "text_split_policy": args.text_split_policy,
            "text_pool_count": args.text_pool_count,
            "calib_text_count": len(args.calib_texts),
            "eval_text_count": len(args.eval_texts),
            "recovery_text_count": len(args.recovery_texts),
            "compression": {
                "bits": args.bits,
                "keep_fraction": args.keep_fraction,
                "rank_fraction": args.rank_fraction,
                "q_method": args.q_method,
                "s_method": args.s_method,
                "r_method": args.r_method,
            },
            "orthofilter_selector": {
                "enabled_in_fair_benchmark": args.include_orthofilter_spq_refine,
                "include_residual_candidates": args.orthofilter_include_residual_candidates,
                "rho_threshold": args.orthofilter_rho_threshold,
                "activation_sample_rows": args.selector_activation_sample_rows,
                "weights": {
                    "hessian": args.orthofilter_hessian_weight,
                    "activation": args.orthofilter_activation_weight,
                    "worst_token": args.orthofilter_worst_token_weight,
                    "conflict": args.orthofilter_conflict_weight,
                    "memory": args.orthofilter_memory_weight,
                    "zero_shot_proxy": args.orthofilter_zero_shot_proxy_weight,
                },
            },
            "residual_stack_validate": {
                "enabled": args.mode == "residual_stack_validate",
                "memory_targets": args.residual_stack_memory_targets,
                "eval_budget": args.residual_stack_eval_budget,
                "splits": args.residual_stack_splits,
                "q_methods": args.residual_stack_q_methods,
                "s_methods": args.residual_stack_s_methods,
                "r_methods": args.residual_stack_r_methods,
                "rho_threshold": args.residual_stack_rho_threshold,
                "include_order_gap": args.residual_stack_include_order_gap,
                "include_dam_comparison": args.include_dam_comparison,
                "dam_alpha_grid": args.dam_alpha_grid,
                "score_weights": {
                    "activation": args.residual_stack_activation_weight,
                    "worst_token_p95": args.residual_stack_worst_token_weight,
                    "hessian": args.residual_stack_hessian_weight,
                },
            },
        },
    )
    write_csv(metrics_dir / "text_source_metadata.csv", args.text_source_metadata)

    baseline_metrics = evaluate_current_model(
        model,
        tokenizer,
        texts=args.eval_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        eval_limit=args.eval_limit,
    )
    baseline_zero_shot, baseline_zero_rows = evaluate_zero_shot_mean(
        model,
        tokenizer,
        tasks=args.zero_shot_tasks,
        limit=args.zero_shot_strategy_limit,
        device=device,
    )
    write_csv(metrics_dir / "zero_shot_baseline.csv", [{"scope": "baseline", **row} for row in baseline_zero_rows])

    covariances, activation_counts = collect_activation_covariances(
        model,
        tokenizer,
        modules,
        texts=args.calib_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
    )
    write_csv(
        metrics_dir / "activation_covariance_summary.csv",
        [
            {
                "layer": name,
                "layer_short": short_layer_name(name),
                "activation_rows": activation_counts[name],
                "input_dim": covariances[name].shape[0],
                "trace": float(torch.trace(covariances[name]).item()),
                "mean_diag": float(torch.diag(covariances[name]).mean().item()),
            }
            for name in modules
        ],
    )
    activation_samples = collect_activation_samples(
        model,
        tokenizer,
        modules,
        texts=args.calib_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
        max_rows=args.selector_activation_sample_rows,
    )
    write_csv(
        metrics_dir / "activation_sample_summary.csv",
        [
            {
                "layer": name,
                "layer_short": short_layer_name(name),
                "activation_sample_rows": int(samples.shape[0]),
                "input_dim": int(samples.shape[1]) if samples.ndim == 2 else 0,
            }
            for name, samples in activation_samples.items()
        ],
    )
    if args.mode == "residual_stack_validate":
        run_residual_stack_validate(
            root=root,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            baseline_weights=baseline_weights,
            covariances=covariances,
            activation_samples=activation_samples,
            baseline_metrics=baseline_metrics,
            baseline_zero_shot=baseline_zero_shot,
            args=args,
        )
        print(root)
        return

    methods = {"q": args.q_method, "s": args.s_method, "r": args.r_method}
    _, deltas = build_base_deltas(baseline_weights, covariances, methods, args)
    method_status = method_status_rows()
    heatmap = hessian_heatmap_rows(deltas, covariances, figures_dir)
    additivity, additivity_zero_rows = additivity_rows(
        model,
        tokenizer,
        modules,
        baseline_weights,
        baseline_metrics,
        baseline_zero_shot,
        deltas,
        covariances,
        args,
    )
    order_gap = order_gap_rows(model, tokenizer, modules, baseline_weights, methods, covariances, baseline_metrics, args)
    correlations = correlation_rows(additivity, order_gap)
    strategy, selection, strategy_zero_rows, low_loss_rows = evaluate_strategies(
        model,
        tokenizer,
        modules,
        baseline_weights,
        covariances,
        baseline_metrics,
        baseline_zero_shot,
        args,
    )
    if args.include_spq_strategies:
        spq_methods = {"q": "rtn", "s": args.spq_s_method, "r": args.spq_r_method}
        spq_fixed_repl = spq_like_replacements(baseline_weights, covariances, spq_methods, args)
        spq_selection = [row for row in selection if str(row.get("selection_family", "")) == "hessian_guided_spq"]
        spq_diagnostics = spq_recipe_diagnostic_rows(
            baseline_weights,
            covariances,
            deltas,
            spq_fixed_repl,
            spq_selection,
            spq_methods,
            args,
        )
    else:
        spq_diagnostics = []
    if args.include_rotation_analysis:
        rotation_quantization = rotation_quantization_rows(baseline_weights, covariances, args)
    else:
        rotation_quantization = []
    if args.include_lossless_frontier:
        lossless_frontier_rows, lossless_frontier_summary, lossless_frontier_zero_rows = evaluate_lossless_frontier(
            model,
            tokenizer,
            modules,
            baseline_weights,
            covariances,
            baseline_metrics,
            baseline_zero_shot,
            args,
        )
    else:
        lossless_frontier_rows = []
        lossless_frontier_summary = []
        lossless_frontier_zero_rows = []
    if args.include_fair_benchmark:
        fair_benchmark_rows, fair_benchmark_zero_rows, fair_benchmark_selection_rows = evaluate_fair_benchmark(
            model,
            tokenizer,
            modules,
            baseline_weights,
            covariances,
            activation_samples,
            baseline_metrics,
            args,
        )
    else:
        fair_benchmark_rows = []
        fair_benchmark_zero_rows = []
        fair_benchmark_selection_rows = []

    write_csv(metrics_dir / "method_status.csv", method_status)
    write_csv(metrics_dir / "hessian_cosine.csv", heatmap)
    write_csv(metrics_dir / "additivity.csv", additivity)
    write_csv(metrics_dir / "order_gap.csv", order_gap)
    write_csv(metrics_dir / "correlations.csv", correlations)
    write_csv(metrics_dir / "strategy_performance.csv", strategy)
    write_csv(metrics_dir / "layerwise_selection.csv", selection)
    write_csv(metrics_dir / "spq_recipe_diagnostics.csv", spq_diagnostics)
    write_csv(metrics_dir / "rotation_quantization.csv", rotation_quantization)
    write_csv(metrics_dir / "low_loss_triple_candidates.csv", low_loss_rows)
    write_csv(metrics_dir / "lossless_frontier_candidates.csv", lossless_frontier_rows)
    write_csv(metrics_dir / "lossless_frontier_summary.csv", lossless_frontier_summary)
    write_csv(metrics_dir / "fair_benchmark.csv", fair_benchmark_rows)
    write_csv(metrics_dir / "fair_benchmark_zero_shot.csv", fair_benchmark_zero_rows)
    write_csv(metrics_dir / "fair_benchmark_selection.csv", fair_benchmark_selection_rows)
    write_csv(metrics_dir / "zero_shot.csv", additivity_zero_rows + strategy_zero_rows + lossless_frontier_zero_rows + fair_benchmark_zero_rows)
    plot_dashboard(figures_dir, additivity, order_gap, correlations, strategy)
    plot_singular_spectrum(figures_dir, order_gap, baseline_weights, covariances, methods, args)
    plot_rotation_quantization(figures_dir, rotation_quantization)
    plot_lossless_frontier(figures_dir, lossless_frontier_summary)
    plot_fair_benchmark(figures_dir, fair_benchmark_rows)
    write_report(root, args, baseline_metrics, baseline_zero_shot, additivity, order_gap, correlations, strategy, method_status)
    restore_weights(modules, baseline_weights)
    print(root)


if __name__ == "__main__":
    main()
