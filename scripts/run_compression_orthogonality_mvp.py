from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_spectral_dynamics.structured.orthogonality import (
    empirical_additivity_error,
    hessian_cosine,
    hessian_inner,
    parameter_cosine,
    spearmanr,
    spectrum_summary,
)


CORPUS = """
compression orthogonality asks whether quantization sparsity and low rank updates
damage the same directions of a trained model. hessian cross terms should predict
when two perturbations are complementary and when they conflict. order matters
because quantization and pruning reshape the singular spectrum before a low rank
projection is applied. a useful framework must connect rho h additivity error
order gap perplexity and accuracy degradation rather than only drawing landscapes.
"""


class TinyCharLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(seq_len * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids).reshape(input_ids.shape[0], -1)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.head(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(seq_len: int, repeat: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    text = " ".join(CORPUS.lower().split())
    text = ((text + " ") * int(repeat)).strip()
    alphabet = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(alphabet)}
    ids = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for index in range(0, ids.numel() - seq_len):
        xs.append(ids[index : index + seq_len])
        ys.append(ids[index + seq_len])
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y, stoi


def batch_iter(x: torch.Tensor, y: torch.Tensor, batch_size: int, batches: int | None = None, *, shuffle: bool = False):
    n = x.shape[0]
    if shuffle:
        while True:
            index = torch.randint(0, n, (batch_size,), device=x.device)
            yield x.index_select(0, index), y.index_select(0, index)
    else:
        total = math.ceil(n / batch_size) if batches is None else int(batches)
        for batch in range(total):
            start = (batch * batch_size) % n
            end = min(start + batch_size, n)
            if batches is not None and end - start < batch_size and n >= batch_size:
                start = n - batch_size
                end = n
            yield x[start:end], y[start:end]


def train_model(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, args: argparse.Namespace) -> list[dict[str, float | int]]:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows: list[dict[str, float | int]] = []
    iterator = batch_iter(train_x, train_y, args.batch_size, shuffle=True)
    model.train()
    for step in range(1, args.train_steps + 1):
        xb, yb = next(iterator)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            rows.append({"step": step, "train_loss": float(loss.detach().cpu())})
    return rows


@torch.no_grad()
def evaluate_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, batches: int | None) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    eval_batches = None if batches is None or int(batches) <= 0 else int(batches)
    for xb, yb in batch_iter(x, y, batch_size, eval_batches):
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss.detach().cpu())
        pred = torch.argmax(logits, dim=-1)
        total_correct += int((pred == yb).sum().detach().cpu())
        total += int(yb.numel())
    mean_loss = total_loss / max(total, 1)
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 50.0))),
        "accuracy": float(total_correct / max(total, 1)),
        "examples": int(total),
    }


def selected_linears(model: nn.Module) -> dict[str, nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}


def clone_linear_weights(linears: dict[str, nn.Linear]) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().clone() for name, module in linears.items()}


def restore_linear_weights(linears: dict[str, nn.Linear], baseline: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in linears.items():
            module.weight.copy_(baseline[name])


def evaluate_with_layer_weights(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    restore_linear_weights(linears, baseline_weights)
    with torch.no_grad():
        for name, weight in replacements.items():
            linears[name].weight.copy_(weight)
    metrics = evaluate_model(model, x, y, args.eval_batch_size, args.eval_batches)
    restore_linear_weights(linears, baseline_weights)
    return metrics


def symmetric_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = max(1, 2 ** (int(bits) - 1) - 1)
    scale = torch.amax(torch.abs(weight), dim=1, keepdim=True) / qmax
    scale = torch.clamp(scale, min=1e-12)
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def magnitude_prune(weight: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    keep = max(1, min(weight.numel(), int(round(float(keep_fraction) * weight.numel()))))
    flat = torch.abs(weight).reshape(-1)
    threshold = torch.topk(flat, keep, largest=True).values[-1]
    return torch.where(torch.abs(weight) >= threshold, weight, torch.zeros_like(weight))


def low_rank_project(weight: torch.Tensor, rank_fraction: float) -> torch.Tensor:
    rows, cols = weight.shape
    rank = max(1, min(rows, cols, int(round(float(rank_fraction) * min(rows, cols)))))
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vh[:rank, :]).to(dtype=weight.dtype)


def apply_op(weight: torch.Tensor, op: str, *, bits: int, keep_fraction: float, rank_fraction: float) -> torch.Tensor:
    if op == "q":
        return symmetric_quantize(weight, bits)
    if op == "s":
        return magnitude_prune(weight, keep_fraction)
    if op == "r":
        return low_rank_project(weight, rank_fraction)
    raise ValueError(f"unknown compression op: {op}")


def apply_order(weight: torch.Tensor, order: tuple[str, ...], *, bits: int, keep_fraction: float, rank_fraction: float) -> torch.Tensor:
    out = weight
    for op in order:
        out = apply_op(out, op, bits=bits, keep_fraction=keep_fraction, rank_fraction=rank_fraction)
    return out


def estimate_hessian_diag(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    x: torch.Tensor,
    y: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    accum = {name: torch.zeros_like(module.weight, dtype=torch.float32) for name, module in linears.items()}
    model.train()
    for xb, yb in batch_iter(x, y, args.calib_batch_size, args.calib_batches):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        for name, module in linears.items():
            if module.weight.grad is not None:
                accum[name] += module.weight.grad.detach().float().pow(2)
    model.zero_grad(set_to_none=True)
    for name in accum:
        accum[name] = accum[name] / max(int(args.calib_batches), 1)
    return accum


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


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


def rho_heatmap_rows(
    baseline_weights: dict[str, torch.Tensor],
    hessian_diag: dict[str, torch.Tensor],
    args: argparse.Namespace,
    figures_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, torch.Tensor]]]:
    rows: list[dict[str, object]] = []
    base_deltas: dict[str, dict[str, torch.Tensor]] = {}
    labels = ["q", "s", "r"]
    matrices: list[np.ndarray] = []
    matrix_titles: list[str] = []
    for layer, weight in baseline_weights.items():
        deltas = {
            "q": apply_op(weight, "q", bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction) - weight,
            "s": apply_op(weight, "s", bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction) - weight,
            "r": apply_op(weight, "r", bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction) - weight,
        }
        base_deltas[layer] = deltas
        h = to_numpy(hessian_diag[layer])
        matrix = np.eye(3, dtype=np.float64)
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                value = hessian_cosine(to_numpy(deltas[left]), to_numpy(deltas[right]), h)
                matrix[i, j] = value
                rows.append({"layer": layer, "left": left, "right": right, "rho_h": value})
        matrices.append(matrix)
        matrix_titles.append(layer)

    aggregate = np.mean(np.stack(matrices, axis=0), axis=0)
    matrices.insert(0, aggregate)
    matrix_titles.insert(0, "mean")
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.1 * len(matrices), 3.2), squeeze=False)
    for ax, matrix, title in zip(axes[0], matrices, matrix_titles):
        im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(3), labels=labels)
        ax.set_yticks(range(3), labels=labels)
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Hessian cosine")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "hessian_cosine_heatmap.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / "hessian_cosine_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows, base_deltas


def parse_range_spec(value: str) -> tuple[np.ndarray, float, float, int]:
    parts = [float(part) for part in str(value).split(":")]
    if len(parts) != 3:
        raise ValueError("range specs must use start:end:num, for example -0.5:1.5:17")
    start, end, count = parts
    count_i = int(count)
    if count_i < 2:
        raise ValueError("range specs need at least two samples")
    return np.linspace(start, end, count_i), float(start), float(end), count_i


def params_from_level(op: str, level: str, args: argparse.Namespace) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "bits": args.q_bits,
        "keep_fraction": args.keep_fraction,
        "rank_fraction": args.rank_fraction,
    }
    text = str(level)
    if op == "q" and text.startswith("bits"):
        params["bits"] = int(text.removeprefix("bits"))
    elif op == "s" and text.startswith("keep"):
        params["keep_fraction"] = float(text.removeprefix("keep"))
    elif op == "r" and text.startswith("rank"):
        params["rank_fraction"] = float(text.removeprefix("rank"))
    return params


def compression_loss_landscape(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    add_rows: list[dict[str, object]],
    args: argparse.Namespace,
    metrics_dir: Path,
    figures_dir: Path,
) -> list[dict[str, object]]:
    if args.skip_landscape or not add_rows:
        return []
    selected = max(add_rows, key=lambda row: float(row["abs_rho_h"]))
    layer = str(selected["layer"])
    pair = str(selected["pair"])
    left_op, right_op = pair[0], pair[1]
    left_level = str(selected["left_level"])
    right_level = str(selected["right_level"])
    left_params = params_from_level(left_op, left_level, args)
    right_params = params_from_level(right_op, right_level, args)
    weight = baseline_weights[layer]
    left_delta = apply_op(weight, left_op, **left_params) - weight
    right_delta = apply_op(weight, right_op, **right_params) - weight
    xs, _, _, _ = parse_range_spec(args.landscape_x)
    ys, _, _, _ = parse_range_spec(args.landscape_y)
    rows: list[dict[str, object]] = []
    z = np.zeros((len(ys), len(xs)), dtype=np.float64)
    for y_idx, beta in enumerate(ys):
        for x_idx, alpha in enumerate(xs):
            candidate = weight + float(alpha) * left_delta + float(beta) * right_delta
            metrics = evaluate_with_layer_weights(model, linears, baseline_weights, {layer: candidate}, val_x, val_y, args)
            z[y_idx, x_idx] = float(metrics["loss"])
            rows.append(
                {
                    "layer": layer,
                    "pair": pair,
                    "x_op": left_op,
                    "x_level": left_level,
                    "y_op": right_op,
                    "y_level": right_level,
                    "x": float(alpha),
                    "y": float(beta),
                    "loss": metrics["loss"],
                    "perplexity": metrics["perplexity"],
                    "accuracy": metrics["accuracy"],
                }
            )
    write_csv(metrics_dir / f"loss_landscape_{layer}_{pair}.csv", rows)
    x_mesh, y_mesh = np.meshgrid(xs, ys)
    contour_base = f"loss_landscape_{layer}_{pair}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    filled = ax.contourf(x_mesh, y_mesh, z, levels=20, cmap="viridis")
    lines = ax.contour(x_mesh, y_mesh, z, levels=10, colors="white", linewidths=0.6, alpha=0.75)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.3g")
    ax.scatter([0, 1, 0, 1], [0, 0, 1, 1], c=["white", "#f97316", "#38bdf8", "#ef4444"], edgecolors="black", s=42, zorder=3)
    ax.text(0, 0, " W", va="bottom", ha="left", fontsize=8)
    ax.text(1, 0, f" {left_op}:{left_level}", va="bottom", ha="left", fontsize=8)
    ax.text(0, 1, f" {right_op}:{right_level}", va="bottom", ha="left", fontsize=8)
    ax.text(1, 1, f" {pair}", va="bottom", ha="left", fontsize=8)
    ax.set_xlabel(f"alpha along Delta_{left_op} ({left_level})")
    ax.set_ylabel(f"beta along Delta_{right_op} ({right_level})")
    ax.set_title(f"Loss contour on {layer}: Delta_{left_op} / Delta_{right_op}")
    fig.colorbar(filled, ax=ax, label="validation loss")
    fig.savefig(figures_dir / f"{contour_base}_contour.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / f"{contour_base}_contour.pdf", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(5.8, 4.6))
    ax3d = fig.add_subplot(111, projection="3d")
    surface = ax3d.plot_surface(x_mesh, y_mesh, z, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.95)
    ax3d.set_xlabel(f"Delta_{left_op} ({left_level})")
    ax3d.set_ylabel(f"Delta_{right_op} ({right_level})")
    ax3d.set_zlabel("loss")
    ax3d.set_title(f"Loss surface on {layer}: {pair}")
    fig.colorbar(surface, ax=ax3d, shrink=0.65, label="validation loss")
    fig.savefig(figures_dir / f"{contour_base}_surface.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures_dir / f"{contour_base}_surface.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


def additivity_rows(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    hessian_diag: dict[str, torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    baseline_metrics: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    individual_cache: dict[tuple[str, str, str], tuple[torch.Tensor, dict[str, float], torch.Tensor]] = {}
    op_levels = {
        "q": [(f"bits{bits}", {"bits": int(bits), "keep_fraction": args.keep_fraction, "rank_fraction": args.rank_fraction}) for bits in args.q_bits_values],
        "s": [(f"keep{keep:g}", {"bits": args.q_bits, "keep_fraction": float(keep), "rank_fraction": args.rank_fraction}) for keep in args.keep_values],
        "r": [(f"rank{rank:g}", {"bits": args.q_bits, "keep_fraction": args.keep_fraction, "rank_fraction": float(rank)}) for rank in args.rank_values],
    }

    def individual(layer: str, op: str, level: tuple[str, dict[str, float | int]]):
        level_name, params = level
        key = (layer, op, level_name)
        if key not in individual_cache:
            weight = baseline_weights[layer]
            compressed = apply_op(weight, op, **params)
            delta = compressed - weight
            metrics = evaluate_with_layer_weights(model, linears, baseline_weights, {layer: compressed}, val_x, val_y, args)
            individual_cache[key] = (compressed, metrics, delta)
        return individual_cache[key]

    for layer, weight in baseline_weights.items():
        h = to_numpy(hessian_diag[layer])
        for left, right in [("q", "s"), ("q", "r"), ("s", "r")]:
            for left_level, right_level in zip(op_levels[left], op_levels[right]):
                _, left_metrics, left_delta = individual(layer, left, left_level)
                _, right_metrics, right_delta = individual(layer, right, right_level)
                pair_weight = weight + left_delta + right_delta
                pair_metrics = evaluate_with_layer_weights(model, linears, baseline_weights, {layer: pair_weight}, val_x, val_y, args)
                rho_h = hessian_cosine(to_numpy(left_delta), to_numpy(right_delta), h)
                param_cos = parameter_cosine(to_numpy(left_delta), to_numpy(right_delta))
                add_err = empirical_additivity_error(
                    baseline_metrics["loss"],
                    left_metrics["loss"],
                    right_metrics["loss"],
                    pair_metrics["loss"],
                )
                pred = (
                    0.5 * hessian_inner(to_numpy(left_delta), to_numpy(left_delta), h)
                    + 0.5 * hessian_inner(to_numpy(right_delta), to_numpy(right_delta), h)
                    + hessian_inner(to_numpy(left_delta), to_numpy(right_delta), h)
                )
                rows.append(
                    {
                        "layer": layer,
                        "pair": f"{left}{right}",
                        "left_level": left_level[0],
                        "right_level": right_level[0],
                        "rho_h": rho_h,
                        "abs_rho_h": abs(rho_h),
                        "parameter_cosine": param_cos,
                        "abs_parameter_cosine": abs(param_cos),
                        "frobenius_delta_sum": float(torch.linalg.vector_norm(left_delta).cpu() + torch.linalg.vector_norm(right_delta).cpu()),
                        "taylor_predicted_loss_delta": pred,
                        "loss_left": left_metrics["loss"],
                        "loss_right": right_metrics["loss"],
                        "loss_pair": pair_metrics["loss"],
                        "loss_degradation_pair": pair_metrics["loss"] - baseline_metrics["loss"],
                        "ppl_pair": pair_metrics["perplexity"],
                        "ppl_degradation_pair": pair_metrics["perplexity"] - baseline_metrics["perplexity"],
                        "accuracy_pair": pair_metrics["accuracy"],
                        "accuracy_degradation_pair": baseline_metrics["accuracy"] - pair_metrics["accuracy"],
                        "additivity_error": add_err,
                        "abs_additivity_error": abs(add_err),
                    }
                )
    return rows


def order_rows(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    hessian_diag: dict[str, torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    baseline_metrics: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[tuple[str, tuple[str, ...]], torch.Tensor], dict[tuple[str, tuple[str, ...]], dict[str, float]]]:
    rows: list[dict[str, object]] = []
    order_weight_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}
    order_metric_cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
    comparisons = [
        (("r", "q"), ("q", "r")),
        (("r", "s"), ("s", "r")),
        (("r", "q", "s"), ("q", "s", "r")),
        (("r", "s", "q"), ("s", "q", "r")),
    ]

    def order_weight(layer: str, order: tuple[str, ...]) -> torch.Tensor:
        key = (layer, order)
        if key not in order_weight_cache:
            order_weight_cache[key] = apply_order(
                baseline_weights[layer],
                order,
                bits=args.q_bits,
                keep_fraction=args.keep_fraction,
                rank_fraction=args.rank_fraction,
            )
        return order_weight_cache[key]

    def order_metrics(layer: str, order: tuple[str, ...]) -> dict[str, float]:
        key = (layer, order)
        if key not in order_metric_cache:
            order_metric_cache[key] = evaluate_with_layer_weights(
                model,
                linears,
                baseline_weights,
                {layer: order_weight(layer, order)},
                val_x,
                val_y,
                args,
            )
        return order_metric_cache[key]

    def conditional_overlap(layer: str, order: tuple[str, ...]) -> float:
        if len(order) < 2:
            return float("nan")
        weight = baseline_weights[layer]
        h = to_numpy(hessian_diag[layer])
        first_weight = apply_op(weight, order[0], bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction)
        second_weight = apply_op(first_weight, order[1], bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction)
        return hessian_cosine(to_numpy(first_weight - weight), to_numpy(second_weight - first_weight), h)

    for layer, weight in baseline_weights.items():
        original_spec = spectrum_summary(to_numpy(weight))
        for left_order, right_order in comparisons:
            left_weight = order_weight(layer, left_order)
            right_weight = order_weight(layer, right_order)
            left_metrics = order_metrics(layer, left_order)
            right_metrics = order_metrics(layer, right_order)
            first_left = apply_op(weight, left_order[0], bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction)
            left_cond_rho = conditional_overlap(layer, left_order)
            right_cond_rho = conditional_overlap(layer, right_order)
            max_abs_cond_rho = max(abs(left_cond_rho), abs(right_cond_rho))
            mean_abs_cond_rho = 0.5 * (abs(left_cond_rho) + abs(right_cond_rho))
            r_first_cond_rho = left_cond_rho if left_order[0] == "r" else right_cond_rho
            non_r_first_cond_rho = right_cond_rho if left_order[0] == "r" else left_cond_rho
            first_spec = spectrum_summary(to_numpy(first_left))
            row = {
                "layer": layer,
                "left_order": "".join(left_order),
                "right_order": "".join(right_order),
                "loss_left": left_metrics["loss"],
                "loss_right": right_metrics["loss"],
                "loss_gap_left_minus_right": left_metrics["loss"] - right_metrics["loss"],
                "abs_loss_gap": abs(left_metrics["loss"] - right_metrics["loss"]),
                "ppl_left": left_metrics["perplexity"],
                "ppl_right": right_metrics["perplexity"],
                "ppl_gap_left_minus_right": left_metrics["perplexity"] - right_metrics["perplexity"],
                "abs_ppl_gap": abs(left_metrics["perplexity"] - right_metrics["perplexity"]),
                "accuracy_left": left_metrics["accuracy"],
                "accuracy_right": right_metrics["accuracy"],
                "accuracy_gap_left_minus_right": left_metrics["accuracy"] - right_metrics["accuracy"],
                "abs_accuracy_gap": abs(left_metrics["accuracy"] - right_metrics["accuracy"]),
                "baseline_loss": baseline_metrics["loss"],
                "left_conditional_hessian_overlap": left_cond_rho,
                "right_conditional_hessian_overlap": right_cond_rho,
                "r_first_conditional_hessian_overlap": r_first_cond_rho,
                "abs_r_first_conditional_hessian_overlap": abs(r_first_cond_rho),
                "non_r_first_conditional_hessian_overlap": non_r_first_cond_rho,
                "abs_non_r_first_conditional_hessian_overlap": abs(non_r_first_cond_rho),
                "max_abs_conditional_hessian_overlap": max_abs_cond_rho,
                "mean_abs_conditional_hessian_overlap": mean_abs_cond_rho,
                "final_weight_disagreement": float(torch.linalg.vector_norm(left_weight - right_weight).cpu() / max(float(torch.linalg.vector_norm(weight).cpu()), 1e-12)),
            }
            for key in ["rank_90", "rank_95", "rank_99", "spectral_entropy", "top1_energy", "stable_rank"]:
                delta = float(first_spec[key]) - float(original_spec[key])
                row[f"first_{key}_delta"] = delta
                row[f"abs_first_{key}_delta"] = abs(delta)
            rows.append(row)
    return rows, order_weight_cache, order_metric_cache


def layerwise_selection_rows(
    model: nn.Module,
    linears: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    hessian_diag: dict[str, torch.Tensor],
    order_weight_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    baseline_metrics: dict[str, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    orders = list(itertools.permutations(("q", "s", "r")))
    fixed_order = tuple(args.fixed_order)
    selection_rows: list[dict[str, object]] = []
    selected_replacements: dict[str, torch.Tensor] = {}
    fixed_replacements: dict[str, torch.Tensor] = {}
    for layer, weight in baseline_weights.items():
        h = to_numpy(hessian_diag[layer])
        best_order = None
        best_cost = float("inf")
        fixed_weight = apply_order(weight, fixed_order, bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction)
        fixed_replacements[layer] = fixed_weight
        for order in orders:
            key = (layer, order)
            if key not in order_weight_cache:
                order_weight_cache[key] = apply_order(weight, order, bits=args.q_bits, keep_fraction=args.keep_fraction, rank_fraction=args.rank_fraction)
            final_weight = order_weight_cache[key]
            delta = final_weight - weight
            cost = 0.5 * hessian_inner(to_numpy(delta), to_numpy(delta), h)
            if (cost, "".join(order)) < (best_cost, "".join(best_order or ("z",))):
                best_cost = cost
                best_order = order
                selected_replacements[layer] = final_weight
        fixed_delta = fixed_weight - weight
        fixed_cost = 0.5 * hessian_inner(to_numpy(fixed_delta), to_numpy(fixed_delta), h)
        selection_rows.append(
            {
                "layer": layer,
                "selected_order": "".join(best_order or fixed_order),
                "fixed_order": "".join(fixed_order),
                "selected_predicted_hessian_cost": best_cost,
                "fixed_predicted_hessian_cost": fixed_cost,
                "predicted_cost_improvement": fixed_cost - best_cost,
            }
        )

    selected_metrics = evaluate_with_layer_weights(model, linears, baseline_weights, selected_replacements, val_x, val_y, args)
    fixed_metrics = evaluate_with_layer_weights(model, linears, baseline_weights, fixed_replacements, val_x, val_y, args)
    perf_rows = []
    for name, metrics in [("hessian_layerwise", selected_metrics), ("fixed_order", fixed_metrics)]:
        perf_rows.append(
            {
                "method": name,
                "q_bits": args.q_bits,
                "keep_fraction": args.keep_fraction,
                "rank_fraction": args.rank_fraction,
                "fixed_order": "".join(fixed_order),
                "loss": metrics["loss"],
                "perplexity": metrics["perplexity"],
                "accuracy": metrics["accuracy"],
                "loss_degradation": metrics["loss"] - baseline_metrics["loss"],
                "ppl_degradation": metrics["perplexity"] - baseline_metrics["perplexity"],
                "accuracy_degradation": baseline_metrics["accuracy"] - metrics["accuracy"],
            }
        )
    return selection_rows, perf_rows


def correlation_rows(add_rows: list[dict[str, object]], order_gap_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    specs = [
        ("additivity", add_rows, "abs_rho_h", "abs_additivity_error"),
        ("additivity", add_rows, "rho_h", "additivity_error"),
        ("real_ppl", add_rows, "abs_rho_h", "ppl_degradation_pair"),
        ("real_accuracy", add_rows, "abs_rho_h", "accuracy_degradation_pair"),
        ("taylor", add_rows, "taylor_predicted_loss_delta", "loss_degradation_pair"),
        ("frobenius_baseline", add_rows, "frobenius_delta_sum", "loss_degradation_pair"),
        ("param_cos_baseline", add_rows, "abs_parameter_cosine", "abs_additivity_error"),
        ("order_gap", order_gap_rows, "max_abs_conditional_hessian_overlap", "abs_loss_gap"),
        ("order_gap_ppl", order_gap_rows, "max_abs_conditional_hessian_overlap", "abs_ppl_gap"),
        ("order_gap_accuracy", order_gap_rows, "max_abs_conditional_hessian_overlap", "abs_accuracy_gap"),
        ("order_gap_r_first_overlap", order_gap_rows, "abs_r_first_conditional_hessian_overlap", "abs_loss_gap"),
        ("order_gap_non_r_first_overlap", order_gap_rows, "abs_non_r_first_conditional_hessian_overlap", "abs_loss_gap"),
        ("order_gap_mean_overlap", order_gap_rows, "mean_abs_conditional_hessian_overlap", "abs_loss_gap"),
        ("spectrum_order_rank90", order_gap_rows, "abs_first_rank_90_delta", "abs_loss_gap"),
        ("spectrum_order_entropy", order_gap_rows, "abs_first_spectral_entropy_delta", "abs_loss_gap"),
        ("spectrum_order_top1", order_gap_rows, "abs_first_top1_energy_delta", "abs_loss_gap"),
        ("spectrum_order_stable_rank", order_gap_rows, "abs_first_stable_rank_delta", "abs_loss_gap"),
        ("order_disagreement", order_gap_rows, "final_weight_disagreement", "abs_loss_gap"),
    ]
    rows = []
    for family, source, x_key, y_key in specs:
        x = [float(row.get(x_key, float("nan"))) for row in source]
        y = [float(row.get(y_key, float("nan"))) for row in source]
        rho, n = spearmanr(x, y)
        rows.append({"family": family, "x": x_key, "y": y_key, "spearman_rho": rho, "n": n})
    return rows


def write_report(
    path: Path,
    baseline_metrics: dict[str, float],
    correlations: list[dict[str, object]],
    selection_perf: list[dict[str, object]],
    add_rows: list[dict[str, object]],
    order_gap_rows: list[dict[str, object]],
) -> None:
    corr = {(row["family"], row["x"], row["y"]): row for row in correlations}
    layerwise = next(row for row in selection_perf if row["method"] == "hessian_layerwise")
    fixed = next(row for row in selection_perf if row["method"] == "fixed_order")
    highest = max(add_rows, key=lambda row: float(row["abs_rho_h"]))
    lowest = min(add_rows, key=lambda row: float(row["abs_rho_h"]))
    largest_gap = max(order_gap_rows, key=lambda row: float(row["abs_loss_gap"]))
    lines = [
        "# Compression Orthogonality MVP",
        "",
        f"- Baseline validation loss: {baseline_metrics['loss']:.4f}; PPL: {baseline_metrics['perplexity']:.4f}; accuracy: {baseline_metrics['accuracy']:.4f}; examples: {int(baseline_metrics['examples'])}.",
        f"- Highest |rho_H| additivity row: layer={highest['layer']}, pair={highest['pair']}, |rho_H|={highest['abs_rho_h']:.4f}, |A_ij|={highest['abs_additivity_error']:.4f}.",
        f"- Lowest |rho_H| additivity row: layer={lowest['layer']}, pair={lowest['pair']}, |rho_H|={lowest['abs_rho_h']:.4f}, |A_ij|={lowest['abs_additivity_error']:.4f}.",
        f"- Largest order gap: layer={largest_gap['layer']}, {largest_gap['left_order']} vs {largest_gap['right_order']}, loss gap={largest_gap['loss_gap_left_minus_right']:.4f}, R-first conditional |rho_H|={largest_gap['abs_r_first_conditional_hessian_overlap']:.4f}, max directional conditional |rho_H|={largest_gap['max_abs_conditional_hessian_overlap']:.4f}.",
        f"- Layer-wise Hessian selection PPL: {layerwise['perplexity']:.4f}; fixed-order PPL: {fixed['perplexity']:.4f}; same q/s/r settings are used for both.",
        "",
        "## Correlations",
        "",
    ]
    for row in correlations:
        rho = row["spearman_rho"]
        rho_text = "nan" if not math.isfinite(float(rho)) else f"{float(rho):.4f}"
        lines.append(f"- {row['family']}: Spearman({row['x']}, {row['y']}) = {rho_text} over n={row['n']}.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `metrics/hessian_cosine.csv` and `figures/hessian_cosine_heatmap.png`",
            "- `metrics/loss_landscape_<layer>_<pair>.csv` and `figures/loss_landscape_<layer>_<pair>_{contour,surface}.png`",
            "- `metrics/additivity.csv`",
            "- `metrics/order_gap.csv`",
            "- `metrics/layerwise_selection.csv` and `metrics/layerwise_performance.csv`",
            "- `metrics/correlations.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_numbers(value: str, cast=float) -> list:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimum viable Hessian cross-term compression experiment.")
    parser.add_argument("--output-dir", default="results/compression_orthogonality_mvp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--corpus-repeat", type=int, default=90)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=450)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=0, help="Number of validation batches; <=0 evaluates the full validation split.")
    parser.add_argument("--calib-batch-size", type=int, default=192)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--q-bits", type=int, default=3)
    parser.add_argument("--keep-fraction", type=float, default=0.62)
    parser.add_argument("--rank-fraction", type=float, default=0.50)
    parser.add_argument("--q-bits-values", default="2,3,4")
    parser.add_argument("--keep-values", default="0.45,0.62,0.78")
    parser.add_argument("--rank-values", default="0.30,0.50,0.70")
    parser.add_argument("--fixed-order", default="qsr")
    parser.add_argument("--landscape-x", default="-0.5:1.5:17")
    parser.add_argument("--landscape-y", default="-0.5:1.5:17")
    parser.add_argument("--skip-landscape", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.q_bits_values = parse_csv_numbers(args.q_bits_values, int)
    args.keep_values = parse_csv_numbers(args.keep_values, float)
    args.rank_values = parse_csv_numbers(args.rank_values, float)
    args.fixed_order = tuple(args.fixed_order.strip().lower())
    if sorted(args.fixed_order) != ["q", "r", "s"]:
        raise ValueError("--fixed-order must be a permutation of qsr")
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    metrics_dir = out_dir / "metrics"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, stoi = make_dataset(args.seq_len, args.corpus_repeat, device)
    split = int(0.85 * x.shape[0])
    train_x, train_y = x[:split], y[:split]
    val_x, val_y = x[split:], y[split:]
    model = TinyCharLM(len(stoi), args.seq_len, args.embed_dim, args.hidden_dim).to(device)
    train_rows = train_model(model, train_x, train_y, args)
    baseline_metrics = evaluate_model(model, val_x, val_y, args.eval_batch_size, args.eval_batches)

    linears = selected_linears(model)
    baseline_weights = clone_linear_weights(linears)
    hessian_diag = estimate_hessian_diag(model, linears, train_x, train_y, args)

    heat_rows, _ = rho_heatmap_rows(baseline_weights, hessian_diag, args, figures_dir)
    add_rows = additivity_rows(model, linears, baseline_weights, hessian_diag, val_x, val_y, baseline_metrics, args)
    order_gap, order_weight_cache, _ = order_rows(model, linears, baseline_weights, hessian_diag, val_x, val_y, baseline_metrics, args)
    selection_rows, selection_perf = layerwise_selection_rows(
        model,
        linears,
        baseline_weights,
        hessian_diag,
        order_weight_cache,
        val_x,
        val_y,
        baseline_metrics,
        args,
    )
    correlations = correlation_rows(add_rows, order_gap)
    compression_loss_landscape(model, linears, baseline_weights, val_x, val_y, add_rows, args, metrics_dir, figures_dir)

    write_csv(metrics_dir / "training.csv", train_rows)
    write_csv(metrics_dir / "baseline.csv", [{"metric": key, "value": value} for key, value in baseline_metrics.items()])
    write_csv(metrics_dir / "hessian_cosine.csv", heat_rows)
    write_csv(metrics_dir / "additivity.csv", add_rows)
    write_csv(metrics_dir / "order_gap.csv", order_gap)
    write_csv(metrics_dir / "layerwise_selection.csv", selection_rows)
    write_csv(metrics_dir / "layerwise_performance.csv", selection_perf)
    write_csv(metrics_dir / "correlations.csv", correlations)
    write_json(
        out_dir / "manifest.json",
        {
            "seed": args.seed,
            "device": str(device),
            "vocab_size": len(stoi),
            "num_train_examples": int(train_x.shape[0]),
            "num_val_examples": int(val_x.shape[0]),
            "eval_batches": int(args.eval_batches),
            "baseline_eval_examples": int(baseline_metrics["examples"]),
            "layers": list(linears),
            "q_bits": args.q_bits,
            "keep_fraction": args.keep_fraction,
            "rank_fraction": args.rank_fraction,
            "fixed_order": "".join(args.fixed_order),
        },
    )
    write_report(out_dir / "report.md", baseline_metrics, correlations, selection_perf, add_rows, order_gap)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
