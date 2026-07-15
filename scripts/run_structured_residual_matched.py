from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from llm_spectral_dynamics.structured.data import load_model_and_tokenizer_from_config, load_texts_from_config, token_batches
from llm_spectral_dynamics.structured.oasr import (
    activation_clustered_permutation,
    block_circulant_project,
    compute_activation_error,
    lowrank_project,
    monarch_like_two_block_project,
    norm_sorted_permutations,
    permuted_block_circulant_project,
    quantize_weight,
    relative_fro_error,
)


def parse_csv(value: str, default: list[str]) -> list[str]:
    if not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str, default: list[int]) -> list[int]:
    if not str(value).strip():
        return list(default)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def linear_layer_index(name: str) -> int | None:
    for pattern in (r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.block\.(\d+)\.", r"\.blocks\.(\d+)\."):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def short_layer_name(name: str) -> str:
    idx = linear_layer_index(name)
    suffix = name.rsplit(".", 1)[-1]
    return f"L{idx}:{suffix}" if idx is not None else suffix


def oasr_family_name(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".o_proj") or lower.endswith(".attention.dense") or ".self_attn.o_proj" in lower:
        return "attention_o"
    if lower.endswith(".down_proj") or lower.endswith(".dense_4h_to_h"):
        return "mlp_down"
    if lower.endswith(".up_proj") or lower.endswith(".dense_h_to_4h"):
        return "mlp_up"
    return "other"


def is_oasr_target(name: str, requested: set[str]) -> bool:
    family = oasr_family_name(name)
    if family in requested:
        return True
    suffix = name.rsplit(".", 1)[-1].lower()
    return suffix in requested


def discover_oasr_linears(model: nn.Module, *, target_types: list[str], max_layers: int) -> dict[str, nn.Linear]:
    requested = {item.strip().lower() for item in target_types if item.strip()}
    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.ndim == 2 and is_oasr_target(name, requested):
            candidates.append((name, module))
    candidates.sort(key=lambda item: (linear_layer_index(item[0]) if linear_layer_index(item[0]) is not None else 10**9, item[0]))
    if max_layers > 0:
        candidates = candidates[:max_layers]
    if not candidates:
        available = sorted({name for name, module in model.named_modules() if isinstance(module, nn.Linear)})[:50]
        raise RuntimeError(f"no OASR target linears found; first available linears: {available}")
    return dict(candidates)


def clone_weights(modules: dict[str, nn.Linear]) -> dict[str, torch.Tensor]:
    return {name: module.weight.detach().clone() for name, module in modules.items()}


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
) -> dict[str, np.ndarray]:
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
            if take > 0:
                chunks[name].append(x2[:take].float().cpu())
                counts[name] += take

        return hook

    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        for batch in token_batches(tokenizer, texts, sequence_length=sequence_length, batch_size=batch_size, limit=calib_limit):
            model(input_ids=batch.to(device))
            if all(count >= max_rows for count in counts.values()):
                break
    for handle in handles:
        handle.remove()
    out: dict[str, np.ndarray] = {}
    for name, module in modules.items():
        if chunks[name]:
            out[name] = torch.cat(chunks[name], dim=0).numpy().astype(np.float32)
        else:
            out[name] = np.empty((0, module.weight.shape[1]), dtype=np.float32)
    return out


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


def lowrank_rank_for_params(shape: tuple[int, int], params: int) -> int:
    rows, cols = shape
    return max(0, min(rows, cols, int(params) // max(rows + cols, 1)))


def lowrank_ceil_rank_for_params(shape: tuple[int, int], params: int) -> int:
    rows, cols = shape
    return max(0, min(rows, cols, int(math.ceil(max(int(params), 0) / max(rows + cols, 1)))))


def permutation_metadata_params(method: str, shape: tuple[int, int]) -> int:
    if method in {"norm_sorted_block_circulant", "activation_clustered_block_circulant", "random_permuted_block_circulant"}:
        return int(shape[0] + shape[1])
    return 0


def random_permutations(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.permutation(shape[0]).astype(np.int64), rng.permutation(shape[1]).astype(np.int64)


def build_structured_projection(
    residual: np.ndarray,
    x: np.ndarray,
    *,
    method: str,
    block_size: int,
    seed: int,
):
    if method == "naive_block_circulant":
        return block_circulant_project(residual, block_size=block_size)
    if method == "norm_sorted_block_circulant":
        row_perm, col_perm = norm_sorted_permutations(residual)
        return permuted_block_circulant_project(residual, block_size=block_size, row_perm=row_perm, col_perm=col_perm)
    if method == "activation_clustered_block_circulant":
        row_perm, _col_norm_perm = norm_sorted_permutations(residual)
        col_perm = activation_clustered_permutation(x, block_size=block_size)
        return permuted_block_circulant_project(residual, block_size=block_size, row_perm=row_perm, col_perm=col_perm)
    if method == "random_permuted_block_circulant":
        row_perm, col_perm = random_permutations(residual.shape, seed)
        return permuted_block_circulant_project(residual, block_size=block_size, row_perm=row_perm, col_perm=col_perm)
    if method == "monarch_like_two_block":
        return monarch_like_two_block_project(residual, block_size=block_size)
    raise ValueError(f"unknown structured method: {method}")


def evaluate_layer(
    *,
    model_name: str,
    layer: str,
    q_method: str,
    q_bits: int,
    block_sizes: list[int],
    methods: list[str],
    weight: np.ndarray,
    x: np.ndarray,
    group_size: int,
) -> list[dict[str, object]]:
    q_weight = quantize_weight(weight, method=q_method, bits=q_bits, group_size=group_size)
    residual = weight - q_weight
    q_activation_error = compute_activation_error(x, weight, q_weight)
    rows: list[dict[str, object]] = []
    lowrank_cache = {}
    for block_size in block_sizes:
        for method in methods:
            projection = build_structured_projection(
                residual,
                x,
                method=method,
                block_size=block_size,
                seed=stable_seed(model_name, layer, q_method, q_bits, block_size, method),
            )
            coeff_params = int(projection.params)
            perm_params = permutation_metadata_params(method, weight.shape)
            structured_params = coeff_params + perm_params
            rank = lowrank_rank_for_params(weight.shape, structured_params)
            ceil_rank = lowrank_ceil_rank_for_params(weight.shape, structured_params)
            for cached_rank in {rank, ceil_rank}:
                if cached_rank not in lowrank_cache:
                    lowrank_cache[cached_rank] = lowrank_project(residual, cached_rank)
            lowrank = lowrank_cache[rank]
            ceil_lowrank = lowrank_cache[ceil_rank]
            structured_hat = q_weight + projection.matrix
            lowrank_hat = q_weight + lowrank.matrix
            ceil_lowrank_hat = q_weight + ceil_lowrank.matrix
            structured_activation_error = compute_activation_error(x, weight, structured_hat)
            lowrank_activation_error = compute_activation_error(x, weight, lowrank_hat)
            ceil_lowrank_activation_error = compute_activation_error(x, weight, ceil_lowrank_hat)
            dense_params = max(weight.size, 1)
            lowrank_params = int(rank * (weight.shape[0] + weight.shape[1]))
            ceil_lowrank_params = int(ceil_rank * (weight.shape[0] + weight.shape[1]))
            rows.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "layer_short": short_layer_name(layer),
                    "q_method": q_method,
                    "q_bits": int(q_bits),
                    "block_size": int(block_size),
                    "method": method,
                    "structured_coeff_params": int(coeff_params),
                    "permutation_metadata_params": int(perm_params),
                    "structured_params": int(structured_params),
                    "structured_memory_ratio": float(structured_params / dense_params),
                    "matched_lowrank_rank": int(rank),
                    "matched_lowrank_params": int(lowrank_params),
                    "matched_lowrank_memory_ratio": float(lowrank_params / dense_params),
                    "matched_lowrank_unused_params": int(structured_params - lowrank_params),
                    "matched_lowrank_unused_memory_ratio": float((structured_params - lowrank_params) / dense_params),
                    "ceil_lowrank_rank": int(ceil_rank),
                    "ceil_lowrank_params": int(ceil_lowrank_params),
                    "ceil_lowrank_memory_ratio": float(ceil_lowrank_params / dense_params),
                    "ceil_lowrank_extra_params": int(ceil_lowrank_params - structured_params),
                    "ceil_lowrank_activation_error": float(ceil_lowrank_activation_error),
                    "q_activation_error": float(q_activation_error),
                    "activation_error": float(structured_activation_error),
                    "matched_lowrank_activation_error": float(lowrank_activation_error),
                    "delta_activation_vs_lowrank": float(structured_activation_error - lowrank_activation_error),
                    "delta_activation_vs_ceil_lowrank": float(structured_activation_error - ceil_lowrank_activation_error),
                    "structured_wins": bool(structured_activation_error < lowrank_activation_error),
                    "structured_wins_vs_ceil_lowrank": bool(structured_activation_error < ceil_lowrank_activation_error),
                    "structured_weight_error": relative_fro_error(residual, projection.matrix),
                    "matched_lowrank_weight_error": relative_fro_error(residual, lowrank.matrix),
                    "delta_weight_error_vs_lowrank": relative_fro_error(residual, projection.matrix) - relative_fro_error(residual, lowrank.matrix),
                }
            )
    return rows


def plot_results(root: Path, rows: list[dict[str, object]]) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not rows:
        for name in ["structured_vs_lowrank_scatter.png", "activation_error_by_method.png"]:
            fig, ax = plt.subplots(figsize=(6.0, 3.4))
            ax.axis("off")
            ax.text(0.5, 0.5, "No rows produced.", ha="center", va="center")
            fig.savefig(figures / name, dpi=220, bbox_inches="tight")
            plt.close(fig)
        return

    methods = sorted({str(row["method"]) for row in rows})
    colors = {method: plt.cm.tab10(i % 10) for i, method in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for row in rows:
        ax.scatter(
            float(row["matched_lowrank_activation_error"]),
            float(row["activation_error"]),
            color=colors[str(row["method"])],
            s=42,
            alpha=0.75,
            label=str(row["method"]),
        )
    values = [float(row["matched_lowrank_activation_error"]) for row in rows] + [float(row["activation_error"]) for row in rows]
    low, high = min(values), max(values)
    pad = 0.02 * max(high - low, 1e-12)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="black", linewidth=0.8)
    ax.set_xlabel("matched low-rank activation error")
    ax.set_ylabel("structured activation error")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=7)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(figures / "structured_vs_lowrank_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    layers = sorted({str(row["layer_short"]) for row in rows})
    matrix = np.full((len(methods), len(layers)), np.nan, dtype=np.float64)
    for i, method in enumerate(methods):
        for j, layer in enumerate(layers):
            values = [float(row["activation_error"]) / max(float(row["matched_lowrank_activation_error"]), 1e-12) for row in rows if row["method"] == method and row["layer_short"] == layer]
            if values:
                matrix[i, j] = min(values)
    fig, ax = plt.subplots(figsize=(max(7.0, len(layers) * 1.1), max(3.8, len(methods) * 0.42)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.9, vmax=1.1)
    ax.set_xticks(range(len(layers)), layers, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    fig.colorbar(im, ax=ax, label="best structured / low-rank activation error")
    fig.tight_layout()
    fig.savefig(figures / "activation_error_by_method.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, *, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    wins = [row for row in rows if bool(row["structured_wins"])]
    best = min(rows, key=lambda row: float(row["delta_activation_vs_lowrank"])) if rows else None
    lines = [
        "# Structured Residual Matched-Memory Offline Test",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset request: `{args.dataset}/{args.subset}` split `{args.split}`; backup `{args.backup_name}`; allow_fallback={bool(args.allow_fallback)}",
        f"- Target layers: `{args.target_types}`; max_layers={args.max_layers}",
        f"- Fixed Q base: methods=`{args.q_methods}`, bits={args.q_bits}, group_size={args.group_size}",
        f"- Block sizes: `{args.block_sizes}`",
        f"- Structured methods: `{args.methods}`",
        f"- Calibration only: calib_limit={args.calib_limit}; activation_rows={args.activation_sample_rows}; sequence_length={args.sequence_length}; batch_size={args.batch_size}",
        "- Accounting: explicit row/column permutation metadata is counted as rows+cols parameters for permuted variants. Matched low-rank uses the largest rank not exceeding the structured parameter count; ceil-rank diagnostics are also saved.",
        "",
        "## Decision",
        "",
    ]
    if wins:
        lines.append(f"- Structured residual wins in {len(wins)}/{len(rows)} matched-memory comparisons. PPL can be considered for the winning rows.")
    else:
        lines.append(f"- Structured residual wins in 0/{len(rows)} matched-memory comparisons against the non-overbudget matched low-rank floor rank. Criterion failed, so PPL was not run.")
        lines.append("- Conservative decision: stop the current block-circulant / permutation / Monarch-like structured residual line for now.")
    if best:
        lines.extend(
            [
                "",
                "## Best Structured Case",
                "",
                f"- Method: `{best['method']}`; layer `{best['layer_short']}`; q `{best['q_method']}` q{best['q_bits']}; block_size={best['block_size']}",
                f"- Structured activation error: {float(best['activation_error']):.6g}",
                f"- Matched low-rank activation error: {float(best['matched_lowrank_activation_error']):.6g}",
                f"- Delta structured-lowrank: {float(best['delta_activation_vs_lowrank']):+.6g}",
            ]
        )
    lines.extend(["", "## Win Counts By Method", "", "| Method | Wins | Rows | Best delta act |", "|---|---:|---:|---:|"])
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        method_wins = sum(bool(row["structured_wins"]) for row in subset)
        best_delta = min(float(row["delta_activation_vs_lowrank"]) for row in subset)
        lines.append(f"| {method} | {method_wins} | {len(subset)} | {best_delta:+.6g} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `matched_residual_metrics.csv`",
            "- `figures/structured_vs_lowrank_scatter.png`",
            "- `figures/activation_error_by_method.png`",
            "- `run_config.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-memory offline test for structured residuals against low-rank residuals.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--target-types", default="attention_o,mlp_up,mlp_down")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--q-methods", default="rtn,sinq_like")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument(
        "--methods",
        default="naive_block_circulant,norm_sorted_block_circulant,activation_clustered_block_circulant,random_permuted_block_circulant,monarch_like_two_block",
    )
    parser.add_argument("--calib-limit", type=int, default=8)
    parser.add_argument("--activation-sample-rows", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--subset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--backup-name", default="wikitext_2_raw")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    q_methods = parse_csv(args.q_methods, ["rtn", "sinq_like"])
    block_sizes = parse_int_csv(args.block_sizes, [16, 32, 64])
    methods = parse_csv(args.methods, [])
    model_id = args.model.replace("/", "_").replace(":", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("results") / f"structured_residual_matched_{model_id}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model_and_tokenizer_from_config(
        {
            "model": args.model,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "local_files_only": args.local_files_only,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
    )
    args.device = device
    modules = discover_oasr_linears(model, target_types=parse_csv(args.target_types, []), max_layers=args.max_layers)
    baseline_weights = clone_weights(modules)
    data_cfg = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "backup_name": args.backup_name,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "allow_fallback": args.allow_fallback,
    }
    texts = load_texts_from_config(data_cfg, limit=max(args.calib_limit * 4, 1))
    samples = collect_activation_samples(
        model,
        tokenizer,
        modules,
        texts=texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
        max_rows=args.activation_sample_rows,
    )

    rows: list[dict[str, object]] = []
    for layer, tensor in baseline_weights.items():
        weight = tensor.float().cpu().numpy().astype(np.float32)
        x = samples[layer]
        for q_method in q_methods:
            rows.extend(
                evaluate_layer(
                    model_name=args.model,
                    layer=layer,
                    q_method=q_method,
                    q_bits=args.q_bits,
                    block_sizes=block_sizes,
                    methods=methods,
                    weight=weight,
                    x=x,
                    group_size=args.group_size,
                )
            )
    write_csv(root / "matched_residual_metrics.csv", rows)
    plot_results(root, rows)
    write_summary(root / "summary.md", args=args, rows=rows)
    write_json(
        root / "run_config.json",
        {
            "model": args.model,
            "device": args.device,
            "selected_layers": list(modules),
            "data": data_cfg,
            "args": vars(args),
        },
    )
    print(root)


if __name__ == "__main__":
    main()
