from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from llm_spectral_dynamics.structured.data import load_model_and_tokenizer_from_config, load_texts_from_config, token_batches
from llm_spectral_dynamics.structured.oasr import (
    OASRCandidate,
    block_circulant_project,
    candidate_metrics,
    compute_conditional_overlap,
    estimate_memory,
    feasible_block_sizes,
    lowrank_project,
    lowrank_rank_for_memory,
    quantize_weight,
    random_block_circulant_baseline,
    relative_fro_error,
    score_candidate,
)


EPS = 1e-12


REQUIRED_FIGURES = {
    "memory_ppl_frontier.png",
    "candidate_activation_error_by_layer.png",
    "conditional_overlap_heatmap.png",
    "residual_structure_scatter.png",
}


def parse_csv(value: str, default: list[str]) -> list[str]:
    if not str(value).strip():
        return list(default)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_csv(value: str, default: list[float]) -> list[float]:
    if not str(value).strip():
        return list(default)
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str, default: list[int]) -> list[int]:
    if not str(value).strip():
        return list(default)
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


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


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def linear_layer_index(name: str) -> int | None:
    import re

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


def restore_weights(modules: dict[str, nn.Linear], baseline: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in modules.items():
            module.weight.copy_(baseline[name])


def apply_replacements(modules: dict[str, nn.Linear], replacements: dict[str, np.ndarray]) -> None:
    with torch.no_grad():
        for name, weight in replacements.items():
            module = modules[name]
            module.weight.copy_(torch.as_tensor(weight, device=module.weight.device, dtype=module.weight.dtype))


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
    nll = nll_total / max(token_total, 1)
    return {"nll": nll, "perplexity": float(math.exp(min(nll, 50.0))), "tokens": token_total}


def evaluate_replacements(
    model: nn.Module,
    tokenizer: object,
    modules: dict[str, nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, np.ndarray],
    *,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> dict[str, float | int]:
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
    restore_weights(modules, baseline_weights)
    return metrics


def choose_q_bits(target_memory: float, q_bits_list: list[int], *, require_residual_room: bool = False) -> int | None:
    ordered = sorted({int(bits) for bits in q_bits_list}, reverse=True)
    for bits in ordered:
        q_memory = bits / 16.0
        if q_memory <= target_memory + 1e-12 and (not require_residual_room or q_memory < target_memory - 1e-12):
            return bits
    return None


def feasible_q_bits(target_memory: float, q_bits_list: list[int]) -> list[int]:
    return [bits for bits in sorted({int(bits) for bits in q_bits_list}, reverse=True) if bits / 16.0 <= target_memory + 1e-12]


def build_candidate(
    *,
    layer: str,
    family: str,
    target_memory: float,
    q_method: str,
    q_bits: int,
    split_c: float,
    split_l: float,
    x: np.ndarray,
    weight: np.ndarray,
    q_weight: np.ndarray,
    c_projection,
    l_projection,
    threshold: float,
) -> OASRCandidate:
    c_res = c_projection.matrix if c_projection is not None else np.zeros_like(weight, dtype=np.float32)
    l_res = l_projection.matrix if l_projection is not None else np.zeros_like(weight, dtype=np.float32)
    block_size = int(c_projection.block_size) if c_projection is not None else 0
    l_rank = int(l_projection.rank) if l_projection is not None else 0
    c_params = int(c_projection.params) if c_projection is not None else 0
    row = candidate_metrics(
        layer=layer,
        family=family,
        target_memory_ratio=target_memory,
        q_method=q_method,
        q_bits=q_bits,
        block_size=block_size,
        split_c=split_c,
        split_l=split_l,
        x=x,
        weight=weight,
        q_weight=q_weight,
        c_res=c_res,
        l_res=l_res,
        l_rank=l_rank,
        c_params=c_params,
    )
    if c_projection is not None:
        matched_rank = lowrank_rank_for_memory(weight.shape, c_params / max(weight.size, 1))
        matched_lowrank = lowrank_project(weight - q_weight, matched_rank)
        random_c = random_block_circulant_baseline(weight - q_weight, block_size, seed=stable_seed(layer, block_size))
        row["lowrank_projection_error_at_c_memory"] = relative_fro_error(weight - q_weight, matched_lowrank.matrix)
        row["random_block_circulant_projection_error"] = relative_fro_error(weight - q_weight, random_c.matrix)
    else:
        row["lowrank_projection_error_at_c_memory"] = ""
        row["random_block_circulant_projection_error"] = ""
    rho_values = []
    if np.any(c_res):
        rho_values.append(float(row["rho_q_error_c_res"]))
    if np.any(l_res):
        rho_values.append(float(row["rho_q_plus_c_error_l_res"] if np.any(c_res) else row["rho_q_error_l_res"]))
    filter_rho = max([max(value, 0.0) for value in rho_values], default=0.0)
    row["rho_new_vs_base"] = filter_rho
    row["filter_threshold"] = threshold
    row["filter_pass"] = filter_rho <= threshold
    row["score"] = score_candidate(row)
    return OASRCandidate(
        layer=layer,
        family=family,
        target_memory_ratio=target_memory,
        q_method=q_method,
        q_bits=q_bits,
        block_size=block_size,
        rank=l_rank,
        split_c=split_c,
        split_l=split_l,
        memory_ratio=float(row["memory_ratio"]),
        weight_hat=(q_weight + c_res + l_res).astype(np.float32),
        q_weight=q_weight,
        c_res=c_res,
        l_res=l_res,
        q_error=q_weight - weight,
        q_plus_c_error=q_weight + c_res - weight,
        filter_rho=filter_rho,
        filter_pass=bool(row["filter_pass"]),
        score=float(row["score"]),
        metrics=row,
    )


def generate_layer_candidates(
    layer: str,
    weight: np.ndarray,
    x: np.ndarray,
    *,
    budgets: list[float],
    splits: list[tuple[float, float]],
    q_methods: list[str],
    q_bits_list: list[int],
    block_sizes: list[int],
    threshold: float,
    group_size: int,
    include_l_c: bool,
) -> list[OASRCandidate]:
    candidates: list[OASRCandidate] = []
    for target in budgets:
        for q_method in q_methods:
            for q_bits in feasible_q_bits(target, q_bits_list):
                q_weight = quantize_weight(weight, method=q_method, bits=q_bits, group_size=group_size)
                q_memory = q_bits / 16.0
                residual_budget = max(target - q_memory, 0.0)
                if q_memory <= target + 1e-12:
                    zero = np.zeros_like(weight, dtype=np.float32)
                    candidates.append(
                        build_candidate(
                            layer=layer,
                            family="q_only",
                            target_memory=target,
                            q_method=q_method,
                            q_bits=q_bits,
                            split_c=0.0,
                            split_l=0.0,
                            x=x,
                            weight=weight,
                            q_weight=q_weight,
                            c_projection=None,
                            l_projection=None,
                            threshold=threshold,
                        )
                    )
                    rank = lowrank_rank_for_memory(weight.shape, residual_budget)
                    l_projection = lowrank_project(weight - q_weight, rank)
                    candidates.append(
                        build_candidate(
                            layer=layer,
                            family="q_l",
                            target_memory=target,
                            q_method=q_method,
                            q_bits=q_bits,
                            split_c=0.0,
                            split_l=1.0,
                            x=x,
                            weight=weight,
                            q_weight=q_weight,
                            c_projection=None,
                            l_projection=l_projection,
                            threshold=threshold,
                        )
                    )
                for split_c, split_l in splits:
                    if residual_budget <= 0:
                        continue
                    c_budget = residual_budget * float(split_c)
                    l_budget = residual_budget * float(split_l)
                    feasible = feasible_block_sizes(weight.shape, c_budget, block_sizes)
                    for block_size in feasible:
                        c_projection = block_circulant_project(weight - q_weight, block_size=block_size)
                        c_memory = c_projection.params / max(weight.size, 1)
                        if c_memory > residual_budget + 1e-12:
                            continue
                        candidates.append(
                            build_candidate(
                                layer=layer,
                                family="q_c",
                                target_memory=target,
                                q_method=q_method,
                                q_bits=q_bits,
                                split_c=split_c,
                                split_l=0.0,
                                x=x,
                                weight=weight,
                                q_weight=q_weight,
                                c_projection=c_projection,
                                l_projection=None,
                                threshold=threshold,
                            )
                        )
                        rank = lowrank_rank_for_memory(weight.shape, max(target - q_memory - c_memory, 0.0))
                        l_projection = lowrank_project(weight - q_weight - c_projection.matrix, rank)
                        candidates.append(
                            build_candidate(
                                layer=layer,
                                family="q_c_l",
                                target_memory=target,
                                q_method=q_method,
                                q_bits=q_bits,
                                split_c=split_c,
                                split_l=split_l,
                                x=x,
                                weight=weight,
                                q_weight=q_weight,
                                c_projection=c_projection,
                                l_projection=l_projection,
                                threshold=threshold,
                            )
                        )
                        if include_l_c and l_budget > 0:
                            rank_l_first = lowrank_rank_for_memory(weight.shape, l_budget)
                            l_first = lowrank_project(weight - q_weight, rank_l_first)
                            remaining_for_c = max(target - q_memory - estimate_memory(weight.shape, l_rank=l_first.rank), 0.0)
                            for second_block_size in feasible_block_sizes(weight.shape, remaining_for_c, [block_size]):
                                c_second = block_circulant_project(weight - q_weight - l_first.matrix, block_size=second_block_size)
                                candidates.append(
                                    build_candidate(
                                        layer=layer,
                                        family="q_l_c",
                                        target_memory=target,
                                        q_method=q_method,
                                        q_bits=q_bits,
                                        split_c=split_c,
                                        split_l=split_l,
                                        x=x,
                                        weight=weight,
                                        q_weight=q_weight,
                                        c_projection=c_second,
                                        l_projection=l_first,
                                        threshold=threshold,
                                    )
                                )
    return candidates


def candidate_to_row(candidate: OASRCandidate) -> dict[str, object]:
    row = dict(candidate.metrics)
    row["layer_short"] = short_layer_name(candidate.layer)
    row["oasr_layer_family"] = oasr_family_name(candidate.layer)
    return row


def weighted_memory(candidates: Iterable[OASRCandidate], layer_params: dict[str, int]) -> float:
    total = max(sum(layer_params[candidate.layer] for candidate in candidates), 1)
    return sum(candidate.memory_ratio * layer_params[candidate.layer] for candidate in candidates) / float(total)


def select_recipe(candidates: list[OASRCandidate], target_memory: float, layer_params: dict[str, int]) -> tuple[list[OASRCandidate], dict[str, object]]:
    by_layer: dict[str, list[OASRCandidate]] = {}
    for candidate in candidates:
        if abs(candidate.target_memory_ratio - target_memory) <= 1e-12:
            by_layer.setdefault(candidate.layer, []).append(candidate)
    pools: list[list[OASRCandidate]] = []
    for layer, items in sorted(by_layer.items()):
        feasible = [item for item in items if item.filter_pass]
        pool = feasible if feasible else items
        pools.append(sorted(pool, key=lambda item: (item.score, item.memory_ratio, item.family))[:24])
    best_combo: tuple[float, float, tuple[OASRCandidate, ...]] | None = None
    best_over_budget: tuple[float, float, tuple[OASRCandidate, ...]] | None = None
    for combo in itertools.product(*pools):
        memory = weighted_memory(combo, layer_params)
        score = sum(item.score for item in combo) / max(len(combo), 1)
        key = (score, memory, combo)
        if memory <= target_memory + 1e-12:
            if best_combo is None or (score, memory) < (best_combo[0], best_combo[1]):
                best_combo = key
        elif best_over_budget is None or (memory, score) < (best_over_budget[1], best_over_budget[0]):
            best_over_budget = key
    selected = best_combo or best_over_budget
    if selected is None:
        return [], {"target_memory_ratio": target_memory, "status": "no_candidates"}
    score, memory, combo = selected
    return list(combo), {
        "target_memory_ratio": target_memory,
        "status": "ok" if memory <= target_memory + 1e-12 else "over_budget_fallback",
        "selected_memory_ratio": memory,
        "selected_mean_score": score,
    }


def best_family_recipe(candidates: list[OASRCandidate], target_memory: float, family: str, layer_params: dict[str, int]) -> tuple[list[OASRCandidate], dict[str, object]]:
    selected: list[OASRCandidate] = []
    for layer in sorted({candidate.layer for candidate in candidates}):
        items = [candidate for candidate in candidates if candidate.layer == layer and candidate.family == family and abs(candidate.target_memory_ratio - target_memory) <= 1e-12]
        feasible = [item for item in items if item.memory_ratio <= target_memory + 1e-12]
        pool = feasible if feasible else items
        if pool:
            selected.append(min(pool, key=lambda item: (not item.filter_pass, item.score, item.memory_ratio)))
    if not selected:
        return [], {"status": "no_candidates", "target_memory_ratio": target_memory}
    return selected, {
        "status": "ok",
        "target_memory_ratio": target_memory,
        "selected_memory_ratio": weighted_memory(selected, layer_params),
        "selected_mean_score": sum(item.score for item in selected) / len(selected),
    }


def save_placeholder_figure(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(figures_dir: Path, candidate_rows: list[dict[str, object]], strategy_rows: list[dict[str, object]]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: set[str] = set()
    compressed = [row for row in strategy_rows if row["strategy"] != "dense_baseline"]
    if compressed:
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        families = sorted({str(row["family"]) for row in compressed})
        colors = {family: plt.cm.tab10(index % 10) for index, family in enumerate(families)}
        for row in compressed:
            ax.scatter(float(row["memory_ratio"]), float(row["perplexity"]), color=colors[str(row["family"])], label=str(row["family"]), s=55)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8)
        ax.set_xlabel("memory ratio")
        ax.set_ylabel("PPL")
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(figures_dir / "memory_ppl_frontier.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        generated.add("memory_ppl_frontier.png")

    if candidate_rows:
        layers = sorted({str(row["layer_short"]) for row in candidate_rows})
        families = ["q_only", "q_l", "q_c", "q_c_l"]
        matrix = np.full((len(families), len(layers)), np.nan, dtype=np.float64)
        for i, family in enumerate(families):
            for j, layer in enumerate(layers):
                values = [float(row["activation_error"]) for row in candidate_rows if row["family"] == family and row["layer_short"] == layer]
                if values:
                    matrix[i, j] = min(values)
        fig, ax = plt.subplots(figsize=(max(7.0, len(layers) * 1.1), 3.8))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(layers)), layers, rotation=30, ha="right")
        ax.set_yticks(range(len(families)), families)
        fig.colorbar(im, ax=ax, label="min activation error")
        fig.tight_layout()
        fig.savefig(figures_dir / "candidate_activation_error_by_layer.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        generated.add("candidate_activation_error_by_layer.png")

        overlap_specs = [
            ("rho(Q,L)", "q_l", "rho_q_error_l_res"),
            ("rho(Q,C)", "q_c", "rho_q_error_c_res"),
            ("rho(Q+C,L)", "q_c_l", "rho_q_plus_c_error_l_res"),
            ("rho(C,L)", "q_c_l", "rho_c_res_l_res"),
        ]
        overlap_rows = [row for row in candidate_rows if row["family"] in {spec[1] for spec in overlap_specs}]
        if overlap_rows:
            layer_labels = sorted({str(row["layer_short"]) for row in overlap_rows})
            heat = np.full((len(overlap_specs), len(layer_labels)), np.nan, dtype=np.float64)
            for i, (_label, family, metric_key) in enumerate(overlap_specs):
                for j, layer in enumerate(layer_labels):
                    values = [
                        (float(row["score"]), float(row[metric_key]))
                        for row in overlap_rows
                        if row["family"] == family and row["layer_short"] == layer and row.get(metric_key) not in {"", None}
                    ]
                    if values:
                        heat[i, j] = min(values, key=lambda item: item[0])[1]
            fig, ax = plt.subplots(figsize=(max(7.0, len(layer_labels) * 1.1), 4.4))
            im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(layer_labels)), layer_labels, rotation=30, ha="right")
            ax.set_yticks(range(len(overlap_specs)), [spec[0] for spec in overlap_specs])
            fig.colorbar(im, ax=ax, label="signed conditional rho")
            fig.tight_layout()
            fig.savefig(figures_dir / "conditional_overlap_heatmap.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            generated.add("conditional_overlap_heatmap.png")

        c_rows = [row for row in candidate_rows if row["family"] in {"q_c", "q_c_l"} and row["block_circulant_projection_error"] != ""]
        if c_rows:
            fig, ax = plt.subplots(figsize=(6.2, 4.8))
            x = [float(row["block_circulant_projection_error"]) for row in c_rows]
            y = [float(row["lowrank_projection_error_at_c_memory"]) for row in c_rows]
            color = ["#2563eb" if float(row["block_circulant_projection_error"]) < float(row["random_block_circulant_projection_error"]) else "#dc2626" for row in c_rows]
            ax.scatter(x, y, c=color, alpha=0.75)
            lim = [min(x + y), max(x + y)]
            ax.plot(lim, lim, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("block-circulant residual error")
            ax.set_ylabel("low-rank error at C memory")
            ax.grid(True, alpha=0.22)
            fig.tight_layout()
            fig.savefig(figures_dir / "residual_structure_scatter.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            generated.add("residual_structure_scatter.png")
    for filename in sorted(REQUIRED_FIGURES - generated):
        save_placeholder_figure(figures_dir / filename, f"No data available for {filename}.")


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    baseline: dict[str, float | int],
    strategy_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    selected_recipe: dict[str, object],
) -> None:
    lines = [
        "# OASR Structured Residual Smoke Result",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset request: `{args.dataset}/{args.subset}` split `{args.split}`; backup `{args.backup_name}`; allow_fallback={bool(args.allow_fallback)}",
        f"- Target layers: `{args.target_types}`; max_layers={args.max_layers}",
        f"- Quantizer methods: `{args.q_methods}`; q_bits={args.q_bits_list}; group_size={args.group_size}",
        f"- Block sizes: `{args.block_sizes}`; C:L splits: `{args.splits}`; include_l_c_order={bool(args.include_l_c_order)}",
        f"- Calibration/eval: calib_limit={args.calib_limit}; eval_limit={args.eval_limit}; activation_rows={args.activation_sample_rows}; sequence_length={args.sequence_length}; batch_size={args.batch_size}",
        f"- Dense baseline PPL: {float(baseline['perplexity']):.4f}; NLL: {float(baseline['nll']):.4f}; tokens: {int(baseline['tokens'])}",
        f"- Target memory ratios: {args.memory_ratios}",
        f"- Conditional-overlap filter threshold: {args.overlap_threshold}",
        "",
        "## Strategy PPL",
        "",
        "| Strategy | Family | Target memory | Actual memory | PPL | PPL delta | Mean score | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in strategy_rows:
        lines.append(
            f"| {row['strategy']} | {row['family']} | {float(row['target_memory_ratio']):.3f} | {float(row['memory_ratio']):.3f} | "
            f"{float(row['perplexity']):.4f} | {float(row['ppl_delta']):+.4f} | {float(row.get('selected_mean_score', 0.0)):.4g} | {row.get('status', '')} |"
        )
    same_budget = [row for row in strategy_rows if row["family"] in {"q_l", "q_c_l"}]
    lines.extend(["", "## Same-Budget Interpretation", ""])
    for target in sorted({float(row["target_memory_ratio"]) for row in same_budget}):
        ql = next((row for row in same_budget if float(row["target_memory_ratio"]) == target and row["family"] == "q_l"), None)
        qcl = next((row for row in same_budget if float(row["target_memory_ratio"]) == target and row["family"] == "q_c_l"), None)
        if ql and qcl:
            same_memory = abs(float(ql["memory_ratio"]) - float(qcl["memory_ratio"])) <= 0.005
            better = float(qcl["perplexity"]) < float(ql["perplexity"])
            lines.append(
                f"- Target {target:.3f}: Q+C+L PPL {float(qcl['perplexity']):.4f} vs Q+L {float(ql['perplexity']):.4f}; "
                f"same-memory={same_memory}; Q+C+L better={better}."
            )
    c_rows = [row for row in candidate_rows if row["family"] in {"q_c", "q_c_l"} and row.get("random_block_circulant_projection_error") not in {"", None}]
    if c_rows:
        wins = sum(float(row["block_circulant_projection_error"]) < float(row["random_block_circulant_projection_error"]) for row in c_rows)
        lines.extend(["", "## Residual Structure Diagnostic", ""])
        lines.append(f"- Block-circulant projection beats random structured baseline in {wins}/{len(c_rows)} C-bearing candidates.")
    selected = selected_recipe.get("selected", [])
    if isinstance(selected, list):
        qcl_layers = [row for row in selected if row.get("family") == "q_c_l"]
        lines.append(f"- Selector picked Q+C+L in {len(qcl_layers)}/{len(selected)} selected layer-budget rows.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `candidate_metrics.csv`",
            "- `selected_recipe.json`",
            "- `figures/memory_ppl_frontier.png`",
            "- `figures/candidate_activation_error_by_layer.png`",
            "- `figures/conditional_overlap_heatmap.png`",
            "- `figures/residual_structure_scatter.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal OASR structured residual compression smoke experiment.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--target-types", default="attention_o,mlp_up,mlp_down")
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--q-methods", default="rtn")
    parser.add_argument("--q-bits-list", default="4,3,2")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--memory-ratios", default="0.196,0.220,0.258")
    parser.add_argument("--splits", default="0.25:0.75,0.5:0.5,0.75:0.25")
    parser.add_argument("--overlap-threshold", type=float, default=0.3)
    parser.add_argument("--include-l-c-order", action="store_true")
    parser.add_argument("--calib-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--activation-sample-rows", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--subset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--backup-name", default="wikitext_2_raw")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def parse_splits(value: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in parse_csv(value, ["0.25:0.75", "0.5:0.5", "0.75:0.25"]):
        left, right = item.split(":", 1)
        out.append((float(left), float(right)))
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    budgets = parse_float_csv(args.memory_ratios, [0.196, 0.220, 0.258])
    splits = parse_splits(args.splits)
    q_methods = parse_csv(args.q_methods, ["rtn"])
    q_bits_list = parse_int_csv(args.q_bits_list, [4, 3, 2])
    block_sizes = parse_int_csv(args.block_sizes, [16, 32, 64])
    model_id = args.model.replace("/", "_").replace(":", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_dir) if args.output_dir else Path("results") / f"oasr_structured_residual_{model_id}_{timestamp}"
    metrics_dir = root
    figures_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)

    data_cfg = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "backup_name": args.backup_name,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "allow_fallback": args.allow_fallback,
    }
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
    texts = load_texts_from_config(data_cfg, limit=max(args.calib_limit, args.eval_limit) * 8)
    calib_texts = texts[: max(args.calib_limit * 4, 1)]
    eval_texts = texts[max(args.calib_limit * 4, 1) :] or texts
    baseline = evaluate_current_model(
        model,
        tokenizer,
        texts=eval_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        eval_limit=args.eval_limit,
    )
    samples = collect_activation_samples(
        model,
        tokenizer,
        modules,
        texts=calib_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
        max_rows=args.activation_sample_rows,
    )
    layer_params = {name: int(weight.numel()) for name, weight in baseline_weights.items()}
    all_candidates: list[OASRCandidate] = []
    for name, weight_tensor in baseline_weights.items():
        weight = weight_tensor.float().cpu().numpy().astype(np.float32)
        x = samples[name]
        all_candidates.extend(
            generate_layer_candidates(
                name,
                weight,
                x,
                budgets=budgets,
                splits=splits,
                q_methods=q_methods,
                q_bits_list=q_bits_list,
                block_sizes=block_sizes,
                threshold=args.overlap_threshold,
                group_size=args.group_size,
                include_l_c=args.include_l_c_order,
            )
        )
    candidate_rows = [candidate_to_row(candidate) for candidate in all_candidates]
    write_csv(metrics_dir / "candidate_metrics.csv", candidate_rows)

    selected_payload: dict[str, object] = {"budgets": [], "selected": []}
    strategy_rows: list[dict[str, object]] = [
        {
            "strategy": "dense_baseline",
            "family": "baseline",
            "target_memory_ratio": 1.0,
            "memory_ratio": 1.0,
            "nll": float(baseline["nll"]),
            "perplexity": float(baseline["perplexity"]),
            "tokens": int(baseline["tokens"]),
            "ppl_delta": 0.0,
            "selected_mean_score": 0.0,
            "status": "ok",
        }
    ]
    families = ["q_only", "q_l", "q_c", "q_c_l"] + (["q_l_c"] if args.include_l_c_order else [])
    for target in budgets:
        for family in families:
            recipe, info = best_family_recipe(all_candidates, target, family, layer_params)
            if not recipe:
                continue
            replacements = {candidate.layer: candidate.weight_hat for candidate in recipe}
            metrics = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                texts=eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=device,
                eval_limit=args.eval_limit,
            )
            strategy_rows.append(
                {
                    "strategy": f"{family}_target{target:.3f}",
                    "family": family,
                    "target_memory_ratio": target,
                    "memory_ratio": info.get("selected_memory_ratio", weighted_memory(recipe, layer_params)),
                    "nll": float(metrics["nll"]),
                    "perplexity": float(metrics["perplexity"]),
                    "tokens": int(metrics["tokens"]),
                    "ppl_delta": float(metrics["perplexity"]) - float(baseline["perplexity"]),
                    "selected_mean_score": info.get("selected_mean_score", 0.0),
                    "status": info.get("status", ""),
                }
            )
        recipe, info = select_recipe(all_candidates, target, layer_params)
        selected_payload["budgets"].append(info)
        selected_payload["selected"].extend(
            [
                {
                    "target_memory_ratio": target,
                    "layer": candidate.layer,
                    "layer_short": short_layer_name(candidate.layer),
                    "family": candidate.family,
                    "memory_ratio": candidate.memory_ratio,
                    "score": candidate.score,
                    "filter_rho": candidate.filter_rho,
                    "filter_pass": candidate.filter_pass,
                    "q_method": candidate.q_method,
                    "q_bits": candidate.q_bits,
                    "block_size": candidate.block_size,
                    "rank": candidate.rank,
                }
                for candidate in recipe
            ]
        )
        if recipe:
            replacements = {candidate.layer: candidate.weight_hat for candidate in recipe}
            metrics = evaluate_replacements(
                model,
                tokenizer,
                modules,
                baseline_weights,
                replacements,
                texts=eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=device,
                eval_limit=args.eval_limit,
            )
            strategy_rows.append(
                {
                    "strategy": f"selector_target{target:.3f}",
                    "family": "selector",
                    "target_memory_ratio": target,
                    "memory_ratio": info.get("selected_memory_ratio", weighted_memory(recipe, layer_params)),
                    "nll": float(metrics["nll"]),
                    "perplexity": float(metrics["perplexity"]),
                    "tokens": int(metrics["tokens"]),
                    "ppl_delta": float(metrics["perplexity"]) - float(baseline["perplexity"]),
                    "selected_mean_score": info.get("selected_mean_score", 0.0),
                    "status": info.get("status", ""),
                }
            )

    write_csv(metrics_dir / "strategy_performance.csv", strategy_rows)
    write_json(
        metrics_dir / "selected_recipe.json",
        {
            **selected_payload,
            "config": {
                "model": args.model,
                "selected_layers": list(modules),
                "memory_ratios": budgets,
                "splits": splits,
                "q_methods": q_methods,
                "q_bits_list": q_bits_list,
                "block_sizes": block_sizes,
                "overlap_threshold": args.overlap_threshold,
            },
        },
    )
    plot_outputs(figures_dir, candidate_rows, strategy_rows)
    write_summary(root / "summary.md", args=args, baseline=baseline, strategy_rows=strategy_rows, candidate_rows=candidate_rows, selected_recipe=selected_payload)
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
    restore_weights(modules, baseline_weights)
    print(root)


if __name__ == "__main__":
    main()
