from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass
class Projection:
    matrix: np.ndarray
    params: int
    rank: int = 0
    block_size: int = 0


@dataclass
class OASRCandidate:
    layer: str
    family: str
    target_memory_ratio: float
    q_method: str
    q_bits: int
    block_size: int
    rank: int
    split_c: float
    split_l: float
    memory_ratio: float
    weight_hat: np.ndarray
    q_weight: np.ndarray
    c_res: np.ndarray
    l_res: np.ndarray
    q_error: np.ndarray
    q_plus_c_error: np.ndarray
    filter_rho: float
    filter_pass: bool
    score: float
    metrics: dict[str, float | int | str | bool]


def _as_float_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {out.shape}")
    return out


def rtn_quantize(weight: np.ndarray, bits: int = 4, group_size: int = 128) -> np.ndarray:
    w = _as_float_matrix(weight)
    qmax = max(1, 2 ** (int(bits) - 1) - 1)
    out = np.empty_like(w, dtype=np.float32)
    group = max(int(group_size), 1)
    for start in range(0, w.shape[1], group):
        end = min(start + group, w.shape[1])
        block = w[:, start:end]
        scale = np.max(np.abs(block), axis=1, keepdims=True) / float(qmax)
        scale = np.maximum(scale, 1e-12)
        out[:, start:end] = np.clip(np.round(block / scale), -qmax, qmax) * scale
    return out.astype(np.float32)


def sinq_like_quantize(weight: np.ndarray, bits: int = 4, iterations: int = 4) -> np.ndarray:
    w = _as_float_matrix(weight)
    row_scale = np.ones((w.shape[0], 1), dtype=np.float32)
    col_scale = np.ones((1, w.shape[1]), dtype=np.float32)
    balanced = w.copy()
    for _ in range(max(int(iterations), 1)):
        row = np.sqrt(np.maximum(np.mean(balanced * balanced, axis=1, keepdims=True), EPS))
        balanced = balanced / row
        row_scale *= row
        col = np.sqrt(np.maximum(np.mean(balanced * balanced, axis=0, keepdims=True), EPS))
        balanced = balanced / col
        col_scale *= col
    quantized = rtn_quantize(balanced, bits=bits, group_size=balanced.shape[1])
    return (quantized * row_scale * col_scale).astype(np.float32)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (int(value) - 1).bit_length()


def _fwht_last_dim(matrix: np.ndarray) -> np.ndarray:
    n = int(matrix.shape[-1])
    if n & (n - 1):
        raise ValueError(f"FWHT requires power-of-two last dimension, got {n}")
    out = np.asarray(matrix, dtype=np.float32).copy().reshape(-1, n)
    step = 1
    while step < n:
        out = out.reshape(-1, n // (step * 2), step * 2)
        left = out[:, :, :step].copy()
        right = out[:, :, step:].copy()
        out[:, :, :step] = left + right
        out[:, :, step:] = left - right
        out = out.reshape(-1, n)
        step *= 2
    return (out / math.sqrt(float(n))).reshape(matrix.shape)


def rotated_rtn_quantize(weight: np.ndarray, bits: int = 4) -> np.ndarray:
    w = _as_float_matrix(weight)
    cols = w.shape[1]
    padded_cols = _next_power_of_two(cols)
    padded = np.zeros((w.shape[0], padded_cols), dtype=np.float32)
    padded[:, :cols] = w
    rotated = _fwht_last_dim(padded)
    quantized = rtn_quantize(rotated, bits=bits, group_size=padded_cols)
    recovered = _fwht_last_dim(quantized)[:, :cols]
    return recovered.astype(np.float32)


def quantize_weight(weight: np.ndarray, *, method: str, bits: int, group_size: int = 128) -> np.ndarray:
    method = str(method).lower()
    if method == "rtn":
        return rtn_quantize(weight, bits=bits, group_size=group_size)
    if method in {"sinq", "sinq_like"}:
        return sinq_like_quantize(weight, bits=bits)
    if method in {"rotated_rtn", "rot"}:
        return rotated_rtn_quantize(weight, bits=bits)
    raise ValueError(f"unsupported OASR quantizer: {method}")


def block_circulant_param_count(shape: tuple[int, int], block_size: int) -> int:
    rows, cols = shape
    b = int(block_size)
    return int(math.ceil(rows / b) * math.ceil(cols / b) * b)


def block_circulant_project(weight: np.ndarray, block_size: int = 32) -> Projection:
    if int(block_size) not in {16, 32, 64}:
        raise ValueError("block_size must be one of {16, 32, 64}")
    w = _as_float_matrix(weight)
    rows, cols = w.shape
    b = int(block_size)
    pad_rows = int(math.ceil(rows / b) * b)
    pad_cols = int(math.ceil(cols / b) * b)
    out = np.zeros((pad_rows, pad_cols), dtype=np.float32)
    for row in range(0, pad_rows, b):
        for col in range(0, pad_cols, b):
            row_count = min(b, rows - row)
            col_count = min(b, cols - col)
            block = w[row : row + row_count, col : col + col_count]
            sums = np.zeros(b, dtype=np.float64)
            counts = np.zeros(b, dtype=np.int64)
            for i in range(row_count):
                for j in range(col_count):
                    shift = (j - i) % b
                    sums[shift] += float(block[i, j])
                    counts[shift] += 1
            coeff = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0).astype(np.float32)
            index = (np.arange(b)[None, :] - np.arange(b)[:, None]) % b
            out[row : row + b, col : col + b] = coeff[index]
    return Projection(out[:rows, :cols].astype(np.float32), block_circulant_param_count(w.shape, b), block_size=b)


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    p = np.asarray(perm, dtype=np.int64)
    inv = np.empty_like(p)
    inv[p] = np.arange(p.size, dtype=np.int64)
    return inv


def _validate_permutation(perm: np.ndarray, size: int, name: str) -> np.ndarray:
    p = np.asarray(perm, dtype=np.int64)
    if p.shape != (int(size),):
        raise ValueError(f"{name} permutation has shape {p.shape}, expected {(int(size),)}")
    if p.size and not np.array_equal(np.sort(p), np.arange(p.size)):
        raise ValueError(f"{name} is not a valid permutation")
    return p


def permuted_block_circulant_project(
    weight: np.ndarray,
    *,
    block_size: int = 32,
    row_perm: np.ndarray | None = None,
    col_perm: np.ndarray | None = None,
) -> Projection:
    w = _as_float_matrix(weight)
    rows, cols = w.shape
    rperm = np.arange(rows, dtype=np.int64) if row_perm is None else _validate_permutation(row_perm, rows, "row")
    cperm = np.arange(cols, dtype=np.int64) if col_perm is None else _validate_permutation(col_perm, cols, "col")
    projected = block_circulant_project(w[np.ix_(rperm, cperm)], block_size=block_size)
    restored = projected.matrix[np.ix_(inverse_permutation(rperm), inverse_permutation(cperm))]
    return Projection(restored.astype(np.float32), projected.params, block_size=projected.block_size)


def norm_sorted_permutations(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = _as_float_matrix(weight)
    row_perm = np.argsort(-np.linalg.norm(w, axis=1), kind="mergesort").astype(np.int64)
    col_perm = np.argsort(-np.linalg.norm(w, axis=0), kind="mergesort").astype(np.int64)
    return row_perm, col_perm


def activation_clustered_permutation(x: np.ndarray, *, block_size: int = 32, max_clusters: int = 64) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got shape {x_arr.shape}")
    channels = x_arr.shape[1]
    if channels <= 1:
        return np.arange(channels, dtype=np.int64)
    mean = np.mean(x_arr, axis=0)
    std = np.std(x_arr, axis=0)
    centered = x_arr - mean[None, :]
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        pc1 = vh[0] if vh.shape[0] >= 1 else np.zeros(channels, dtype=np.float32)
        pc2 = vh[1] if vh.shape[0] >= 2 else np.zeros(channels, dtype=np.float32)
    except np.linalg.LinAlgError:
        pc1 = np.zeros(channels, dtype=np.float32)
        pc2 = np.zeros(channels, dtype=np.float32)
    features = np.stack([mean, std, pc1, pc2], axis=1).astype(np.float32)
    scale = np.std(features, axis=0, keepdims=True)
    features = (features - np.mean(features, axis=0, keepdims=True)) / np.maximum(scale, 1e-6)
    clusters = min(max(2, int(math.ceil(channels / max(int(block_size), 1)))), int(max_clusters), channels)
    order = np.argsort(features[:, 2], kind="mergesort")
    init_idx = np.linspace(0, channels - 1, clusters).round().astype(np.int64)
    centroids = features[order[init_idx]].copy()
    labels = np.zeros(channels, dtype=np.int64)
    for _ in range(8):
        dist = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dist, axis=1)
        for idx in range(clusters):
            members = features[labels == idx]
            if members.size:
                centroids[idx] = np.mean(members, axis=0)
    cluster_order = np.argsort(centroids[:, 2], kind="mergesort")
    pieces: list[np.ndarray] = []
    for cluster in cluster_order:
        members = np.flatnonzero(labels == cluster)
        if members.size:
            local = members[np.lexsort((features[members, 0], features[members, 3]))]
            pieces.append(local.astype(np.int64))
    return np.concatenate(pieces).astype(np.int64) if pieces else np.arange(channels, dtype=np.int64)


def interleaved_permutation(size: int, block_size: int) -> np.ndarray:
    size = int(size)
    stride = max(int(block_size), 1)
    pieces = [np.arange(offset, size, stride, dtype=np.int64) for offset in range(stride)]
    pieces = [piece for piece in pieces if piece.size]
    return np.concatenate(pieces).astype(np.int64) if pieces else np.arange(size, dtype=np.int64)


def monarch_like_two_block_project(weight: np.ndarray, block_size: int = 32) -> Projection:
    w = _as_float_matrix(weight)
    first = block_circulant_project(w, block_size=block_size)
    row_perm = interleaved_permutation(w.shape[0], block_size)
    col_perm = interleaved_permutation(w.shape[1], block_size)
    second = permuted_block_circulant_project(w - first.matrix, block_size=block_size, row_perm=row_perm, col_perm=col_perm)
    return Projection((first.matrix + second.matrix).astype(np.float32), first.params + second.params, block_size=block_size)


def lowrank_project(weight: np.ndarray, rank: int) -> Projection:
    w = _as_float_matrix(weight)
    r = max(0, min(int(rank), min(w.shape)))
    if r <= 0:
        return Projection(np.zeros_like(w, dtype=np.float32), 0, rank=0)
    u, s, vh = np.linalg.svd(w, full_matrices=False)
    approx = (u[:, :r] * s[:r]) @ vh[:r, :]
    return Projection(approx.astype(np.float32), int(r * (w.shape[0] + w.shape[1])), rank=r)


def lowrank_rank_for_memory(shape: tuple[int, int], memory_ratio: float) -> int:
    rows, cols = shape
    params = int(max(float(memory_ratio), 0.0) * rows * cols)
    return max(0, min(rows, cols, params // max(rows + cols, 1)))


def estimate_memory(
    shape_or_weight: tuple[int, int] | np.ndarray,
    *,
    q_bits: int = 0,
    c_params: int = 0,
    l_rank: int = 0,
    l_params: int | None = None,
) -> float:
    shape = tuple(np.asarray(shape_or_weight).shape) if not isinstance(shape_or_weight, tuple) else shape_or_weight
    rows, cols = int(shape[0]), int(shape[1])
    dense = max(rows * cols, 1)
    q_ratio = float(q_bits) / 16.0 if int(q_bits) > 0 else 0.0
    lr_params = int(l_params) if l_params is not None else int(max(l_rank, 0) * (rows + cols))
    return q_ratio + (int(c_params) + lr_params) / float(dense)


def compute_activation_error(x: np.ndarray, weight: np.ndarray, weight_hat: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    w = _as_float_matrix(weight)
    wh = _as_float_matrix(weight_hat)
    ref = x_arr @ w.T
    err = x_arr @ (wh - w).T
    return float(np.sum(err * err) / max(float(np.sum(ref * ref)), EPS))


def token_error_ratios(x: np.ndarray, weight: np.ndarray, weight_hat: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    w = _as_float_matrix(weight)
    wh = _as_float_matrix(weight_hat)
    ref = x_arr @ w.T
    err = x_arr @ (wh - w).T
    return np.sum(err * err, axis=1) / np.maximum(np.sum(ref * ref, axis=1), EPS)


def worst_token_p95_error(x: np.ndarray, weight: np.ndarray, weight_hat: np.ndarray) -> float:
    ratios = token_error_ratios(x, weight, weight_hat)
    return float(np.quantile(ratios, 0.95)) if ratios.size else float("nan")


def compute_hessian_cost_norm(x: np.ndarray, weight: np.ndarray, weight_hat: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    w = _as_float_matrix(weight)
    delta = _as_float_matrix(weight_hat) - w
    xd = x_arr @ delta.T
    ref = x_arr @ w.T
    return float(np.sum(xd * xd) / max(float(np.sum(ref * ref)), EPS))


def compute_conditional_overlap(x: np.ndarray, delta_a: np.ndarray, delta_b: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    a = x_arr @ _as_float_matrix(delta_a).T
    b = x_arr @ _as_float_matrix(delta_b).T
    numerator = float(np.sum(a * b))
    denom = math.sqrt(max(float(np.sum(a * a)), 0.0)) * math.sqrt(max(float(np.sum(b * b)), 0.0))
    if denom <= EPS:
        return 0.0
    value = numerator / denom
    return max(-1.0, min(1.0, float(value)))


def relative_fro_error(reference: np.ndarray, approx: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    err = np.asarray(approx, dtype=np.float32) - ref
    return float(np.linalg.norm(err) / max(float(np.linalg.norm(ref)), EPS))


def stable_rank(matrix: np.ndarray) -> float:
    m = _as_float_matrix(matrix)
    if not np.any(m):
        return 0.0
    s = np.linalg.svd(m, compute_uv=False)
    return float(np.sum(s * s) / max(float(s[0] * s[0]), EPS))


def effective_rank(matrix: np.ndarray) -> float:
    m = _as_float_matrix(matrix)
    if not np.any(m):
        return 0.0
    s = np.linalg.svd(m, compute_uv=False)
    energy = s * s
    p = energy / max(float(np.sum(energy)), EPS)
    entropy = -float(np.sum(p * np.log(np.maximum(p, EPS))))
    return float(math.exp(entropy))


def random_block_circulant_baseline(weight: np.ndarray, block_size: int, *, seed: int = 0) -> Projection:
    w = _as_float_matrix(weight)
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(w.reshape(-1)).reshape(w.shape).astype(np.float32)
    projected = block_circulant_project(permuted, block_size=block_size)
    scale = np.linalg.norm(w) / max(float(np.linalg.norm(projected.matrix)), EPS)
    return Projection((projected.matrix * scale).astype(np.float32), projected.params, block_size=block_size)


def score_candidate(metrics: dict[str, float | int | str | bool]) -> float:
    return float(metrics["activation_error"]) + 0.5 * float(metrics["worst_token_p95_error"]) + 0.2 * float(metrics["hessian_cost_norm"])


def candidate_metrics(
    *,
    layer: str,
    family: str,
    target_memory_ratio: float,
    q_method: str,
    q_bits: int,
    block_size: int,
    split_c: float,
    split_l: float,
    x: np.ndarray,
    weight: np.ndarray,
    q_weight: np.ndarray,
    c_res: np.ndarray,
    l_res: np.ndarray,
    l_rank: int = 0,
    c_params: int = 0,
) -> dict[str, float | int | str | bool]:
    w = _as_float_matrix(weight)
    what = (q_weight + c_res + l_res).astype(np.float32)
    q_error = q_weight - w
    qc_error = q_weight + c_res - w
    effective_l_rank = int(l_rank) if np.any(l_res) else 0
    effective_c_params = int(c_params) if np.any(c_res) else 0
    memory_ratio = estimate_memory(w.shape, q_bits=q_bits, c_params=effective_c_params, l_rank=effective_l_rank)
    c_projection_error = relative_fro_error(w - q_weight, c_res) if np.any(c_res) else float("nan")
    l_projection_error = relative_fro_error(w - q_weight - c_res, l_res) if np.any(l_res) else float("nan")
    rho_q_c = compute_conditional_overlap(x, q_error, c_res) if np.any(c_res) else 0.0
    rho_q_l = compute_conditional_overlap(x, q_error, l_res) if np.any(l_res) else 0.0
    rho_qc_l = compute_conditional_overlap(x, qc_error, l_res) if np.any(l_res) else 0.0
    rho_c_l = compute_conditional_overlap(x, c_res, l_res) if np.any(c_res) and np.any(l_res) else 0.0
    metrics: dict[str, float | int | str | bool] = {
        "layer": layer,
        "family": family,
        "target_memory_ratio": float(target_memory_ratio),
        "q_method": q_method,
        "q_bits": int(q_bits),
        "block_size": int(block_size),
        "rank": int(effective_l_rank),
        "split_c": float(split_c),
        "split_l": float(split_l),
        "memory_ratio": float(memory_ratio),
        "weight_error": relative_fro_error(w, what),
        "activation_error": compute_activation_error(x, w, what),
        "worst_token_p95_error": worst_token_p95_error(x, w, what),
        "hessian_cost_norm": compute_hessian_cost_norm(x, w, what),
        "rho_q_error_c_res": rho_q_c,
        "rho_q_error_l_res": rho_q_l,
        "rho_q_plus_c_error_l_res": rho_qc_l,
        "rho_c_res_l_res": rho_c_l,
        "residual_effective_rank": effective_rank(w - q_weight),
        "residual_stable_rank": stable_rank(w - q_weight),
        "block_circulant_projection_error": c_projection_error,
        "lowrank_projection_error_at_matched_memory": l_projection_error,
    }
    metrics["score"] = score_candidate(metrics)
    return metrics


def feasible_block_sizes(shape: tuple[int, int], max_memory_ratio: float, block_sizes: Iterable[int]) -> list[int]:
    return [
        int(size)
        for size in block_sizes
        if int(size) in {16, 32, 64} and block_circulant_param_count(shape, int(size)) / max(shape[0] * shape[1], 1) <= float(max_memory_ratio) + 1e-12
    ]
