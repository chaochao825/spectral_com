from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


EPS = 1e-12


def hessian_inner(delta_a: np.ndarray, delta_b: np.ndarray, hessian_diag: np.ndarray) -> float:
    """Diagonal-Hessian bilinear form used by the compression overlap MVP."""

    a = np.asarray(delta_a, dtype=np.float64)
    b = np.asarray(delta_b, dtype=np.float64)
    h = np.asarray(hessian_diag, dtype=np.float64)
    if a.shape != b.shape or a.shape != h.shape:
        raise ValueError(f"shape mismatch: delta_a={a.shape}, delta_b={b.shape}, hessian_diag={h.shape}")
    return float(np.sum(a * h * b))


def hessian_cosine(delta_a: np.ndarray, delta_b: np.ndarray, hessian_diag: np.ndarray) -> float:
    numerator = hessian_inner(delta_a, delta_b, hessian_diag)
    norm_a = math.sqrt(max(hessian_inner(delta_a, delta_a, hessian_diag), 0.0))
    norm_b = math.sqrt(max(hessian_inner(delta_b, delta_b, hessian_diag), 0.0))
    denom = max(norm_a * norm_b, EPS)
    value = float(numerator / denom)
    if not math.isfinite(value):
        return value
    return max(-1.0, min(1.0, value))


def parameter_cosine(delta_a: np.ndarray, delta_b: np.ndarray) -> float:
    a = np.asarray(delta_a, dtype=np.float64)
    b = np.asarray(delta_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: delta_a={a.shape}, delta_b={b.shape}")
    return float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def empirical_additivity_error(loss_base: float, loss_i: float, loss_j: float, loss_ij: float, *, eps: float = EPS) -> float:
    numerator = float(loss_ij) - float(loss_i) - float(loss_j) + float(loss_base)
    denom = abs(float(loss_i) - float(loss_base)) + abs(float(loss_j) - float(loss_base)) + float(eps)
    return float(numerator / denom)


def rankdata(values: Sequence[float]) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError("rankdata expects a one-dimensional sequence")
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(vals.size, dtype=np.float64)
    index = 0
    while index < vals.size:
        end = index + 1
        while end < vals.size and vals[order[end]] == vals[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearmanr(x: Sequence[float], y: Sequence[float]) -> tuple[float, int]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: x={left.shape}, y={right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2:
        return float("nan"), int(left.size)
    rx = rankdata(left)
    ry = rankdata(right)
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= EPS:
        return float("nan"), int(left.size)
    return float(np.dot(rx, ry) / denom), int(left.size)


def spectrum_summary(weight: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"spectrum_summary expects a matrix, got shape {matrix.shape}")
    values = np.linalg.svd(matrix, compute_uv=False)
    energy = values * values
    total = float(np.sum(energy))
    if total <= EPS:
        return {
            "sigma_max": 0.0,
            "stable_rank": 0.0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "spectral_entropy": 0.0,
            "top1_energy": 0.0,
        }
    probs = energy / total
    cdf = np.cumsum(probs)
    return {
        "sigma_max": float(values[0]) if values.size else 0.0,
        "stable_rank": float(total / max(float(values[0] ** 2), EPS)) if values.size else 0.0,
        "rank_90": int(np.searchsorted(cdf, 0.90) + 1),
        "rank_95": int(np.searchsorted(cdf, 0.95) + 1),
        "rank_99": int(np.searchsorted(cdf, 0.99) + 1),
        "spectral_entropy": float(-np.sum(probs * np.log(np.maximum(probs, EPS)))),
        "top1_energy": float(probs[0]) if probs.size else 0.0,
    }
