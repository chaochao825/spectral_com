from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .approximations import low_rank_approximation, low_rank_approximation_from_factors, low_rank_rank_for_budget, svd_factors
from .metrics import relative_fro_error, residual_distribution_metrics, weight_spectrum_metrics


@dataclass
class ResidualResult:
    residual_type: str
    matrix: np.ndarray
    params: int
    rank: int = 0
    channels: int = 0


def residual_budget_params(weight: np.ndarray, residual_fraction: float) -> int:
    return max(0, int(np.asarray(weight).size * float(residual_fraction)))


def empty_residual(weight: np.ndarray) -> ResidualResult:
    return ResidualResult("none", np.zeros_like(weight, dtype=np.float32), 0)


def low_rank_residual(
    residual: np.ndarray,
    params: int,
    factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    svd_device: str = "cpu",
) -> ResidualResult:
    if params <= 0:
        return empty_residual(residual)
    rank = low_rank_rank_for_budget(residual.shape, params)
    if factors is None:
        approx = low_rank_approximation(residual, rank=rank, svd_device=svd_device)
    else:
        approx = low_rank_approximation_from_factors(np.asarray(residual).shape, factors, rank=rank)
    return ResidualResult("low_rank", approx.matrix, approx.params, rank=approx.rank)


def sparse_residual(residual: np.ndarray, params: int) -> ResidualResult:
    if params <= 0:
        return empty_residual(residual)
    r = np.asarray(residual, dtype=np.float32)
    k = min(int(params), r.size)
    flat = np.zeros(r.size, dtype=np.float32)
    values = np.abs(r).reshape(-1)
    if k > 0:
        idx = np.argpartition(values, -k)[-k:]
        flat[idx] = r.reshape(-1)[idx]
    return ResidualResult("sparse", flat.reshape(r.shape), k)


def channel_residual(residual: np.ndarray, params: int) -> ResidualResult:
    if params <= 0:
        return empty_residual(residual)
    r = np.asarray(residual, dtype=np.float32)
    rows, cols = r.shape
    row_count = min(rows, max(0, params // max(cols, 1)))
    col_count = min(cols, max(0, params // max(rows, 1)))
    candidates: list[ResidualResult] = []
    if row_count > 0:
        row_norm = np.linalg.norm(r, axis=1)
        keep_rows = np.argpartition(row_norm, -row_count)[-row_count:]
        mat = np.zeros_like(r)
        mat[keep_rows, :] = r[keep_rows, :]
        candidates.append(ResidualResult("channel_out", mat, int(row_count * cols), channels=int(row_count)))
    if col_count > 0:
        col_norm = np.linalg.norm(r, axis=0)
        keep_cols = np.argpartition(col_norm, -col_count)[-col_count:]
        mat = np.zeros_like(r)
        mat[:, keep_cols] = r[:, keep_cols]
        candidates.append(ResidualResult("channel_in", mat, int(col_count * rows), channels=int(col_count)))
    if not candidates:
        return empty_residual(r)
    return min(candidates, key=lambda item: relative_fro_error(r, item.matrix))


def build_residual(
    residual: np.ndarray,
    *,
    residual_type: str,
    residual_fraction: float,
    low_rank_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    svd_device: str = "cpu",
) -> ResidualResult:
    params = residual_budget_params(residual, residual_fraction)
    kind = residual_type.lower()
    if kind == "none" or residual_fraction <= 0:
        return empty_residual(residual)
    if kind == "low_rank":
        return low_rank_residual(residual, params, factors=low_rank_factors, svd_device=svd_device)
    if kind == "sparse":
        return sparse_residual(residual, params)
    if kind in {"channel", "channel_wise", "channel-wise"}:
        return channel_residual(residual, params)
    raise ValueError(f"unknown residual type: {residual_type}")


def residual_analysis_rows(
    weight: np.ndarray,
    structured: np.ndarray,
    *,
    compression_ratio: float,
    residual_fractions: list[float],
    residual_types: list[str],
    svd_device: str = "cpu",
    residual_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, object]]:
    residual = np.asarray(weight, dtype=np.float32) - np.asarray(structured, dtype=np.float32)
    if residual_factors is None:
        residual_factors = svd_factors(residual, device=svd_device)
    residual_s = residual_factors[1]
    base_metrics = residual_distribution_metrics(residual)
    spectrum = weight_spectrum_metrics(residual, residual_s)
    rows: list[dict[str, object]] = []
    for residual_fraction in residual_fractions:
        types_for_fraction = ["none"] if float(residual_fraction) <= 0 else residual_types
        for residual_type in types_for_fraction:
            rr = build_residual(
                residual,
                residual_type=residual_type,
                residual_fraction=residual_fraction,
                low_rank_factors=residual_factors,
            )
            combined = np.asarray(structured, dtype=np.float32) + rr.matrix
            row = {
                "compression_ratio_target": float(compression_ratio),
                "residual_fraction": float(residual_fraction),
                "residual_type": rr.residual_type,
                "residual_params": int(rr.params),
                "residual_rank": int(rr.rank),
                "residual_channels": int(rr.channels),
                "relative_weight_error_after_residual": relative_fro_error(weight, combined),
                "residual_effective_rank": spectrum["effective_rank"],
                "residual_rank_90": spectrum["rank_90"],
                "residual_rank_95": spectrum["rank_95"],
                "residual_rank_99": spectrum["rank_99"],
            }
            row.update(base_metrics)
            rows.append(row)
    return rows
