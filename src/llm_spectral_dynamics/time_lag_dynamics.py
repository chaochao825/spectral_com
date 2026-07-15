from __future__ import annotations

from typing import Iterable

import numpy as np

from .dmd import exact_dmd_from_pairs, pca_project, summarize_dmd_eigenvalues, transition_pairs_from_sequences


def _as_sequence_list(sequences: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    if isinstance(sequences, list):
        return [np.asarray(seq, dtype=np.float64) for seq in sequences]
    arr = np.asarray(sequences, dtype=np.float64)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"expected 2D/3D sequences, got {arr.shape}")


def project_sequences_to_pcs(
    sequences: np.ndarray | list[np.ndarray],
    *,
    rank: int = 64,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    seqs = _as_sequence_list(sequences)
    stacked = np.concatenate(seqs, axis=0)
    projected, basis, mean = pca_project(stacked, rank=rank, center=True)
    out: list[np.ndarray] = []
    offset = 0
    for seq in seqs:
        out.append(projected[offset : offset + seq.shape[0]])
        offset += seq.shape[0]
    return out, basis, mean


def time_lag_covariance(sequences: np.ndarray | list[np.ndarray], tau: int) -> np.ndarray:
    if tau <= 0:
        raise ValueError("tau must be positive")
    seqs = _as_sequence_list(sequences)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for seq in seqs:
        if seq.shape[0] > tau:
            left.append(seq[tau:])
            right.append(seq[:-tau])
    if not left:
        raise ValueError(f"no sequence is longer than tau={tau}")
    x = np.concatenate(left, axis=0)
    y = np.concatenate(right, axis=0)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    return (x.T @ y) / max(x.shape[0] - 1, 1)


def pc_autocorrelation(
    sequences: np.ndarray | list[np.ndarray],
    lags: Iterable[int],
    *,
    eps: float = 1e-12,
) -> dict[int, np.ndarray]:
    seqs = _as_sequence_list(sequences)
    out: dict[int, np.ndarray] = {}
    for tau in lags:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for seq in seqs:
            if seq.shape[0] > tau:
                xs.append(seq[:-tau])
                ys.append(seq[tau:])
        if not xs:
            continue
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        denom = x.std(axis=0, ddof=1) * y.std(axis=0, ddof=1)
        corr = np.divide(np.mean(x * y, axis=0), denom, out=np.zeros(x.shape[1], dtype=np.float64), where=denom > eps)
        out[int(tau)] = corr
    return out


def fit_autocorrelation_decay(autocorr: dict[int, np.ndarray], *, eps: float = 1e-8) -> np.ndarray:
    if not autocorr:
        return np.empty((0,), dtype=np.float64)
    lags = np.asarray(sorted(autocorr.keys()), dtype=np.float64)
    vals = np.stack([np.abs(autocorr[int(t)]) for t in lags], axis=0)
    timescales = np.full(vals.shape[1], np.nan, dtype=np.float64)
    for i in range(vals.shape[1]):
        y = vals[:, i]
        valid = y > eps
        if valid.sum() < 2:
            continue
        slope = float(np.polyfit(lags[valid], np.log(y[valid]), deg=1)[0])
        if slope < -eps:
            timescales[i] = -1.0 / slope
    return timescales


def dynamic_summary_rows(
    sequences: np.ndarray | list[np.ndarray],
    *,
    lags: Iterable[int] = (1, 2, 4, 8, 16, 32),
    pca_rank: int = 64,
    dmd_rank: int | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    projected, _, _ = project_sequences_to_pcs(sequences, rank=pca_rank)
    autocorr = pc_autocorrelation(projected, lags)
    timescales = fit_autocorrelation_decay(autocorr)
    rows: list[dict[str, float | int]] = []
    for tau, corr in autocorr.items():
        rows.append(
            {
                "tau": int(tau),
                "mean_abs_pc_autocorr": float(np.mean(np.abs(corr))),
                "max_abs_pc_autocorr": float(np.max(np.abs(corr))),
                "pc1_autocorr": float(corr[0]) if corr.size else float("nan"),
            }
        )
    x0, x1 = transition_pairs_from_sequences(projected)
    dmd_result = exact_dmd_from_pairs(x0, x1, rank=dmd_rank or pca_rank)
    dmd_summary = summarize_dmd_eigenvalues(dmd_result.eigenvalues)
    dmd_summary["dmd_rank"] = int(dmd_result.rank)
    dmd_summary["dmd_residual"] = float(dmd_result.residual)
    dmd_summary["mean_pc_timescale"] = float(np.nanmean(timescales)) if np.any(np.isfinite(timescales)) else float("nan")
    return rows, dmd_summary
