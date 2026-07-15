from __future__ import annotations

from typing import Iterable

import numpy as np

from .fit_powerlaw import PowerLawFit


def covariance_eigenspectrum(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    cov = np.asarray(matrix, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"expected square covariance matrix, got {cov.shape}")
    if not np.isfinite(cov).all():
        raise ValueError("covariance contains NaN or inf")
    sym = 0.5 * (cov + cov.T)
    vals = np.linalg.eigvalsh(sym)
    vals = np.where(vals < eps, np.maximum(vals, 0.0), vals)
    vals.sort()
    return vals[::-1]


def normalized_eigenvalues(eigenvalues: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vals = np.maximum(vals, 0.0)
    total = float(vals.sum())
    if total <= eps:
        return np.zeros_like(vals)
    return vals / total


def explained_variance_at_k(eigenvalues: np.ndarray, ks: Iterable[int] = (1, 5, 10, 50, 100)) -> dict[str, float]:
    p = normalized_eigenvalues(eigenvalues)
    out: dict[str, float] = {}
    cdf = np.cumsum(p)
    for k in ks:
        kk = min(max(int(k), 1), int(p.size))
        out[f"top_{k}_explained_variance"] = float(cdf[kk - 1]) if p.size else float("nan")
    return out


def participation_ratio(eigenvalues: np.ndarray, *, eps: float = 1e-15) -> float:
    vals = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    denom = float(np.dot(vals, vals))
    if denom <= eps:
        return float("nan")
    return float(vals.sum() ** 2 / denom)


def spectral_entropy(eigenvalues: np.ndarray, *, eps: float = 1e-15, normalized: bool = True) -> float:
    p = normalized_eigenvalues(eigenvalues, eps=eps)
    p = p[p > eps]
    if p.size == 0:
        return float("nan")
    entropy = float(-np.sum(p * np.log(p)))
    if normalized and p.size > 1:
        entropy /= float(np.log(p.size))
    return entropy


def effective_rank(eigenvalues: np.ndarray, *, eps: float = 1e-15) -> float:
    p = normalized_eigenvalues(eigenvalues, eps=eps)
    p = p[p > eps]
    if p.size == 0:
        return float("nan")
    return float(np.exp(-np.sum(p * np.log(p))))


def condition_number(eigenvalues: np.ndarray, *, eps: float = 1e-12) -> float:
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > eps)]
    if vals.size == 0:
        return float("inf")
    return float(vals.max() / vals.min())


def anisotropy_score(samples: np.ndarray, *, max_samples: int = 2048, eps: float = 1e-12) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    centered = arr - arr.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    valid = norms > eps
    centered = centered[valid]
    norms = norms[valid]
    if centered.shape[0] < 2:
        return float("nan")
    unit = centered / norms[:, None]
    sim = unit @ unit.T
    n = sim.shape[0]
    off_diag_sum = float(sim.sum() - np.trace(sim))
    return off_diag_sum / float(n * (n - 1))


def outlier_score(samples: np.ndarray, *, eps: float = 1e-12) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    channel_magnitude = np.mean(np.abs(arr), axis=0)
    median = float(np.median(channel_magnitude))
    if median <= eps:
        return float("inf")
    return float(np.max(channel_magnitude) / median)


def summarize_spectrum(
    eigenvalues: np.ndarray,
    *,
    powerlaw: PowerLawFit | None = None,
    samples: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    out: dict[str, float | int | None] = {
        "rank": int(vals.size),
        "trace": float(np.maximum(vals, 0.0).sum()),
        "lambda_max": float(vals.max()) if vals.size else float("nan"),
        "lambda_min_positive": float(vals[vals > 1e-12].min()) if np.any(vals > 1e-12) else float("nan"),
        "participation_ratio": participation_ratio(vals),
        "effective_rank": effective_rank(vals),
        "spectral_entropy": spectral_entropy(vals),
        "condition_number": condition_number(vals),
    }
    out.update(explained_variance_at_k(vals))
    if powerlaw is not None:
        out.update(
            {
                "alpha": float(powerlaw.alpha),
                "alpha_ci_low": powerlaw.ci_low,
                "alpha_ci_high": powerlaw.ci_high,
                "alpha_r2": float(powerlaw.r2),
                "alpha_rank_min": int(powerlaw.rank_min),
                "alpha_rank_max": int(powerlaw.rank_max),
            }
        )
    if samples is not None:
        out["anisotropy_score"] = anisotropy_score(samples)
        out["outlier_score"] = outlier_score(samples)
    return out

