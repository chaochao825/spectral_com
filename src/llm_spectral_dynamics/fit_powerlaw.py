from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    intercept: float
    r2: float
    rank_min: int
    rank_max: int
    n_points: int
    ci_low: float | None = None
    ci_high: float | None = None


def _rank_window(
    eigenvalues: np.ndarray,
    rank_min: int,
    rank_max: int | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > eps)]
    if vals.size == 0:
        raise ValueError("no positive finite eigenvalues")
    vals = np.sort(vals)[::-1]
    lo = max(int(rank_min), 1)
    hi = int(rank_max) if rank_max is not None else int(vals.size)
    hi = min(max(hi, lo), int(vals.size))
    window = vals[lo - 1 : hi]
    ranks = np.arange(lo, hi + 1, dtype=np.float64)
    if window.size < 2:
        raise ValueError("rank window has fewer than two points")
    return np.log(ranks), np.log(window), lo, hi


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n < 2:
        raise ValueError("need at least two points")
    if n > 600:
        return float(np.polyfit(x, y, deg=1)[0])
    slopes: list[float] = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        valid = np.abs(dx) > 1e-15
        if np.any(valid):
            slopes.extend(((y[i + 1 :][valid] - y[i]) / dx[valid]).tolist())
    if not slopes:
        raise ValueError("all x values are identical")
    return float(np.median(np.asarray(slopes, dtype=np.float64)))


def fit_powerlaw(
    eigenvalues: np.ndarray,
    *,
    rank_min: int = 2,
    rank_max: int | None = None,
    eps: float = 1e-15,
) -> PowerLawFit:
    """Fit lambda_rank ~= exp(intercept) * rank^(-alpha) on a log-log window."""

    x, y, lo, hi = _rank_window(eigenvalues, rank_min, rank_max, eps)
    slope = _theil_sen_slope(x, y)
    intercept = float(np.median(y - slope * x))
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot <= eps else 1.0 - ss_res / ss_tot
    return PowerLawFit(
        alpha=float(-slope),
        intercept=intercept,
        r2=float(r2),
        rank_min=lo,
        rank_max=hi,
        n_points=int(x.size),
    )


def bootstrap_powerlaw_ci(
    eigenvalues: np.ndarray,
    *,
    rank_min: int = 2,
    rank_max: int | None = None,
    n_boot: int = 200,
    seed: int = 0,
    eps: float = 1e-15,
    quantiles: tuple[float, float] = (2.5, 97.5),
) -> PowerLawFit:
    base = fit_powerlaw(eigenvalues, rank_min=rank_min, rank_max=rank_max, eps=eps)
    x, y, _, _ = _rank_window(eigenvalues, base.rank_min, base.rank_max, eps)
    rng = np.random.default_rng(seed)
    alphas: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, x.size, size=x.size)
        xb = x[idx]
        yb = y[idx]
        order = np.argsort(xb)
        xb = xb[order]
        yb = yb[order]
        try:
            slope = _theil_sen_slope(xb, yb)
        except ValueError:
            continue
        alphas.append(float(-slope))
    if not alphas:
        return base
    low, high = np.percentile(np.asarray(alphas), quantiles)
    return PowerLawFit(
        alpha=base.alpha,
        intercept=base.intercept,
        r2=base.r2,
        rank_min=base.rank_min,
        rank_max=base.rank_max,
        n_points=base.n_points,
        ci_low=float(low),
        ci_high=float(high),
    )

