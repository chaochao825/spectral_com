from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_2d_float64(values: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    if arr.shape[0] == 0:
        return arr
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or inf")
    return arr


@dataclass
class RunningMoments:
    """Streaming mean and variance for scalar values."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray | Iterable[float]) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        if not np.isfinite(arr).all():
            raise ValueError("scalar moments input contains NaN or inf")
        batch_count = int(arr.size)
        batch_mean = float(arr.mean())
        centered = arr - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total

    @property
    def variance(self) -> float:
        if self.count < 2:
            return float("nan")
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


class RunningCovariance:
    """Float64 streaming covariance with optional bounded reservoir samples."""

    def __init__(
        self,
        dim: int | None = None,
        *,
        sample_limit: int = 0,
        seed: int = 0,
    ) -> None:
        self.dim = dim
        self.count = 0
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.sample_limit = int(sample_limit)
        self.store_full_covariance = True
        self._rng = np.random.default_rng(seed)
        self._reservoir: list[np.ndarray] = []
        self._seen_for_reservoir = 0

    def update(self, values: np.ndarray | Iterable[float]) -> None:
        arr = _as_2d_float64(values)
        if arr.shape[0] == 0:
            return
        if self.dim is None:
            self.dim = int(arr.shape[1])
            self.mean = np.zeros(self.dim, dtype=np.float64)
            self.store_full_covariance = not (self.sample_limit > 0 and self.dim > self.sample_limit)
            self.m2 = np.zeros((self.dim, self.dim), dtype=np.float64) if self.store_full_covariance else None
        if arr.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {arr.shape[1]}")
        assert self.mean is not None

        self._update_reservoir(arr)

        batch_count = int(arr.shape[0])
        batch_mean = arr.mean(axis=0)
        if not self.store_full_covariance:
            if self.count == 0:
                self.count = batch_count
                self.mean = batch_mean
                return
            total = self.count + batch_count
            self.mean = self.mean + (batch_mean - self.mean) * batch_count / total
            self.count = total
            return

        assert self.m2 is not None
        centered = arr - batch_mean
        batch_m2 = centered.T @ centered

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + np.outer(delta, delta) * self.count * batch_count / total
        self.mean = self.mean + delta * batch_count / total
        self.count = total

    def _update_reservoir(self, arr: np.ndarray) -> None:
        if self.sample_limit <= 0:
            return
        for row in arr:
            seen = self._seen_for_reservoir
            self._seen_for_reservoir += 1
            if len(self._reservoir) < self.sample_limit:
                self._reservoir.append(row.copy())
                continue
            j = int(self._rng.integers(0, seen + 1))
            if j < self.sample_limit:
                self._reservoir[j] = row.copy()

    def covariance(self, *, ddof: int = 1) -> np.ndarray:
        if self.count <= ddof:
            raise ValueError(f"insufficient samples for covariance: count={self.count}, ddof={ddof}")
        if self.m2 is None:
            raise ValueError("full covariance was not stored; use reservoir sample-space eigenspectrum")
        return self.m2 / (self.count - ddof)

    def correlation(self, *, eps: float = 1e-12) -> np.ndarray:
        cov = self.covariance()
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))
        denom = np.outer(std, std)
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > eps)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def sample_array(self) -> np.ndarray:
        if not self._reservoir:
            if self.dim is None:
                return np.empty((0, 0), dtype=np.float64)
            return np.empty((0, self.dim), dtype=np.float64)
        return np.stack(self._reservoir, axis=0)

    def state_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "count": self.count,
            "mean": None if self.mean is None else self.mean.copy(),
            "m2": None if self.m2 is None else self.m2.copy(),
            "sample_limit": self.sample_limit,
            "store_full_covariance": self.store_full_covariance,
        }


def sample_space_covariance_eigenvalues(values: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Eigenvalues of feature covariance computed through sample-space SVD."""

    arr = _as_2d_float64(values)
    if arr.shape[0] < 2:
        raise ValueError("need at least two samples")
    if center:
        arr = arr - arr.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(arr, full_matrices=False)
    eig = (singular_values**2) / (arr.shape[0] - 1)
    eig.sort()
    return eig[::-1]
