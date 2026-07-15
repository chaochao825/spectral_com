from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DMDResult:
    eigenvalues: np.ndarray
    rank: int
    residual: float


def transition_pairs_from_sequences(sequences: np.ndarray | list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(sequences, list):
        arrays = [np.asarray(seq, dtype=np.float64) for seq in sequences if np.asarray(seq).shape[0] >= 2]
        if not arrays:
            raise ValueError("no sequences with at least two time steps")
        x0 = np.concatenate([seq[:-1] for seq in arrays], axis=0)
        x1 = np.concatenate([seq[1:] for seq in arrays], axis=0)
        return x0, x1
    arr = np.asarray(sequences, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] < 2:
            raise ValueError("sequence has fewer than two time steps")
        return arr[:-1], arr[1:]
    if arr.ndim == 3:
        if arr.shape[1] < 2:
            raise ValueError("sequences have fewer than two time steps")
        return arr[:, :-1, :].reshape(-1, arr.shape[-1]), arr[:, 1:, :].reshape(-1, arr.shape[-1])
    raise ValueError(f"expected 2D or 3D sequence array, got {arr.shape}")


def pca_project(values: np.ndarray, rank: int, *, center: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2D matrix")
    mean = arr.mean(axis=0, keepdims=True) if center else np.zeros((1, arr.shape[1]), dtype=np.float64)
    centered = arr - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    r = min(int(rank), vt.shape[0])
    basis = vt[:r].T
    return centered @ basis, basis, mean.reshape(-1)


def exact_dmd_from_pairs(x0: np.ndarray, x1: np.ndarray, *, rank: int = 64, eps: float = 1e-10) -> DMDResult:
    a = np.asarray(x0, dtype=np.float64)
    b = np.asarray(x1, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"expected matching 2D transition matrices, got {a.shape} and {b.shape}")
    if a.shape[0] < 2:
        raise ValueError("need at least two transition pairs")
    mean = a.mean(axis=0, keepdims=True)
    x = (a - mean).T
    y = (b - mean).T
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    keep = np.where(s > eps)[0]
    if keep.size == 0:
        raise ValueError("transition matrix is numerically rank deficient")
    r = min(int(rank), int(keep.size))
    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]
    a_tilde = u_r.T @ y @ vh_r.T @ np.diag(1.0 / s_r)
    eigvals = np.linalg.eigvals(a_tilde)
    y_hat = (u_r @ a_tilde @ u_r.T @ x).T + mean
    residual = float(np.linalg.norm((b - y_hat), ord="fro") / max(np.linalg.norm(b, ord="fro"), eps))
    return DMDResult(eigenvalues=eigvals, rank=r, residual=residual)


def exact_dmd(sequences: np.ndarray | list[np.ndarray], *, rank: int = 64) -> DMDResult:
    x0, x1 = transition_pairs_from_sequences(sequences)
    return exact_dmd_from_pairs(x0, x1, rank=rank)


def summarize_dmd_eigenvalues(eigenvalues: np.ndarray, *, eps: float = 1e-8) -> dict[str, float | int]:
    vals = np.asarray(eigenvalues, dtype=np.complex128).reshape(-1)
    if vals.size == 0:
        return {
            "dmd_count": 0,
            "dmd_mean_radius": float("nan"),
            "dmd_max_radius": float("nan"),
            "dmd_real_ratio": float("nan"),
            "dmd_rotation_score": float("nan"),
            "dmd_decay_score": float("nan"),
        }
    radii = np.abs(vals)
    angles = np.abs(np.angle(vals))
    real_ratio = float(np.mean(np.abs(vals.imag) <= eps))
    return {
        "dmd_count": int(vals.size),
        "dmd_mean_radius": float(radii.mean()),
        "dmd_max_radius": float(radii.max()),
        "dmd_real_ratio": real_ratio,
        "dmd_rotation_score": float(angles.mean()),
        "dmd_decay_score": float(np.mean(np.abs(1.0 - radii))),
    }

