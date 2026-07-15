from __future__ import annotations

import numpy as np


def pca_basis(values: np.ndarray, *, rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected [samples, dim]")
    mean = arr.mean(axis=0, keepdims=True)
    centered = arr - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    r = vt.shape[0] if rank is None else min(int(rank), vt.shape[0])
    return vt[:r].T, mean.reshape(-1)


def remove_pc_subspace(values: np.ndarray, basis: np.ndarray, *, which: str = "top", rank: int = 1, attenuation: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    pcs = np.asarray(basis, dtype=np.float64)
    if arr.ndim != 2 or pcs.ndim != 2:
        raise ValueError("expected 2D values and basis")
    if pcs.shape[0] != arr.shape[1]:
        raise ValueError("basis feature dimension does not match values")
    r = min(int(rank), pcs.shape[1])
    if which == "top":
        sub = pcs[:, :r]
    elif which == "tail":
        sub = pcs[:, -r:]
    else:
        raise ValueError("which must be 'top' or 'tail'")
    projection = arr @ sub @ sub.T
    return arr - float(attenuation) * projection


def intervention_loss_delta_rows(baseline_loss: float, intervened: dict[str, float]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, loss in intervened.items():
        rows.append({"intervention": name, "loss": float(loss), "loss_delta": float(loss - baseline_loss)})
    return rows

