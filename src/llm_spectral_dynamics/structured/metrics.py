from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from llm_spectral_dynamics.spectral_metrics import effective_rank, participation_ratio, spectral_entropy


def singular_values(weight: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float32)
    use_torch = device not in {"cpu", "numpy", ""}
    if use_torch:
        try:
            import torch

            target = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if str(target).startswith("cuda") and not torch.cuda.is_available():
                target = "cpu"
            tensor = torch.as_tensor(matrix, device=target, dtype=torch.float32)
            values = torch.linalg.svdvals(tensor)
            return values.detach().cpu().numpy().astype(np.float64)
        except Exception:
            pass
    return np.linalg.svd(matrix, compute_uv=False).astype(np.float64)


def energy_ranks(s: np.ndarray, thresholds: Iterable[float] = (0.9, 0.95, 0.99)) -> dict[str, int]:
    vals = np.asarray(s, dtype=np.float64)
    energy = vals * vals
    total = float(energy.sum())
    if total <= 0:
        return {f"rank_{int(t * 100)}": 0 for t in thresholds}
    cdf = np.cumsum(energy) / total
    return {f"rank_{int(t * 100)}": int(np.searchsorted(cdf, t) + 1) for t in thresholds}


def outlier_channel_metrics(weight: np.ndarray, *, z_threshold: float = 6.0) -> dict[str, float | int]:
    w = np.asarray(weight, dtype=np.float64)
    row_norm = np.linalg.norm(w, axis=1)
    col_norm = np.linalg.norm(w, axis=0)
    out: dict[str, float | int] = {}
    for prefix, values in (("out", row_norm), ("in", col_norm)):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        max_val = float(np.max(values)) if values.size else float("nan")
        denom = max(median, 1e-12)
        z = 0.6745 * (values - median) / max(mad, 1e-12)
        out[f"{prefix}_channel_max_over_median"] = max_val / denom
        out[f"{prefix}_channel_outlier_count"] = int(np.sum(z > z_threshold))
        top_count = max(1, int(math.ceil(0.01 * values.size)))
        out[f"{prefix}_channel_top1pct_energy_frac"] = float(np.sort(values * values)[-top_count:].sum() / max(float(np.dot(values, values)), 1e-30))
    return out


def weight_spectrum_metrics(weight: np.ndarray, s: np.ndarray | None = None) -> dict[str, float | int]:
    if s is None:
        s = singular_values(weight)
    vals = np.asarray(s, dtype=np.float64)
    eig = vals * vals
    frob2 = float(eig.sum())
    rank = int(vals.size)
    out: dict[str, float | int] = {
        "rows": int(weight.shape[0]),
        "cols": int(weight.shape[1]),
        "params": int(weight.size),
        "rank": rank,
        "sigma_max": float(vals.max()) if vals.size else float("nan"),
        "sigma_min_positive": float(vals[vals > 1e-12].min()) if np.any(vals > 1e-12) else float("nan"),
        "frobenius_norm": float(np.sqrt(max(frob2, 0.0))),
        "stable_rank": frob2 / max(float(vals[0] ** 2), 1e-30) if vals.size else float("nan"),
        "effective_rank": effective_rank(eig),
        "participation_ratio": participation_ratio(eig),
        "spectral_entropy": spectral_entropy(eig),
        "top_1_energy": float(eig[0] / max(frob2, 1e-30)) if vals.size else float("nan"),
        "top_10_energy": float(eig[: min(10, vals.size)].sum() / max(frob2, 1e-30)) if vals.size else float("nan"),
    }
    out.update(energy_ranks(vals))
    out.update(outlier_channel_metrics(weight))
    return out


def relative_fro_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    return float(np.linalg.norm(ref - est, ord="fro") / max(np.linalg.norm(ref, ord="fro"), 1e-12))


def residual_distribution_metrics(residual: np.ndarray) -> dict[str, float | int]:
    r = np.asarray(residual, dtype=np.float64)
    abs_r = np.abs(r)
    total = float(np.sum(abs_r))
    sorted_abs = np.sort(abs_r.reshape(-1))[::-1]
    nnz_1e_6 = int(np.count_nonzero(abs_r > 1e-6))
    out = {
        "residual_l1": total,
        "residual_l2": float(np.linalg.norm(r)),
        "residual_density_1e_6": float(nnz_1e_6 / max(abs_r.size, 1)),
        "residual_top_1pct_l1_frac": float(sorted_abs[: max(1, int(0.01 * sorted_abs.size))].sum() / max(total, 1e-30)),
    }
    out.update(outlier_channel_metrics(r))
    return out
