from __future__ import annotations

import numpy as np

from .spectral_metrics import covariance_eigenspectrum, summarize_spectrum


def _to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().to(dtype=getattr(tensor, "dtype", None)).float().numpy()
    return np.asarray(tensor)


def _iter_past_key_values(past_key_values):
    if past_key_values is None:
        return
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    for layer, pair in enumerate(past_key_values):
        if len(pair) < 2:
            continue
        yield layer, pair[0], pair[1]


def iter_kv_cache_arrays(past_key_values, *, exclude_first_tokens: int = 0):
    for layer, key, value in _iter_past_key_values(past_key_values) or []:
        for name, tensor in (("k_cache", key), ("v_cache", value)):
            arr = _to_numpy(tensor)
            if arr.ndim != 4:
                continue
            _batch, heads, seq, dim = arr.shape
            if exclude_first_tokens > 0:
                if seq <= exclude_first_tokens:
                    continue
                arr = arr[:, :, exclude_first_tokens:, :]
            for head in range(heads):
                yield int(layer), int(head), name, arr[:, head].reshape(-1, dim)


def kv_cache_spectral_rows(past_key_values) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer, key, value in _iter_past_key_values(past_key_values) or []:
        for name, tensor in (("k_cache", key), ("v_cache", value)):
            arr = _to_numpy(tensor)
            if arr.ndim != 4:
                continue
            # HF cache layout is usually [batch, heads, seq, head_dim].
            batch, heads, _seq, dim = arr.shape
            for head in range(heads):
                values = arr[:, head].reshape(-1, dim)
                if values.shape[0] < 2:
                    continue
                centered = values - values.mean(axis=0, keepdims=True)
                cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
                eig = covariance_eigenspectrum(cov)
                summary = summarize_spectrum(eig, samples=values)
                summary.update({"layer": int(layer), "head": int(head), "site": name, "samples": int(values.shape[0])})
                rows.append(summary)
    return rows


def attention_entropy(attention_probs: np.ndarray, *, eps: float = 1e-12) -> tuple[float, float]:
    probs = np.asarray(attention_probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    concentration = np.max(probs, axis=-1)
    return float(entropy.mean()), float(concentration.mean())


def low_rank_reconstruction_error(values: np.ndarray, ranks: list[int]) -> dict[int, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected [samples, dim]")
    centered = arr - arr.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    denom = float(np.linalg.norm(centered, ord="fro"))
    out: dict[int, float] = {}
    for rank in ranks:
        r = min(int(rank), s.size)
        recon = (u[:, :r] * s[:r]) @ vt[:r]
        out[int(rank)] = float(np.linalg.norm(centered - recon, ord="fro") / max(denom, 1e-12))
    return out
