from __future__ import annotations

import numpy as np


def time_shuffle_sequences(sequences: np.ndarray, *, seed: int = 0) -> np.ndarray:
    arr = np.asarray(sequences).copy()
    if arr.ndim != 3:
        raise ValueError("expected shape [batch, time, dim]")
    rng = np.random.default_rng(seed)
    for i in range(arr.shape[0]):
        perm = rng.permutation(arr.shape[1])
        arr[i] = arr[i, perm]
    return arr


def dimension_shuffle(values: np.ndarray, *, seed: int = 0) -> np.ndarray:
    arr = np.asarray(values).copy()
    if arr.ndim < 2:
        raise ValueError("expected at least 2D values")
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1, arr.shape[-1])
    for j in range(flat.shape[1]):
        flat[:, j] = flat[rng.permutation(flat.shape[0]), j]
    return flat.reshape(arr.shape)


def token_order_shuffle(input_ids: np.ndarray, *, seed: int = 0, preserve_first: bool = True) -> np.ndarray:
    arr = np.asarray(input_ids).copy()
    if arr.ndim != 2:
        raise ValueError("expected shape [batch, time]")
    rng = np.random.default_rng(seed)
    start = 1 if preserve_first else 0
    for i in range(arr.shape[0]):
        perm = rng.permutation(np.arange(start, arr.shape[1]))
        arr[i, start:] = arr[i, perm]
    return arr

