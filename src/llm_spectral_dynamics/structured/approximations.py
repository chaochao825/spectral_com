from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import numpy as np

from .metrics import relative_fro_error


LOGGER = logging.getLogger(__name__)
_LOGGED_SVD_BACKENDS: set[str] = set()


@dataclass
class ApproximationResult:
    method: str
    matrix: np.ndarray
    params: int
    rank: int
    block_size: int | None = None
    terms: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(x) for x in self.matrix.shape)


def original_params(weight: np.ndarray) -> int:
    return int(np.asarray(weight).size)


def budget_params(weight: np.ndarray, compression_ratio: float) -> int:
    return max(1, int(np.asarray(weight).size / float(compression_ratio)))


def low_rank_rank_for_budget(shape: tuple[int, int], params: int) -> int:
    rows, cols = shape
    return max(1, min(rows, cols, int(params // max(rows + cols, 1))))


def svd_factors(weight: np.ndarray, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(weight, dtype=np.float32)
    if device not in {"cpu", "numpy", ""}:
        try:
            import torch

            target = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if str(target).startswith("cuda") and not torch.cuda.is_available():
                if os.environ.get("LLM_SC_SVD_FAIL_FAST", "0") == "1":
                    raise RuntimeError(f"requested SVD backend {target} is unavailable")
                LOGGER.warning("requested SVD backend %s is unavailable; falling back to NumPy", target)
                return np.linalg.svd(matrix, full_matrices=False)
            backend = str(target)
            if backend not in _LOGGED_SVD_BACKENDS:
                LOGGER.info("using torch SVD backend %s", backend)
                _LOGGED_SVD_BACKENDS.add(backend)
            tensor = torch.as_tensor(matrix, device=target, dtype=torch.float32)
            u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
            return (
                u.detach().cpu().numpy().astype(np.float32),
                s.detach().cpu().numpy().astype(np.float32),
                vh.detach().cpu().numpy().astype(np.float32),
            )
        except Exception as exc:
            if os.environ.get("LLM_SC_SVD_FAIL_FAST", "0") == "1":
                raise
            LOGGER.warning("SVD backend %s failed; falling back to NumPy: %s", device, exc)
    if "numpy" not in _LOGGED_SVD_BACKENDS:
        LOGGER.info("using NumPy SVD backend")
        _LOGGED_SVD_BACKENDS.add("numpy")
    return np.linalg.svd(matrix, full_matrices=False)


def batched_svd_factors(blocks: np.ndarray, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = np.asarray(blocks, dtype=np.float32)
    if matrices.ndim != 3:
        raise ValueError(f"expected batched matrices with shape [batch, rows, cols], got {matrices.shape}")
    if matrices.shape[0] == 0:
        size = min(matrices.shape[1:])
        return (
            np.empty((0, matrices.shape[1], size), dtype=np.float32),
            np.empty((0, size), dtype=np.float32),
            np.empty((0, size, matrices.shape[2]), dtype=np.float32),
        )
    if device not in {"cpu", "numpy", ""}:
        try:
            import torch

            target = "cuda" if device == "auto" and torch.cuda.is_available() else device
            if str(target).startswith("cuda") and not torch.cuda.is_available():
                if os.environ.get("LLM_SC_SVD_FAIL_FAST", "0") == "1":
                    raise RuntimeError(f"requested batched SVD backend {target} is unavailable")
                LOGGER.warning("requested batched SVD backend %s is unavailable; falling back to NumPy", target)
            else:
                backend = f"{target}-batched"
                if backend not in _LOGGED_SVD_BACKENDS:
                    LOGGER.info("using torch batched SVD backend %s", target)
                    _LOGGED_SVD_BACKENDS.add(backend)
                tensor = torch.as_tensor(matrices, device=target, dtype=torch.float32)
                u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
                return (
                    u.detach().cpu().numpy().astype(np.float32),
                    s.detach().cpu().numpy().astype(np.float32),
                    vh.detach().cpu().numpy().astype(np.float32),
                )
        except Exception as exc:
            if os.environ.get("LLM_SC_SVD_FAIL_FAST", "0") == "1":
                raise
            LOGGER.warning("batched SVD backend %s failed; falling back to NumPy: %s", device, exc)
    return np.linalg.svd(matrices, full_matrices=False)


def low_rank_approximation_from_factors(
    shape: tuple[int, int],
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    params: int | None = None,
    rank: int | None = None,
) -> ApproximationResult:
    u, s, vh = factors
    if rank is None:
        if params is None:
            raise ValueError("provide params or rank for low-rank approximation")
        rank = low_rank_rank_for_budget(shape, params)
    r = max(1, min(int(rank), int(s.size)))
    approx = (u[:, :r] * s[:r]) @ vh[:r, :]
    return ApproximationResult(method="low_rank", matrix=approx.astype(np.float32), params=int(r * (shape[0] + shape[1])), rank=r)


def low_rank_approximation(
    weight: np.ndarray,
    *,
    params: int | None = None,
    rank: int | None = None,
    svd_device: str = "cpu",
) -> ApproximationResult:
    w = np.asarray(weight, dtype=np.float32)
    return low_rank_approximation_from_factors(w.shape, svd_factors(w, device=svd_device), params=params, rank=rank)


def _nearest_circulant(block: np.ndarray) -> np.ndarray:
    size = block.shape[0]
    coeff = np.zeros(size, dtype=np.float32)
    counts = np.zeros(size, dtype=np.float32)
    rows, cols = np.indices(block.shape)
    diag = (cols - rows) % size
    np.add.at(coeff, diag.reshape(-1), block.reshape(-1))
    np.add.at(counts, diag.reshape(-1), 1.0)
    coeff /= np.maximum(counts, 1.0)
    return coeff[(np.arange(size)[None, :] - np.arange(size)[:, None]) % size]


def block_circulant_param_count(shape: tuple[int, int], block_size: int) -> int:
    rows, cols = shape
    return int(np.ceil(rows / block_size) * np.ceil(cols / block_size) * block_size)


def choose_block_size(shape: tuple[int, int], budget: int, candidates: list[int]) -> int:
    ordered = sorted({int(x) for x in candidates if int(x) > 0})
    feasible = [size for size in ordered if block_circulant_param_count(shape, size) <= budget]
    return feasible[0] if feasible else ordered[-1]


def block_circulant_approximation(weight: np.ndarray, *, budget: int, block_sizes: list[int]) -> ApproximationResult:
    w = np.asarray(weight, dtype=np.float32)
    rows, cols = w.shape
    block_size = choose_block_size(w.shape, budget, block_sizes)
    pad_rows = int(np.ceil(rows / block_size) * block_size)
    pad_cols = int(np.ceil(cols / block_size) * block_size)
    padded = np.zeros((pad_rows, pad_cols), dtype=np.float32)
    padded[:rows, :cols] = w
    row_blocks = pad_rows // block_size
    col_blocks = pad_cols // block_size
    blocks = padded.reshape(row_blocks, block_size, col_blocks, block_size).transpose(0, 2, 1, 3)
    block_rows = np.arange(block_size)
    coeff = np.stack(
        [blocks[..., block_rows, (block_rows + shift) % block_size].mean(axis=-1) for shift in range(block_size)],
        axis=-1,
    ).astype(np.float32)
    circulant_index = (np.arange(block_size)[None, :] - np.arange(block_size)[:, None]) % block_size
    approx_blocks = coeff[..., circulant_index]
    approx = approx_blocks.transpose(0, 2, 1, 3).reshape(pad_rows, pad_cols)
    out = approx[:rows, :cols]
    return ApproximationResult(
        method="block_circulant",
        matrix=out.astype(np.float32),
        params=block_circulant_param_count(w.shape, block_size),
        rank=0,
        block_size=block_size,
    )


def monarch_param_count(shape: tuple[int, int], block_size: int, rank_per_block: int, terms: int) -> int:
    rows, cols = shape
    row_blocks = int(np.ceil(rows / block_size))
    effective_terms = min(int(terms), int(np.ceil(cols / block_size)))
    effective_rank = min(int(rank_per_block), int(block_size))
    return int(effective_terms * row_blocks * effective_rank * 2 * block_size)


def monarch_rank_for_budget(shape: tuple[int, int], budget: int, block_size: int, terms: int) -> int:
    rows, cols = shape
    row_blocks = int(np.ceil(rows / block_size))
    effective_terms = min(int(terms), int(np.ceil(cols / block_size)))
    denom = max(effective_terms * row_blocks * 2 * block_size, 1)
    return max(1, min(int(block_size), int(budget // denom)))


def monarch_like_approximation(
    weight: np.ndarray,
    *,
    budget: int,
    block_size: int,
    terms: int = 2,
    svd_device: str = "cpu",
) -> ApproximationResult:
    w = np.asarray(weight, dtype=np.float32)
    rows, cols = w.shape
    rank_per_block = monarch_rank_for_budget(w.shape, budget, block_size, terms)
    row_blocks = int(np.ceil(rows / block_size))
    col_blocks = int(np.ceil(cols / block_size))
    pad_rows = row_blocks * block_size
    pad_cols = col_blocks * block_size
    padded = np.zeros((pad_rows, pad_cols), dtype=np.float32)
    padded[:rows, :cols] = w
    approx = np.zeros_like(padded)
    effective_terms = min(int(terms), col_blocks)
    contribution_counts = np.zeros((row_blocks, col_blocks), dtype=np.int32)
    selected_blocks: list[np.ndarray] = []
    selected_coordinates: list[tuple[int, int]] = []
    for term in range(effective_terms):
        for row_block in range(row_blocks):
            col_block = (row_block * 3 + term) % col_blocks
            row = row_block * block_size
            col = col_block * block_size
            selected_blocks.append(padded[row : row + block_size, col : col + block_size])
            selected_coordinates.append((row_block, col_block))
    u_batch, s_batch, vh_batch = batched_svd_factors(np.stack(selected_blocks), device=svd_device)
    r = min(rank_per_block, s_batch.shape[-1])
    reconstructed = (u_batch[:, :, :r] * s_batch[:, None, :r]) @ vh_batch[:, :r, :]
    for block, (row_block, col_block) in zip(reconstructed, selected_coordinates):
        row = row_block * block_size
        col = col_block * block_size
        approx[row : row + block_size, col : col + block_size] += block
        contribution_counts[row_block, col_block] += 1
    for row_block in range(row_blocks):
        for col_block in range(col_blocks):
            count = int(contribution_counts[row_block, col_block])
            if count > 1:
                row = row_block * block_size
                col = col_block * block_size
                approx[row : row + block_size, col : col + block_size] /= count
    return ApproximationResult(
        method="monarch_like",
        matrix=approx[:rows, :cols].astype(np.float32),
        params=monarch_param_count(w.shape, block_size, rank_per_block, terms),
        rank=rank_per_block,
        block_size=block_size,
        terms=effective_terms,
    )


def approximate_weight(
    weight: np.ndarray,
    *,
    method: str,
    compression_ratio: float,
    block_sizes: list[int] | None = None,
    monarch_block_size: int = 64,
    monarch_terms: int = 2,
    low_rank_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    svd_device: str = "cpu",
) -> ApproximationResult:
    budget = budget_params(weight, compression_ratio)
    if method == "low_rank":
        if low_rank_factors is not None:
            return low_rank_approximation_from_factors(np.asarray(weight).shape, low_rank_factors, params=budget)
        return low_rank_approximation(weight, params=budget, svd_device=svd_device)
    if method == "block_circulant":
        return block_circulant_approximation(weight, budget=budget, block_sizes=block_sizes or [16, 32, 64, 128])
    if method == "monarch_like":
        return monarch_like_approximation(
            weight,
            budget=budget,
            block_size=monarch_block_size,
            terms=monarch_terms,
            svd_device=svd_device,
        )
    raise ValueError(f"unknown approximation method: {method}")


def approximation_error_row(weight: np.ndarray, result: ApproximationResult, *, compression_ratio: float) -> dict[str, object]:
    params = original_params(weight)
    return {
        "method": result.method,
        "compression_ratio_target": float(compression_ratio),
        "params_original": params,
        "params_structured": int(result.params),
        "compression_ratio_actual": float(params / max(result.params, 1)),
        "rank": int(result.rank),
        "block_size": result.block_size or "",
        "terms": result.terms or "",
        "relative_weight_error": relative_fro_error(weight, result.matrix),
    }
