from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from llm_spectral_dynamics.structured.approximations import (
    block_circulant_approximation,
    budget_params,
    monarch_like_approximation,
    monarch_rank_for_budget,
)
from llm_spectral_dynamics.structured.metrics import relative_fro_error


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _timed(fn, repeats: int):
    values = []
    result = None
    for _ in range(max(1, repeats)):
        _synchronize()
        start = time.perf_counter()
        result = fn()
        _synchronize()
        values.append(time.perf_counter() - start)
    return result, float(np.median(values))


def _legacy_monarch(weight: np.ndarray, *, compression_ratio: float, block_size: int, terms: int):
    rows, cols = weight.shape
    budget = budget_params(weight, compression_ratio)
    rank_per_block = monarch_rank_for_budget(weight.shape, budget, block_size, terms)
    row_blocks = int(np.ceil(rows / block_size))
    col_blocks = int(np.ceil(cols / block_size))
    padded = np.zeros((row_blocks * block_size, col_blocks * block_size), dtype=np.float32)
    padded[:rows, :cols] = weight
    approx = np.zeros_like(padded)
    counts = np.zeros((row_blocks, col_blocks), dtype=np.int32)
    for term in range(min(terms, col_blocks)):
        for row_block in range(row_blocks):
            col_block = (row_block * 3 + term) % col_blocks
            row = row_block * block_size
            col = col_block * block_size
            block = padded[row : row + block_size, col : col + block_size]
            u, s, vh = np.linalg.svd(block, full_matrices=False)
            r = min(rank_per_block, s.size)
            approx[row : row + block_size, col : col + block_size] += (u[:, :r] * s[:r]) @ vh[:r, :]
            counts[row_block, col_block] += 1
    for row_block, col_block in np.argwhere(counts > 1):
        row = row_block * block_size
        col = col_block * block_size
        approx[row : row + block_size, col : col + block_size] /= counts[row_block, col_block]
    return approx[:rows, :cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark structured approximation CUDA acceleration.")
    parser.add_argument("--rows", type=int, default=1536)
    parser.add_argument("--cols", type=int, default=8960)
    parser.add_argument("--compression-ratio", type=float, default=4.0)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--terms", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    weight = rng.normal(size=(args.rows, args.cols)).astype(np.float32)
    rows: list[dict[str, object]] = []

    block_result, seconds = _timed(
        lambda: block_circulant_approximation(
            weight,
            budget=budget_params(weight, args.compression_ratio),
            block_sizes=[args.block_size],
        ),
        args.repeats,
    )
    rows.append({"operation": "block_circulant", "backend": "numpy_vectorized", "seconds": seconds, "relative_diff": 0.0})

    cpu_result, seconds = _timed(
        lambda: monarch_like_approximation(
            weight,
            budget=budget_params(weight, args.compression_ratio),
            block_size=args.block_size,
            terms=args.terms,
            svd_device="cpu",
        ),
        args.repeats,
    )
    rows.append({"operation": "monarch_like", "backend": "numpy_batched", "seconds": seconds, "relative_diff": 0.0})

    try:
        import torch

        if torch.cuda.is_available():
            cuda_result, seconds = _timed(
                lambda: monarch_like_approximation(
                    weight,
                    budget=budget_params(weight, args.compression_ratio),
                    block_size=args.block_size,
                    terms=args.terms,
                    svd_device="cuda",
                ),
                args.repeats,
            )
            rows.append(
                {
                    "operation": "monarch_like",
                    "backend": "torch_cuda_batched",
                    "seconds": seconds,
                    "relative_diff": relative_fro_error(cpu_result.matrix, cuda_result.matrix),
                }
            )
    except ImportError:
        pass

    if args.include_legacy:
        legacy, seconds = _timed(
            lambda: _legacy_monarch(
                weight,
                compression_ratio=args.compression_ratio,
                block_size=args.block_size,
                terms=args.terms,
            ),
            args.repeats,
        )
        rows.append(
            {
                "operation": "monarch_like",
                "backend": "numpy_legacy_per_block",
                "seconds": seconds,
                "relative_diff": relative_fro_error(cpu_result.matrix, legacy),
            }
        )

    payload = {
        "shape": [args.rows, args.cols],
        "compression_ratio": args.compression_ratio,
        "block_size": args.block_size,
        "terms": args.terms,
        "rows": rows,
        "block_params": block_result.params,
        "monarch_params": cpu_result.params,
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["operation", "backend", "seconds", "relative_diff"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
