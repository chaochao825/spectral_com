from __future__ import annotations

import numpy as np

from .approximations import approximate_weight
from .metrics import relative_fro_error
from .residuals import build_residual


def symmetric_quantize(values: np.ndarray, *, bits: int, axis: int | None = 1) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    qmax = max(1, 2 ** (int(bits) - 1) - 1)
    if axis is None:
        scale = np.max(np.abs(x)) / qmax
        if scale <= 1e-12:
            return np.zeros_like(x)
        return np.clip(np.round(x / scale), -qmax, qmax) * scale
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-12)
    return np.clip(np.round(x / scale), -qmax, qmax) * scale


def quantization_error_rows(weight: np.ndarray, *, bit_widths: list[int], prefix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bits in bit_widths:
        quant = symmetric_quantize(weight, bits=bits, axis=1)
        rows.append({"target": prefix or "weight", "bit_width": int(bits), "relative_quantization_error": relative_fro_error(weight, quant)})
    return rows


def apply_residual_precision(values: np.ndarray, residual_precision: str) -> np.ndarray:
    precision = str(residual_precision).lower()
    if precision in {"float16", "fp16", "half"}:
        return np.asarray(values, dtype=np.float16).astype(np.float32)
    if precision in {"bfloat16", "bf16"}:
        try:
            import torch

            return torch.as_tensor(np.asarray(values, dtype=np.float32)).to(torch.bfloat16).to(torch.float32).cpu().numpy()
        except ImportError:
            return np.asarray(values, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def structured_quantization_rows(
    weight: np.ndarray,
    *,
    compression_ratio: float,
    method: str,
    bit_widths: list[int],
    residual_fraction: float,
    residual_type: str,
    block_sizes: list[int],
    monarch_block_size: int,
    monarch_terms: int,
    residual_precision: str = "float32",
    svd_device: str = "cpu",
) -> list[dict[str, object]]:
    approx = approximate_weight(
        weight,
        method=method,
        compression_ratio=compression_ratio,
        block_sizes=block_sizes,
        monarch_block_size=monarch_block_size,
        monarch_terms=monarch_terms,
        svd_device=svd_device,
    )
    residual = np.asarray(weight, dtype=np.float32) - approx.matrix
    rr = build_residual(
        residual,
        residual_type=residual_type,
        residual_fraction=residual_fraction,
        svd_device=svd_device,
    )
    residual_matrix = apply_residual_precision(rr.matrix, residual_precision)
    rows: list[dict[str, object]] = []
    for bits in bit_widths:
        quant_bulk = symmetric_quantize(approx.matrix, bits=bits, axis=1)
        combined = quant_bulk + residual_matrix
        rows.append(
            {
                "method": method,
                "compression_ratio_target": float(compression_ratio),
                "bit_width": int(bits),
                "residual_fraction": float(residual_fraction),
                "residual_type": rr.residual_type,
                "residual_precision": str(residual_precision),
                "bulk_params": int(approx.params),
                "residual_params": int(rr.params),
                "relative_structured_quantized_error": relative_fro_error(weight, combined),
                "relative_bulk_quantization_error": relative_fro_error(approx.matrix, quant_bulk),
            }
        )
    return rows
