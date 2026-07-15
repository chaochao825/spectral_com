from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .approximations import approximate_weight
from .residuals import build_residual
from .weights import get_submodule_parent, iter_linear_layers, weight_to_numpy


@dataclass
class ReplacementRecord:
    name: str
    layer: int
    module_type: str
    method: str
    compression_ratio: float
    residual_type: str
    residual_fraction: float
    params_original: int
    params_structured: int
    params_residual: int


class StructuredLinear:
    """Dense execution wrapper for structured approximations used during evaluation."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for StructuredLinear") from exc
        self.torch = torch
        self.weight = torch.as_tensor(weight, dtype=torch.float32)
        self.bias = None if bias is None else torch.as_tensor(bias, dtype=torch.float32)

    def to_module(self, *, dtype=None, device=None):
        torch = self.torch
        out_features, in_features = self.weight.shape
        module = torch.nn.Linear(in_features, out_features, bias=self.bias is not None)
        if dtype is not None or device is not None:
            module = module.to(device=device, dtype=dtype)
        with torch.no_grad():
            target_weight = self.weight.to(device=device or module.weight.device, dtype=dtype or module.weight.dtype)
            module.weight.copy_(target_weight)
            if self.bias is not None and module.bias is not None:
                module.bias.copy_(self.bias.to(device=device or module.bias.device, dtype=dtype or module.bias.dtype))
        return module


def replace_model_linears(
    model: object,
    *,
    target_modules: list[str],
    layers: str | list[int],
    module_types_to_replace: list[str],
    method_by_module: dict[str, str],
    compression_ratio: float,
    residual_type: str,
    residual_fraction: float,
    block_sizes: list[int],
    monarch_block_size: int,
    monarch_terms: int,
    svd_device: str = "cpu",
) -> list[ReplacementRecord]:
    records: list[ReplacementRecord] = []
    refs = iter_linear_layers(model, target_modules, layers)
    for ref in refs:
        if ref.module_type not in module_types_to_replace:
            continue
        weight = weight_to_numpy(ref.module)
        method = method_by_module.get(ref.module_type, method_by_module.get(ref.name, "low_rank"))
        approx = approximate_weight(
            weight,
            method=method,
            compression_ratio=compression_ratio,
            block_sizes=block_sizes,
            monarch_block_size=monarch_block_size,
            monarch_terms=monarch_terms,
            svd_device=svd_device,
        )
        residual = weight - approx.matrix
        rr = build_residual(
            residual,
            residual_type=residual_type,
            residual_fraction=residual_fraction,
            svd_device=svd_device,
        )
        dense_weight = approx.matrix + rr.matrix
        bias = None
        if getattr(ref.module, "bias") is not None:
            bias = ref.module.bias.detach().float().cpu().numpy()
        dtype = ref.module.weight.dtype
        device = ref.module.weight.device
        new_module = StructuredLinear(dense_weight, bias).to_module(dtype=dtype, device=device)
        parent, child_name = get_submodule_parent(model, ref.name)
        setattr(parent, child_name, new_module)
        records.append(
            ReplacementRecord(
                name=ref.name,
                layer=ref.layer,
                module_type=ref.module_type,
                method=method,
                compression_ratio=float(compression_ratio),
                residual_type=rr.residual_type,
                residual_fraction=float(residual_fraction),
                params_original=int(weight.size),
                params_structured=int(approx.params),
                params_residual=int(rr.params),
            )
        )
    return records
