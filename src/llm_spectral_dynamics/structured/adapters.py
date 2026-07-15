from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdapterSpec:
    method: str
    params: int
    rank: int
    block_size: int
    requested_budget: int
    within_budget: bool


def _rank_from_budget(in_features: int, out_features: int, budget: int) -> int:
    return max(1, int(budget // max(in_features + out_features, 1)))


def _block_circulant_params(in_features: int, out_features: int, block_size: int) -> int:
    import math

    b = max(1, int(block_size))
    return int(math.ceil(out_features / b) * math.ceil(in_features / b) * b)


def _block_size_candidates(preferred: int, in_features: int, out_features: int) -> list[int]:
    max_dim = max(int(in_features), int(out_features), int(preferred), 1)
    max_block = min(2048, 1 << max_dim.bit_length())
    values = set()
    b = 1
    while b <= max_block:
        if b >= max(1, int(preferred)):
            values.add(b)
        b *= 2
    values.add(max(1, int(preferred)))
    return sorted(values)


def _block_size_from_budget(in_features: int, out_features: int, budget: int, preferred: int) -> tuple[int, int, bool]:
    candidates = [
        (block_size, _block_circulant_params(in_features, out_features, block_size))
        for block_size in _block_size_candidates(preferred, in_features, out_features)
    ]
    feasible = [(block_size, params) for block_size, params in candidates if params <= int(budget)]
    if feasible:
        block_size, params = max(feasible, key=lambda item: item[1])
        return block_size, params, True
    block_size, params = min(candidates, key=lambda item: item[1])
    return block_size, params, False


def _capped_lora_rank(in_features: int, out_features: int, budget: int, rank: int | None, *, mora: bool = False) -> tuple[int, int, bool]:
    max_rank = 1
    max_params = in_features + out_features + (1 if mora else 0)
    for candidate in range(1, max(2, int(budget) + 1)):
        params = candidate * (in_features + out_features) + (candidate * candidate if mora else 0)
        if params > int(budget):
            break
        max_rank = candidate
        max_params = params
    requested = max_rank if rank is None else max(1, int(rank))
    effective = min(requested, max_rank)
    params = effective * (in_features + out_features) + (effective * effective if mora else 0)
    return effective, int(params), params <= int(budget)


def _structured_lora_spec(in_features: int, out_features: int, budget: int, rank: int | None, block_size: int) -> tuple[int, int, int, bool]:
    requested_rank = _rank_from_budget(in_features, out_features, budget) if rank is None else max(1, int(rank))
    candidates: list[tuple[bool, int, int, int]] = []
    for candidate_block in _block_size_candidates(block_size, in_features, out_features):
        structured_params = _block_circulant_params(in_features, out_features, candidate_block)
        remaining = int(budget) - structured_params
        if remaining <= 0:
            continue
        max_rank = remaining // max(in_features + out_features, 1)
        if max_rank <= 0:
            continue
        effective_rank = min(requested_rank, int(max_rank))
        params = structured_params + effective_rank * (in_features + out_features)
        candidates.append((effective_rank == requested_rank, effective_rank, params, candidate_block))
    if candidates:
        exact, effective_rank, params, candidate_block = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        return effective_rank, params, candidate_block, bool(params <= int(budget))
    candidate_block, structured_params, feasible = _block_size_from_budget(in_features, out_features, budget, block_size)
    return 1, structured_params + in_features + out_features, candidate_block, False and feasible


def adapter_spec(method: str, in_features: int, out_features: int, *, budget: int, rank: int | None = None, block_size: int = 16) -> AdapterSpec:
    kind = method.lower()
    if kind in {"lora", "structured_lora"}:
        if kind == "structured_lora":
            rank, params, block_size, within_budget = _structured_lora_spec(in_features, out_features, budget, rank, block_size)
        else:
            rank, params, within_budget = _capped_lora_rank(in_features, out_features, budget, rank, mora=False)
    elif kind == "mora":
        rank, params, within_budget = _capped_lora_rank(in_features, out_features, budget, rank, mora=True)
    elif kind == "fourierft":
        params = int(min(budget, in_features * out_features))
        rank = 0
        within_budget = params <= int(budget)
    elif kind in {"structured", "bca"}:
        block_size, params, within_budget = _block_size_from_budget(in_features, out_features, budget, block_size)
        rank = 0
    else:
        raise ValueError(f"unknown adapter method: {method}")
    return AdapterSpec(kind, int(params), int(rank), int(block_size), int(budget), bool(within_budget and int(params) <= int(budget)))


class AdapterWrappedLinear:
    """Wrap a frozen Linear layer with a trainable update used by Phase 4."""

    def __init__(self, base, *, method: str, budget: int, rank: int | None = None, block_size: int = 16, alpha: float = 1.0):
        import torch

        self.torch = torch
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.spec = adapter_spec(method, int(base.in_features), int(base.out_features), budget=budget, rank=rank, block_size=block_size)
        self.alpha = float(alpha)
        module_cls = {
            "lora": LoRAUpdate,
            "structured_lora": StructuredLoRAUpdate,
            "mora": MoRAUpdate,
            "fourierft": FourierFTUpdate,
            "structured": BlockCirculantUpdate,
            "bca": BlockCirculantUpdate,
        }[self.spec.method]
        if self.spec.method == "structured_lora":
            self.update = module_cls(base.in_features, base.out_features, rank=self.spec.rank, block_size=self.spec.block_size)
        elif self.spec.method in {"structured", "bca"}:
            self.update = module_cls(base.in_features, base.out_features, block_size=self.spec.block_size)
        elif self.spec.method == "fourierft":
            self.update = module_cls(base.in_features, base.out_features, budget=self.spec.params)
        else:
            self.update = module_cls(base.in_features, base.out_features, rank=self.spec.rank)

    def to_module(self):
        torch = self.torch

        class Wrapped(torch.nn.Module):
            def __init__(self, base, update, alpha):
                super().__init__()
                self.base = base
                self.update = update
                self.update_modules = torch.nn.ModuleList(getattr(update, "torch_modules", []))
                self.alpha = alpha
                self.in_features = base.in_features
                self.out_features = base.out_features

            def forward(self, x):
                return self.base(x) + self.alpha * self.update(x)

            def delta_weight(self):
                return self.alpha * self.update.weight_matrix()

        wrapper = Wrapped(self.base, self.update, self.alpha)
        wrapper.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        return wrapper


class LoRAUpdate:
    def __init__(self, in_features: int, out_features: int, *, rank: int):
        import torch

        self.torch = torch
        self.module = torch.nn.Module()
        self.module.a = torch.nn.Parameter(torch.empty(rank, in_features))
        self.module.b = torch.nn.Parameter(torch.zeros(out_features, rank))
        torch.nn.init.kaiming_uniform_(self.module.a, a=5**0.5)
        self.scale = 1.0 / max(rank, 1)
        self.torch_modules = [self.module]

    def __call__(self, x):
        torch = self.torch
        return torch.nn.functional.linear(x, self.weight_matrix())

    def parameters(self):
        return self.module.parameters()

    def weight_matrix(self):
        return self.scale * (self.module.b @ self.module.a)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self


class MoRAUpdate(LoRAUpdate):
    def __init__(self, in_features: int, out_features: int, *, rank: int):
        import torch

        self.torch = torch
        self.module = torch.nn.Module()
        self.module.a = torch.nn.Parameter(torch.empty(rank, in_features))
        self.module.m = torch.nn.Parameter(torch.empty(rank, rank))
        self.module.b = torch.nn.Parameter(torch.zeros(out_features, rank))
        torch.nn.init.kaiming_uniform_(self.module.a, a=5**0.5)
        torch.nn.init.eye_(self.module.m)
        self.scale = 1.0 / max(rank, 1)
        self.torch_modules = [self.module]

    def weight_matrix(self):
        return self.scale * (self.module.b @ self.module.m @ self.module.a)


class BlockCirculantUpdate:
    def __init__(self, in_features: int, out_features: int, *, block_size: int):
        import math
        import torch

        self.torch = torch
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = int(block_size)
        self.row_blocks = int(math.ceil(out_features / block_size))
        self.col_blocks = int(math.ceil(in_features / block_size))
        self.module = torch.nn.Module()
        self.module.coeff = torch.nn.Parameter(torch.zeros(self.row_blocks, self.col_blocks, self.block_size))
        self.torch_modules = [self.module]

    def __call__(self, x):
        return self.torch.nn.functional.linear(x, self.weight_matrix())

    def parameters(self):
        return self.module.parameters()

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def weight_matrix(self):
        torch = self.torch
        rows = torch.arange(self.block_size, device=self.module.coeff.device)
        idx = (rows[None, :] - rows[:, None]) % self.block_size
        blocks = self.module.coeff[:, :, idx]
        full = blocks.permute(0, 2, 1, 3).reshape(self.row_blocks * self.block_size, self.col_blocks * self.block_size)
        return full[: self.out_features, : self.in_features]


class StructuredLoRAUpdate:
    def __init__(self, in_features: int, out_features: int, *, rank: int, block_size: int):
        self.structured = BlockCirculantUpdate(in_features, out_features, block_size=block_size)
        self.lora = LoRAUpdate(in_features, out_features, rank=rank)
        self.torch_modules = [*self.structured.torch_modules, *self.lora.torch_modules]

    def __call__(self, x):
        return self.structured(x) + self.lora(x)

    def parameters(self):
        yield from self.structured.parameters()
        yield from self.lora.parameters()

    def to(self, *args, **kwargs):
        self.structured.to(*args, **kwargs)
        self.lora.to(*args, **kwargs)
        return self

    def weight_matrix(self):
        return self.structured.weight_matrix() + self.lora.weight_matrix()


class FourierFTUpdate:
    def __init__(self, in_features: int, out_features: int, *, budget: int):
        import torch

        self.torch = torch
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.coeff_count = max(1, min(int(budget), self.in_features * self.out_features))
        self.module = torch.nn.Module()
        self.module.coeff = torch.nn.Parameter(torch.zeros(self.coeff_count))
        flat = torch.linspace(0, self.in_features * self.out_features - 1, steps=self.coeff_count).long()
        self.flat_index = flat
        self.torch_modules = [self.module]

    def __call__(self, x):
        return self.torch.nn.functional.linear(x, self.weight_matrix())

    def parameters(self):
        return self.module.parameters()

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        self.flat_index = self.flat_index.to(self.module.coeff.device)
        return self

    def weight_matrix(self):
        torch = self.torch
        flat = torch.zeros(self.out_features * self.in_features, device=self.module.coeff.device, dtype=self.module.coeff.dtype)
        flat[self.flat_index.to(self.module.coeff.device)] = self.module.coeff
        return flat.reshape(self.out_features, self.in_features)
