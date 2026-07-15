from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.|(?:^|\.)h\.(\d+)\.|(?:^|\.)blocks\.(\d+)\.")


@dataclass(frozen=True)
class LinearLayerRef:
    name: str
    module_type: str
    layer: int
    module: object
    in_features: int
    out_features: int
    has_bias: bool


def parse_layer_index(name: str) -> int:
    match = LAYER_RE.search(name)
    if match is None:
        return -1
    for group in match.groups():
        if group is not None:
            return int(group)
    return -1


def module_type_from_name(name: str) -> str:
    return name.split(".")[-1]


def resolve_layer_filter(layers: str | Iterable[int] | None, available: Iterable[int]) -> list[int]:
    available_layers = sorted({int(layer) for layer in available if int(layer) >= 0})
    if layers is None or layers == "all":
        return available_layers
    requested = {int(layer) for layer in layers}
    return [layer for layer in available_layers if layer in requested]


def iter_linear_layers(model: object, target_modules: Iterable[str], layers: str | Iterable[int] | None = "all") -> list[LinearLayerRef]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model weight extraction") from exc

    suffixes = tuple(str(item) for item in target_modules)
    refs: list[LinearLayerRef] = []
    all_named = list(model.named_modules())
    available_layers = [parse_layer_index(name) for name, _ in all_named]
    selected_layers = set(resolve_layer_filter(layers, available_layers))
    for name, module in all_named:
        if not isinstance(module, torch.nn.Linear):
            continue
        module_type = module_type_from_name(name)
        if module_type not in suffixes:
            continue
        layer = parse_layer_index(name)
        if selected_layers and layer not in selected_layers:
            continue
        refs.append(
            LinearLayerRef(
                name=name,
                module_type=module_type,
                layer=layer,
                module=module,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                has_bias=module.bias is not None,
            )
        )
    refs.sort(key=lambda item: (item.layer, item.module_type, item.name))
    return refs


def get_submodule_parent(model: object, name: str) -> tuple[object, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def weight_to_numpy(module: object):
    weight = getattr(module, "weight")
    return weight.detach().float().cpu().numpy()
