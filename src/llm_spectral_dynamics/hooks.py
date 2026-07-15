from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _extract_tensor(output: Any):
    if isinstance(output, tuple):
        return _extract_tensor(output[0])
    if isinstance(output, list):
        return _extract_tensor(output[0])
    return output


def has_submodule(model: Any, path: str) -> bool:
    try:
        model.get_submodule(path)
        return True
    except Exception:
        return False


def _config_n_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    for name in ("n_layer", "num_hidden_layers", "num_layers"):
        value = getattr(cfg, name, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer number of transformer layers from model.config")


def resolve_layers(model: Any, layers: str | list[int]) -> list[int]:
    n_layers = _config_n_layers(model)
    if layers == "all":
        return list(range(n_layers))
    out: list[int] = []
    for layer in layers:
        idx = int(layer)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer {layer} resolves to {idx}, outside [0, {n_layers})")
        out.append(idx)
    return sorted(set(out))


def module_path_for_site(model: Any, layer: int, site: str) -> str | None:
    candidates: list[str] = []
    if site == "resid_post":
        candidates = [
            f"transformer.h.{layer}",
            f"gpt_neox.layers.{layer}",
            f"model.layers.{layer}",
        ]
    elif site == "attn_out":
        candidates = [
            f"transformer.h.{layer}.attn",
            f"gpt_neox.layers.{layer}.attention",
            f"model.layers.{layer}.self_attn",
        ]
    elif site == "mlp_out":
        candidates = [
            f"transformer.h.{layer}.mlp",
            f"gpt_neox.layers.{layer}.mlp",
            f"model.layers.{layer}.mlp",
        ]
    elif site in {"q", "k", "v", "qkv"}:
        candidates = [
            f"transformer.h.{layer}.attn.c_attn",
            f"gpt_neox.layers.{layer}.attention.query_key_value",
            f"model.layers.{layer}.self_attn.q_proj",
        ]
    for path in candidates:
        if has_submodule(model, path):
            return path
    return None


@dataclass(frozen=True)
class HookKey:
    layer: int
    site: str


class ActivationHookManager:
    """Register lightweight forward hooks for common HF causal-LM modules."""

    def __init__(self, model: Any, layers: list[int], sites: list[str]) -> None:
        self.model = model
        self.layers = layers
        self.sites = sites
        self.handles: list[Any] = []
        self.cache: dict[HookKey, Any] = {}

    def install(self) -> None:
        hook_sites = {"resid_post", "attn_out", "mlp_out", "q", "k", "v", "qkv"}
        for layer in self.layers:
            for site in self.sites:
                if site not in hook_sites:
                    continue
                path = module_path_for_site(self.model, layer, site)
                if path is None:
                    continue
                module = self.model.get_submodule(path)
                key = HookKey(layer, "qkv" if site in {"q", "k", "v"} else site)
                handle = module.register_forward_hook(self._make_hook(key))
                self.handles.append(handle)

    def _make_hook(self, key: HookKey):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _extract_tensor(output)
            self.cache[key] = tensor.detach()

        return hook

    def clear(self) -> None:
        self.cache.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cache.clear()


def split_qkv(tensor: Any, site: str):
    if site == "qkv":
        return tensor
    parts = tensor.chunk(3, dim=-1)
    index = {"q": 0, "k": 1, "v": 2}[site]
    return parts[index]
