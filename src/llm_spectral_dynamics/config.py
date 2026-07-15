from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level config must be a mapping: {path}")
    return data


def parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_layers(value: str | None) -> str | list[int] | None:
    if value is None:
        return None
    if value.strip().lower() == "all":
        return "all"
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def project_root_from_file(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "eigenvalues": root / "eigenvalues",
        "plots": root / "plots",
        "eigenspectra": root / "plots" / "eigenspectra",
        "heatmaps": root / "plots" / "heatmaps",
        "comparisons": root / "plots" / "comparisons",
        "kv": root / "kv_cache",
        "interventions": root / "interventions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
