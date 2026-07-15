from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from llm_spectral_dynamics.config import deep_update, load_yaml


def set_global_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_structured_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    return deep_update(cfg, overrides or {})


def ensure_phase_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "phase1": root / "phase1",
        "phase1_metrics": root / "phase1" / "metrics",
        "phase2": root / "phase2",
        "phase2_metrics": root / "phase2" / "metrics",
        "phase3": root / "phase3",
        "phase3_metrics": root / "phase3" / "metrics",
        "phase4": root / "phase4",
        "phase4_metrics": root / "phase4" / "metrics",
        "phase5": root / "phase5",
        "phase5_metrics": root / "phase5" / "metrics",
        "manifests": root / "manifests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")


def jsonify(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if torch is not None and hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    return value


def safe_name(text: str) -> str:
    return text.replace("/", "__").replace(":", "_").replace(" ", "_")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_float_csv(value: str | None) -> list[float] | None:
    parsed = parse_csv(value)
    if parsed is None:
        return None
    return [float(part) for part in parsed]


def parse_layers(value: str | None) -> str | list[int] | None:
    if value is None or value.strip() == "":
        return None
    if value.strip().lower() == "all":
        return "all"
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def select_layer_positions(layers: list[int], positions: list[str | int]) -> list[int]:
    if not layers:
        return []
    out: list[int] = []
    for pos in positions:
        if isinstance(pos, int):
            candidate = pos
        else:
            name = str(pos).lower()
            if name == "first":
                candidate = layers[0]
            elif name == "middle":
                candidate = layers[len(layers) // 2]
            elif name == "last":
                candidate = layers[-1]
            else:
                candidate = int(pos)
        if candidate in layers and candidate not in out:
            out.append(candidate)
    return out
