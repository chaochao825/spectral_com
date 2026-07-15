from __future__ import annotations

import re
from pathlib import Path


MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_RESULT_PT_BYTES = 10 * 1024 * 1024
IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "trash",
    "wandb",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".onnx",
    ".pb",
    ".pth",
    ".safetensors",
    ".tflite",
}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:github_pat_|ghp_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
}


def validate_tree(
    root: Path,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_result_pt_bytes: int = MAX_RESULT_PT_BYTES,
) -> tuple[int, list[str]]:
    root = root.resolve()
    errors: list[str] = []
    checked = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in relative.parts):
            continue
        checked += 1
        size = path.stat().st_size
        suffix = path.suffix.lower()
        display = relative.as_posix()
        if size > max_file_bytes:
            errors.append(f"{display}: {size} bytes exceeds the {max_file_bytes}-byte publication limit")
        if suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"{display}: forbidden model/archive suffix {suffix}")
        if suffix == ".pt":
            if not relative.parts or relative.parts[0] != "results":
                errors.append(f"{display}: .pt payloads are allowed only under results/")
            elif size > max_result_pt_bytes:
                errors.append(f"{display}: result .pt payload exceeds {max_result_pt_bytes} bytes")
        if size <= 2 * 1024 * 1024 and suffix not in {".pdf", ".png", ".pt"}:
            payload = path.read_bytes()
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(payload):
                    errors.append(f"{display}: possible {label}")
    return checked, errors
