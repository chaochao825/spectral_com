from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


FALLBACK_TEXTS = [
    "Language models compress information through repeated linear transformations and nonlinear mixing.",
    "Structured matrices can reduce parameter count, but the residual often carries important task-specific detail.",
    "A useful compression experiment must compare weight error, activation error, and downstream perplexity.",
    "Calibration data gives a practical view of which approximation errors matter for model behavior.",
]


def load_texts_from_config(data_cfg: dict[str, object], *, limit: int) -> list[str]:
    try:
        from datasets import load_dataset, load_from_disk

        dataset_name = str(data_cfg.get("dataset", "wikitext"))
        subset = data_cfg.get("subset")
        split = str(data_cfg.get("split", "validation"))
        backup_name = str(data_cfg.get("backup_name", "")).strip()
        backup_root = Path(os.environ.get("LLM_SC_DATASET_BACKUP_ROOT", "~/dataset_backup")).expanduser()
        backup_path = backup_root / backup_name if backup_name else None
        if backup_path is not None and backup_path.exists():
            saved = load_from_disk(str(backup_path))
            ds = saved[split] if hasattr(saved, "keys") and split in saved else saved
        elif os.environ.get("LLM_SC_DATA_OFFLINE", "0") == "1":
            raise FileNotFoundError(f"dataset backup not found: {backup_path}")
        elif subset:
            ds = load_dataset(dataset_name, str(subset), split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
        if len(ds) <= 0:
            raise RuntimeError(f"dataset {dataset_name} produced no rows")
        if "text" not in set(getattr(ds, "column_names", [])):
            raise RuntimeError(f"dataset {dataset_name} is missing text column")
        texts = [str(row.get("text", "")).strip() for row in ds if str(row.get("text", "")).strip()]
        if texts:
            return texts[: max(limit, 1)]
        raise RuntimeError(f"dataset {dataset_name} produced no non-empty text rows")
    except Exception as exc:
        allow_fallback = bool(data_cfg.get("allow_fallback", False)) or os.environ.get("LLM_SC_ALLOW_FALLBACK", "0") == "1"
        if not allow_fallback:
            raise RuntimeError(
                f"failed to load dataset {data_cfg.get('dataset', 'wikitext')}/"
                f"{data_cfg.get('subset', '')} split={data_cfg.get('split', 'validation')}"
            ) from exc
    needed = max(limit, 1)
    reps = (needed + len(FALLBACK_TEXTS) - 1) // len(FALLBACK_TEXTS)
    return (FALLBACK_TEXTS * reps)[:needed]


def token_batches(tokenizer: object, texts: Iterable[str], *, sequence_length: int, batch_size: int, limit: int):
    import torch

    joined = "\n\n".join(texts)
    token_ids = tokenizer.encode(joined, add_special_tokens=False)
    if not token_ids:
        eos = getattr(tokenizer, "eos_token_id", None)
        token_ids = [int(eos or 0)]
    needed = max(int(limit), 1) * int(sequence_length)
    reps = (needed + len(token_ids) - 1) // len(token_ids)
    arr = (token_ids * reps)[:needed]
    input_ids = torch.tensor(arr, dtype=torch.long).reshape(max(int(limit), 1), int(sequence_length))
    for start in range(0, input_ids.shape[0], max(int(batch_size), 1)):
        yield input_ids[start : start + max(int(batch_size), 1)]


def load_model_and_tokenizer_from_config(cfg: dict[str, object]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = str(cfg["model"])
    revision = cfg.get("revision") or None
    trust_remote_code = bool(cfg.get("trust_remote_code", False))
    dtype_name = str(cfg.get("torch_dtype", "auto"))
    dtype = None
    if dtype_name not in {"", "none", "None"}:
        if dtype_name == "auto":
            dtype = "auto"
        elif dtype_name in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif dtype_name in {"float16", "fp16"}:
            dtype = torch.float16
        elif dtype_name in {"float32", "fp32"}:
            dtype = torch.float32
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": bool(cfg.get("local_files_only", False)),
        "low_cpu_mem_usage": bool(cfg.get("low_cpu_mem_usage", True)),
    }
    if revision:
        kwargs["revision"] = str(revision)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": bool(cfg.get("local_files_only", False)),
    }
    if revision:
        tokenizer_kwargs["revision"] = str(revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = str(cfg.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device
