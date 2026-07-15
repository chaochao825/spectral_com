from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .data import load_texts_from_config, token_batches


def evaluate_perplexity(
    model: object,
    tokenizer: object,
    *,
    data_cfg: dict[str, object],
    device: str,
    eval_limit: int,
) -> dict[str, float | int]:
    import torch

    texts = load_texts_from_config(data_cfg, limit=max(int(eval_limit), 1) * 2)
    seq_len = int(data_cfg.get("sequence_length", 128))
    batch_size = int(data_cfg.get("batch_size", 1))
    nll_total = 0.0
    token_total = 0
    with torch.no_grad():
        for batch in token_batches(tokenizer, texts, sequence_length=seq_len, batch_size=batch_size, limit=eval_limit):
            batch = batch.to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            nll_total += float(loss.detach().cpu())
            token_total += int(labels.numel())
    mean_nll = nll_total / max(token_total, 1)
    return {"nll": mean_nll, "perplexity": float(math.exp(min(mean_nll, 50.0))), "tokens": token_total}


def _conditional_nll(model: object, tokenizer: object, prompt: str, continuation: str, *, device: str) -> float:
    import torch

    full_text = prompt + continuation
    continuation_positions: list[int] = []
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long, device=device)
        offsets = torch.as_tensor(encoded["offset_mapping"])[0].tolist()
        continuation_positions = [
            index
            for index, (_, end) in enumerate(offsets)
            if index > 0 and int(end) > len(prompt)
        ]
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError) as exc:
        if os.environ.get("LLM_SC_ZERO_SHOT_STRICT", "0") == "1":
            raise RuntimeError("strict zero-shot scoring requires combined tokenization with offset mappings") from exc
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        continuation_positions = list(range(max(len(prompt_ids), 1), len(full_ids)))
    if not continuation_positions:
        if os.environ.get("LLM_SC_ZERO_SHOT_STRICT", "0") == "1":
            raise RuntimeError("strict zero-shot scoring found no continuation token positions")
        return float("inf")

    with torch.no_grad():
        full_logits = model(input_ids=input_ids).logits.float()
    predictor_positions = torch.tensor([index - 1 for index in continuation_positions], dtype=torch.long, device=device)
    label_positions = torch.tensor(continuation_positions, dtype=torch.long, device=device)
    logits = full_logits.index_select(1, predictor_positions)
    labels = input_ids.index_select(1, label_positions)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
    return float(loss.detach().cpu()) / len(continuation_positions)


ZERO_SHOT_BACKUP_NAMES = {
    "piqa": "piqa",
    "arc_easy": "ai2_arc_easy",
    "hellaswag": "hellaswag",
}

ZERO_SHOT_REQUIRED_COLUMNS = {
    "piqa": {"goal", "sol1", "sol2", "label"},
    "arc_easy": {"question", "choices", "answerKey"},
    "hellaswag": {"ctx", "endings", "label"},
}


def _load_zero_shot_dataset(task: str):
    from datasets import load_dataset, load_from_disk

    backup_root = Path(os.environ.get("LLM_SC_DATASET_BACKUP_ROOT", "~/dataset_backup")).expanduser()
    backup_name = ZERO_SHOT_BACKUP_NAMES.get(task)
    backup_path = backup_root / backup_name if backup_name else None
    if backup_path is not None and backup_path.exists():
        saved = load_from_disk(str(backup_path))
        dataset = saved["validation"] if hasattr(saved, "keys") and "validation" in saved else saved
        _validate_zero_shot_dataset(task, dataset)
        return dataset
    if os.environ.get("LLM_SC_ZERO_SHOT_OFFLINE", "0") == "1":
        raise FileNotFoundError(f"zero-shot backup not found for {task}: {backup_path}")
    if task == "piqa":
        dataset = load_dataset("piqa", split="validation")
        _validate_zero_shot_dataset(task, dataset)
        return dataset
    if task == "arc_easy":
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation")
        _validate_zero_shot_dataset(task, dataset)
        return dataset
    if task == "hellaswag":
        dataset = load_dataset("hellaswag", split="validation")
        _validate_zero_shot_dataset(task, dataset)
        return dataset
    raise ValueError(f"unsupported zero-shot task: {task}")


def _validate_zero_shot_dataset(task: str, dataset: object) -> None:
    if len(dataset) <= 0:
        raise ValueError(f"zero-shot dataset {task} is empty")
    columns = set(getattr(dataset, "column_names", []))
    missing = ZERO_SHOT_REQUIRED_COLUMNS[task] - columns
    if missing:
        raise ValueError(f"zero-shot dataset {task} is missing columns: {sorted(missing)}")


def _load_zero_shot_examples(task: str, limit: int):
    ds = _load_zero_shot_dataset(task)

    if task == "piqa":
        for row in list(ds)[:limit]:
            yield str(row["goal"]), [_choice_continuation(row["sol1"]), _choice_continuation(row["sol2"])], int(row["label"])
    elif task == "arc_easy":
        for row in list(ds)[:limit]:
            labels = list(row["choices"]["label"])
            choices = [_choice_continuation(x) for x in row["choices"]["text"]]
            answer = str(row["answerKey"])
            label = labels.index(answer) if answer in labels else 0
            yield str(row["question"]) + "\nAnswer:", choices, int(label)
    elif task == "hellaswag":
        for row in list(ds)[:limit]:
            yield str(row["ctx"]), [_choice_continuation(x) for x in row["endings"]], int(row["label"])
    else:
        raise ValueError(f"unsupported zero-shot task: {task}")


def _choice_continuation(value: object) -> str:
    text = str(value)
    return text if not text or text[0].isspace() else " " + text


def evaluate_zero_shot(
    model: object,
    tokenizer: object,
    *,
    tasks: list[str],
    limit: int,
    device: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        try:
            correct = 0
            total = 0
            for prompt, choices, label in _load_zero_shot_examples(task, limit):
                scores = [_conditional_nll(model, tokenizer, prompt, choice, device=device) for choice in choices]
                if os.environ.get("LLM_SC_ZERO_SHOT_STRICT", "0") == "1" and not all(np.isfinite(score) for score in scores):
                    raise ValueError(f"zero-shot task {task} produced non-finite choice scores")
                pred = int(np.argmin(scores))
                correct += int(pred == int(label))
                total += 1
            rows.append({"task": task, "accuracy": correct / max(total, 1), "examples": total, "status": "ok"})
        except Exception as exc:
            if os.environ.get("LLM_SC_ZERO_SHOT_STRICT", "0") == "1":
                raise RuntimeError(f"zero-shot task {task} failed") from exc
            rows.append({"task": task, "accuracy": float("nan"), "examples": 0, "status": f"unavailable: {type(exc).__name__}: {exc}"})
    return rows
