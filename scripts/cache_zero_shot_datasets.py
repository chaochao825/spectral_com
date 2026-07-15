from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


SPECS = {
    "wikitext": ("wikitext", "wikitext-2-raw-v1"),
    "piqa": ("piqa", None),
    "arc_easy": ("ai2_arc", "ARC-Easy"),
    "hellaswag": ("hellaswag", None),
}

BACKUP_NAMES = {
    "wikitext": "wikitext_2_raw",
    "piqa": "piqa",
    "arc_easy": "ai2_arc_easy",
    "hellaswag": "hellaswag",
}

REQUIRED_COLUMNS = {
    "wikitext": {"text"},
    "piqa": {"goal", "sol1", "sol2", "label"},
    "arc_easy": {"question", "choices", "answerKey"},
    "hellaswag": {"ctx", "endings", "label"},
}


def load_validation(task: str, dataset_name: str, subset: str | None):
    kwargs = {} if subset is None else {"name": subset}
    try:
        return load_dataset(dataset_name, split="validation", **kwargs)
    except Exception:
        if task != "wikitext":
            raise
        candidates = sorted(
            Path("~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1").expanduser().glob(
                "**/wikitext-validation.arrow"
            )
        )
        if not candidates:
            raise
        return Dataset.from_file(str(candidates[-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache structured-compression datasets as local save_to_disk backups.")
    parser.add_argument("--root", default="~/dataset_backup")
    parser.add_argument("--tasks", default="wikitext,piqa,arc_easy,hellaswag")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for task in [part.strip() for part in args.tasks.split(",") if part.strip()]:
        dataset_name, subset = SPECS[task]
        target = root / BACKUP_NAMES[task]
        if target.exists():
            print(f"skip download {task}: {target} exists")
        else:
            validation = load_validation(task, dataset_name, subset)
            DatasetDict({"validation": validation}).save_to_disk(str(target))
            print(f"saved {task}: {target}")
        saved = load_from_disk(str(target))
        validation = saved["validation"] if hasattr(saved, "keys") and "validation" in saved else saved
        if len(validation) <= 0:
            raise RuntimeError(f"{task} validation backup is empty: {target}")
        missing = REQUIRED_COLUMNS[task] - set(validation.column_names)
        if missing:
            raise RuntimeError(f"{task} backup is missing columns {sorted(missing)}: {target}")
        print(f"verified {task}: validation_rows={len(validation)}")


if __name__ == "__main__":
    main()
