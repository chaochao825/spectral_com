from __future__ import annotations

"""Build the preregistered confirmatory Hessian-repair data protocol.

This script is intentionally data-only: it loads a pinned local Pythia-70M
tokenizer and an already-cached WikiText-2 dataset, then writes a deterministic
split manifest.  It never loads model weights, runs a forward pass, downloads
data, or substitutes fallback text.

The manifest contains source row identifiers, normalized-content SHA256
digests, token lengths, and token-range allocations.  It never contains source
text or token IDs.
"""

import argparse
import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "confirmatory_hessian_protocol.v2"
PROTOCOL_DATE = "2026-07-14"
MODEL_ID = "EleutherAI/pythia-70m"
MODEL_SNAPSHOT_COMMIT = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEEDS = (17, 29, 43, 59, 71, 89, 101, 113)
WINDOW_TOKEN_LENGTH = 256
CALIBRATION_WINDOWS_PER_SEED = 32
VALIDATION_WINDOWS = 32
TEST_WINDOWS = 64
NGRAM_N = 5
NEAR_DUPLICATE_THRESHOLD = 0.8
EPSILON_GRID = (
    0.0,
    1.0 / 32.0,
    1.0 / 16.0,
    3.0 / 32.0,
    1.0 / 8.0,
    3.0 / 16.0,
    1.0 / 4.0,
    3.0 / 8.0,
    1.0 / 2.0,
    5.0 / 8.0,
    3.0 / 4.0,
    7.0 / 8.0,
    1.0,
)
LOCAL_FIT_EPSILONS = EPSILON_GRID[1:5]

# Held-out allocations are selected before calibration.  These constants are
# protocol parameters, not experiment seeds.
TEST_SELECTION_SEED = 2026071401
VALIDATION_SELECTION_SEED = 2026071402
SPLIT_DEDUPLICATION_PRIORITY = ("test", "validation", "train")


@dataclass(frozen=True)
class RowRecord:
    row_id: str
    split: str
    normalized_sha256: str
    normalized_text: str
    token_ids: tuple[int, ...]

    @property
    def token_length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    allocation: dict[str, Any]
    sources: tuple[dict[str, Any], ...]
    token_ids: tuple[int, ...]

    @property
    def token_ngrams(self) -> frozenset[tuple[int, ...]]:
        return token_ngrams(self.token_ids, n=NGRAM_N)


def normalize_text(text: object) -> str:
    """Apply the exact preregistered content normalization."""

    normalized = unicodedata.normalize("NFKC", "" if text is None else str(text)).lower()
    return " ".join(normalized.split())


def normalized_sha256(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def token_ngrams(token_ids: Sequence[int], *, n: int = NGRAM_N) -> frozenset[tuple[int, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return frozenset()
    return frozenset(tuple(int(value) for value in token_ids[start : start + n]) for start in range(len(token_ids) - n + 1))


def ngram_jaccard(left: frozenset[tuple[int, ...]], right: frozenset[tuple[int, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def _tokenize(tokenizer: object, text: str) -> tuple[int, ...]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must expose encode(text, add_special_tokens=False)")
    values = encode(text, add_special_tokens=False)
    return tuple(int(value) for value in values)


def prepare_unique_rows(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    tokenizer: object,
) -> tuple[dict[str, list[RowRecord]], dict[str, Any]]:
    """Exactly deduplicate normalized content while tokenizing native raw text.

    Test and validation rows take priority over train duplicates so accidental
    repeated content can never leak into calibration.  A SHA256 collision with
    unequal normalized content is treated as a hard error.
    """

    missing = set(SPLIT_DEDUPLICATION_PRIORITY).difference(split_rows)
    if missing:
        raise ValueError(f"missing required WikiText splits: {sorted(missing)}")

    prepared = {split: [] for split in SPLIT_DEDUPLICATION_PRIORITY}
    raw_counts = {split: 0 for split in SPLIT_DEDUPLICATION_PRIORITY}
    empty_counts = {split: 0 for split in SPLIT_DEDUPLICATION_PRIORITY}
    duplicate_counts = {split: 0 for split in SPLIT_DEDUPLICATION_PRIORITY}
    zero_token_counts = {split: 0 for split in SPLIT_DEDUPLICATION_PRIORITY}
    seen: dict[str, str] = {}

    for split in SPLIT_DEDUPLICATION_PRIORITY:
        for row_index, row in enumerate(split_rows[split]):
            raw_counts[split] += 1
            raw_text = "" if row.get("text") is None else str(row.get("text"))
            normalized = normalize_text(raw_text)
            if not normalized:
                empty_counts[split] += 1
                continue
            digest = normalized_sha256(normalized)
            if digest in seen:
                if seen[digest] != normalized:
                    raise RuntimeError(f"SHA256 collision while processing {split} row {row_index}")
                duplicate_counts[split] += 1
                continue
            # Normalization defines only the leakage/deduplication identity.
            # Tokenize the original dataset text so confirmatory NLL remains
            # comparable with the standard WikiText raw-text protocol.
            tokens = _tokenize(tokenizer, raw_text)
            if not tokens:
                zero_token_counts[split] += 1
                continue
            seen[digest] = normalized
            prepared[split].append(
                RowRecord(
                    row_id=f"{DATASET_CONFIG}/{split}/{row_index}",
                    split=split,
                    normalized_sha256=digest,
                    normalized_text=normalized,
                    token_ids=tokens,
                )
            )

    audit = {
        "normalization": "Unicode NFKC, lowercase, then collapse all whitespace runs to one ASCII space",
        "normalization_scope": "exact-content SHA256 deduplication only",
        "tokenization_input": "native raw dataset text without normalization",
        "deduplication_priority": list(SPLIT_DEDUPLICATION_PRIORITY),
        "raw_row_count": raw_counts,
        "empty_after_normalization_count": empty_counts,
        "exact_duplicate_removed_count": duplicate_counts,
        "zero_token_removed_count": zero_token_counts,
        "unique_nonempty_row_count": {split: len(rows) for split, rows in prepared.items()},
        "global_unique_normalized_sha256_count": len(seen),
    }
    return prepared, audit


def _deterministic_row_order(rows: Iterable[RowRecord], selection_seed: int) -> list[RowRecord]:
    # random.Random has a specified-enough deterministic implementation for the
    # pinned Python protocol, while sorting first makes input container order
    # irrelevant.  The selected native row IDs remain visible in the manifest.
    ordered = sorted(rows, key=lambda row: row.row_id)
    random.Random(int(selection_seed)).shuffle(ordered)
    return ordered


def _candidate_window(
    rows: Sequence[RowRecord],
    start: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[int, ...], int]:
    """Pack a 256-token candidate while consuming each source row at most once."""

    cursor = 0
    position = start
    sources: list[dict[str, Any]] = []
    tokens: list[int] = []
    while cursor < WINDOW_TOKEN_LENGTH and position < len(rows):
        row = rows[position]
        position += 1
        take = min(row.token_length, WINDOW_TOKEN_LENGTH - cursor)
        if take <= 0:
            continue
        sources.append(
            {
                "row_id": row.row_id,
                "normalized_sha256": row.normalized_sha256,
                "token_length": row.token_length,
                "allocation": {
                    "row_token_start": 0,
                    "row_token_stop": take,
                    "window_token_start": cursor,
                    "window_token_stop": cursor + take,
                },
            }
        )
        tokens.extend(row.token_ids[:take])
        cursor += take
    return tuple(sources), tuple(tokens), position


def _window_conflict(
    candidate_tokens: tuple[int, ...],
    accepted: Sequence[WindowRecord],
    threshold: float,
) -> tuple[bool, float, str | None]:
    candidate_ngrams = token_ngrams(candidate_tokens)
    max_score = 0.0
    max_window_id: str | None = None
    for prior in accepted:
        score = ngram_jaccard(candidate_ngrams, prior.token_ngrams)
        if score > max_score:
            max_score = score
            max_window_id = prior.window_id
        if score >= threshold:
            return True, score, prior.window_id
    return False, max_score, max_window_id


def allocate_windows(
    rows: Sequence[RowRecord],
    *,
    count: int,
    selection_seed: int,
    split: str,
    role: str,
    experiment_seed: int | None,
    accepted: list[WindowRecord],
    consumed_row_ids: set[str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> tuple[list[WindowRecord], int]:
    if count <= 0:
        raise ValueError("window count must be positive")
    available = [row for row in rows if row.row_id not in consumed_row_ids]
    ordered = _deterministic_row_order(available, selection_seed)
    selected: list[WindowRecord] = []
    rejected_near_duplicates = 0
    position = 0
    while len(selected) < count:
        sources, tokens, next_position = _candidate_window(ordered, position)
        for source in sources:
            consumed_row_ids.add(str(source["row_id"]))
        position = next_position
        if len(tokens) != WINDOW_TOKEN_LENGTH:
            raise RuntimeError(
                f"insufficient unique local {split} data: requested {count} {WINDOW_TOKEN_LENGTH}-token "
                f"{role} windows, formed only {len(selected)} after {rejected_near_duplicates} near-duplicate rejections"
            )
        conflict, _, _ = _window_conflict(tokens, accepted, threshold)
        if conflict:
            rejected_near_duplicates += 1
            continue
        index = len(selected)
        seed_label = f"seed-{experiment_seed}" if experiment_seed is not None else "fixed"
        window = WindowRecord(
            window_id=f"{role}/{seed_label}/{index:03d}",
            allocation={
                "role": role,
                "native_split": split,
                "seed": experiment_seed,
                "window_index": index,
            },
            sources=sources,
            token_ids=tokens,
        )
        selected.append(window)
        accepted.append(window)
    return selected, rejected_near_duplicates


def audit_final_near_duplicates(
    windows: Sequence[WindowRecord],
    *,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    pairs_checked = 0
    max_score = 0.0
    max_pair: list[str] | None = None
    violations: list[dict[str, Any]] = []
    ngram_sets = [window.token_ngrams for window in windows]
    for left_index, left in enumerate(windows):
        for right_index in range(left_index + 1, len(windows)):
            right = windows[right_index]
            pairs_checked += 1
            score = ngram_jaccard(ngram_sets[left_index], ngram_sets[right_index])
            if score > max_score:
                max_score = score
                max_pair = [left.window_id, right.window_id]
            if score >= threshold:
                violations.append(
                    {
                        "left_window_id": left.window_id,
                        "right_window_id": right.window_id,
                        "jaccard": score,
                    }
                )
    return {
        "token_ngram_n": NGRAM_N,
        "jaccard_threshold": threshold,
        "comparison_rule": "reject when set Jaccard is greater than or equal to the threshold",
        "pairs_checked": pairs_checked,
        "max_jaccard": max_score,
        "max_pair": max_pair,
        "violation_count": len(violations),
        "violations": violations,
    }


def _public_window(window: WindowRecord) -> dict[str, Any]:
    return {
        "window_id": window.window_id,
        "allocation": window.allocation,
        "token_length": len(window.token_ids),
        "sources": list(window.sources),
    }


def build_protocol(
    split_rows: Mapping[str, Iterable[Mapping[str, object]]],
    tokenizer: object,
    *,
    dataset_fingerprints: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prepared, exact_audit = prepare_unique_rows(split_rows, tokenizer)
    accepted: list[WindowRecord] = []
    consumed_row_ids: set[str] = set()
    rejected: dict[str, int] = {}

    test_windows, rejected["test"] = allocate_windows(
        prepared["test"],
        count=TEST_WINDOWS,
        selection_seed=TEST_SELECTION_SEED,
        split="test",
        role="test",
        experiment_seed=None,
        accepted=accepted,
        consumed_row_ids=consumed_row_ids,
    )
    validation_windows, rejected["validation"] = allocate_windows(
        prepared["validation"],
        count=VALIDATION_WINDOWS,
        selection_seed=VALIDATION_SELECTION_SEED,
        split="validation",
        role="validation",
        experiment_seed=None,
        accepted=accepted,
        consumed_row_ids=consumed_row_ids,
    )

    calibration_by_seed: dict[str, list[WindowRecord]] = {}
    for seed in SEEDS:
        windows, rejected[f"calibration_seed_{seed}"] = allocate_windows(
            prepared["train"],
            count=CALIBRATION_WINDOWS_PER_SEED,
            selection_seed=seed,
            split="train",
            role="calibration",
            experiment_seed=seed,
            accepted=accepted,
            consumed_row_ids=consumed_row_ids,
        )
        calibration_by_seed[str(seed)] = windows

    near_duplicate_audit = audit_final_near_duplicates(accepted)
    if near_duplicate_audit["violation_count"]:
        raise RuntimeError("internal error: final protocol contains a token 5-gram near-duplicate")

    allocated_row_ids = [
        str(source["row_id"])
        for window in accepted
        for source in window.sources
    ]
    if len(allocated_row_ids) != len(set(allocated_row_ids)):
        raise RuntimeError("internal error: a source row was allocated to more than one window")

    tokenizer_name = type(tokenizer).__name__
    vocab_size = getattr(tokenizer, "vocab_size", None)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_date": PROTOCOL_DATE,
        "status": "preregistered_data_split_manifest",
        "model": {
            "model_id": MODEL_ID,
            "snapshot_commit": MODEL_SNAPSHOT_COMMIT,
            "weights_loaded_by_this_script": False,
        },
        "dataset": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "native_split_roles": {
                "train": "per-seed calibration",
                "validation": "fixed protocol-selection/evaluation windows",
                "test": "fixed confirmatory endpoint windows",
            },
            "fingerprints": dict(sorted((dataset_fingerprints or {}).items())),
            "local_cache_only": True,
            "fallback_allowed": False,
        },
        "tokenization": {
            "tokenizer_class": tokenizer_name,
            "vocab_size": int(vocab_size) if vocab_size is not None else None,
            "snapshot_commit": MODEL_SNAPSHOT_COMMIT,
            "add_special_tokens": False,
            "input_text": "native raw dataset text without normalization",
            "deduplication_identity": "SHA256 of NFKC/lowercase/whitespace-collapsed text",
            "window_token_length": WINDOW_TOKEN_LENGTH,
            "source_row_reuse_allowed": False,
            "text_or_token_ids_stored": False,
        },
        "seeds": list(SEEDS),
        "epsilon_grid": list(EPSILON_GRID),
        "local_linear_quadratic_fit_positive_epsilons": list(LOCAL_FIT_EPSILONS),
        "allocation_counts": {
            "calibration_windows_per_seed": CALIBRATION_WINDOWS_PER_SEED,
            "calibration_window_total": CALIBRATION_WINDOWS_PER_SEED * len(SEEDS),
            "validation_windows": VALIDATION_WINDOWS,
            "test_windows": TEST_WINDOWS,
            "total_windows": len(accepted),
        },
        "selection": {
            "heldout_priority": ["test", "validation", "calibration"],
            "test_selection_seed": TEST_SELECTION_SEED,
            "validation_selection_seed": VALIDATION_SELECTION_SEED,
            "calibration_selection_seed": "the corresponding experiment seed",
            "row_consumption": "a source row is consumed by one candidate only; unused tail tokens are discarded",
        },
        "audits": {
            "exact_content": exact_audit,
            "near_duplicate": near_duplicate_audit,
            "near_duplicate_candidate_rejections": rejected,
            "allocated_source_row_count": len(allocated_row_ids),
            "allocated_source_row_id_unique_count": len(set(allocated_row_ids)),
        },
        "windows": {
            "calibration_by_seed": {
                seed: [_public_window(window) for window in windows]
                for seed, windows in calibration_by_seed.items()
            },
            "validation": [_public_window(window) for window in validation_windows],
            "test": [_public_window(window) for window in test_windows],
        },
    }
    return manifest


def render_protocol_json(protocol: Mapping[str, Any]) -> str:
    return json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def render_summary(protocol: Mapping[str, Any]) -> str:
    counts = protocol["allocation_counts"]
    exact = protocol["audits"]["exact_content"]
    near = protocol["audits"]["near_duplicate"]
    epsilon_text = ", ".join(f"{float(value):g}" for value in protocol["epsilon_grid"])
    fit_text = ", ".join(
        f"{float(value):g}" for value in protocol["local_linear_quadratic_fit_positive_epsilons"]
    )
    lines = [
        "# Confirmatory Hessian-repair protocol (2026-07-14)",
        "",
        "This is a preregistered data/split manifest, not a model result. The builder loads only a pinned local tokenizer and local WikiText cache; it has no download or text fallback path.",
        "",
        "## Fixed design",
        "",
        f"- Model/tokenizer: `{protocol['model']['model_id']}` snapshot `{protocol['model']['snapshot_commit']}`.",
        f"- Seeds: `{protocol['seeds']}`.",
        f"- Calibration: `{counts['calibration_windows_per_seed']} x 256` train tokens per seed; `{counts['calibration_window_total']}` pairwise source-row-disjoint windows in total.",
        f"- Validation: `{counts['validation_windows']} x 256` fixed validation windows.",
        f"- Test: `{counts['test_windows']} x 256` fixed test windows.",
        f"- Epsilon grid ({len(protocol['epsilon_grid'])} points): `{epsilon_text}`.",
        f"- Positive local linear+quadratic fit points: `{fit_text}`.",
        "",
        "## Leakage controls",
        "",
        f"- Exact content uses `{exact['normalization']}` before SHA256 deduplication.",
        f"- Tokenization uses `{exact['tokenization_input']}`; normalization never changes the model/PPL input.",
        "- Deduplication priority is test, validation, then train, so duplicate calibration content cannot displace held-out evidence.",
        f"- Token 5-gram set-Jaccard rejects pairs at `>= {near['jaccard_threshold']}`; `{near['pairs_checked']}` final pairs were checked and `{near['violation_count']}` violations remain.",
        f"- Maximum retained-window Jaccard: `{near['max_jaccard']:.9f}`.",
        "- Each native source row is consumed by at most one candidate window, including rejected candidates; seed allocations therefore cannot share source rows.",
        "",
        "## Privacy/reproducibility boundary",
        "",
        "`protocol.json` stores only native row IDs, SHA256 of normalized content, token lengths, and allocation ranges. It stores neither source text nor token IDs. Dataset fingerprints and the pinned tokenizer snapshot identify the required local inputs.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(output_dir: Path, protocol: Mapping[str, Any], *, check: bool = False) -> list[Path]:
    outputs = {
        output_dir / "protocol.json": render_protocol_json(protocol),
        output_dir / "summary.md": render_summary(protocol),
    }
    if check:
        drift = [path for path, expected in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != expected]
        if drift:
            raise SystemExit("confirmatory protocol drift detected: " + ", ".join(str(path) for path in drift))
        return list(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
    return list(outputs)


def _load_local_tokenizer(model_snapshot: Path) -> object:
    resolved = model_snapshot.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"pinned model snapshot directory does not exist: {resolved}")
    if resolved.name != MODEL_SNAPSHOT_COMMIT:
        raise ValueError(
            f"expected Pythia-70M snapshot {MODEL_SNAPSHOT_COMMIT}, got directory {resolved.name!r}"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the experiment host
        raise RuntimeError("transformers is required to read the pinned local tokenizer") from exc
    return AutoTokenizer.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=False,
    )


def _load_local_wikitext(dataset_path: Path | None, dataset_cache_dir: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        from datasets import DownloadConfig, load_dataset, load_from_disk
    except ImportError as exc:  # pragma: no cover - exercised on the experiment host
        raise RuntimeError("datasets is required to read the local WikiText cache") from exc

    if dataset_path is not None:
        resolved = dataset_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"local dataset path does not exist: {resolved}")
        loaded = load_from_disk(str(resolved))
    else:
        cache_dir = None if dataset_cache_dir is None else str(dataset_cache_dir.expanduser().resolve())
        download_config = DownloadConfig(cache_dir=cache_dir, local_files_only=True)
        try:
            loaded = load_dataset(
                DATASET_NAME,
                DATASET_CONFIG,
                cache_dir=cache_dir,
                download_config=download_config,
                download_mode="reuse_dataset_if_exists",
            )
        except Exception as exc:
            raise RuntimeError(
                "the exact WikiText-2 cache is unavailable locally; the protocol builder has no download or fallback path"
            ) from exc

    missing = set(SPLIT_DEDUPLICATION_PRIORITY).difference(loaded)
    if missing:
        raise RuntimeError(f"local WikiText dataset lacks required splits: {sorted(missing)}")
    splits = {split: loaded[split] for split in SPLIT_DEDUPLICATION_PRIORITY}
    fingerprints = {
        split: str(getattr(dataset, "_fingerprint", ""))
        for split, dataset in splits.items()
    }
    if any(not value for value in fingerprints.values()):
        raise RuntimeError("all local WikiText splits must expose a datasets fingerprint")
    return splits, fingerprints


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
        help=f"local {MODEL_ID} snapshot directory ending in {MODEL_SNAPSHOT_COMMIT}",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-path", type=Path, help="optional datasets.load_from_disk WikiText DatasetDict")
    source.add_argument("--dataset-cache-dir", type=Path, help="optional existing Hugging Face datasets cache")
    parser.add_argument("--output-dir", type=Path, default=Path("results/confirmatory_hessian_protocol_20260714"))
    parser.add_argument("--check", action="store_true", help="verify committed outputs instead of writing")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tokenizer = _load_local_tokenizer(args.model_snapshot)
    splits, fingerprints = _load_local_wikitext(args.dataset_path, args.dataset_cache_dir)
    protocol = build_protocol(splits, tokenizer, dataset_fingerprints=fingerprints)
    write_outputs(args.output_dir, protocol, check=args.check)


if __name__ == "__main__":
    main()
