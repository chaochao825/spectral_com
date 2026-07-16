from __future__ import annotations

"""Fail-closed consumer for the confirmatory Hessian protocol manifest.

The v2 manifest deliberately stores neither source text nor token IDs.  This
module binds that manifest to an externally supplied SHA256, a pinned model
and tokenizer, and the exact native WikiText rows.  It then reconstructs every
window by tokenizing each source row independently.  It never strips text,
joins rows as text, inserts a separator, downloads a substitute dataset, or
uses fallback text.
"""

import hashlib
import hmac
import json
import numbers
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence


SCHEMA_VERSION = "confirmatory_hessian_protocol.v2"
PROTOCOL_STATUS = "preregistered_data_split_manifest"
PROTOCOL_DATE = "2026-07-14"
MODEL_ID = "EleutherAI/pythia-70m"
MODEL_SNAPSHOT_COMMIT = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_CLASS = "GPTNeoXTokenizerFast"
TOKENIZER_VOCAB_SIZE = 50_254
WINDOW_TOKEN_LENGTH = 256
SEEDS = (17, 29, 43, 59, 71, 89, 101, 113)
CALIBRATION_WINDOWS_PER_SEED = 32
VALIDATION_WINDOWS = 32
TEST_WINDOWS = 64
DATASET_FINGERPRINTS = (
    ("test", "9cc25baaccbbdb50"),
    ("train", "3204e0c774ed92de"),
    ("validation", "180064024b41ff90"),
)
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
_SPLITS = ("train", "validation", "test")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


class ProtocolValidationError(ValueError):
    """Raised when any protocol identity or reconstruction check fails."""


@dataclass(frozen=True)
class LoadedProtocolDataset:
    """Exact dataset returned by an injectable loader.

    ``fallback_used`` is explicit so a loader cannot silently return substitute
    text while still satisfying the split interface.
    """

    name: str
    config: str
    splits: Mapping[str, Sequence[Mapping[str, object]]]
    fingerprints: Mapping[str, str]
    fallback_used: bool = False


DatasetLoader = Callable[[str, str], LoadedProtocolDataset]


@dataclass(frozen=True)
class ProtocolWindow:
    window_id: str
    role: Literal["calibration", "validation", "test"]
    seed: int | None
    token_ids: tuple[int, ...]
    token_digest: str


@dataclass(frozen=True)
class ProtocolProvenance:
    # These first fields intentionally match run_config["data"]["protocol"].
    manifest_path: str
    manifest_sha256: str
    schema_version: str
    selected_seed: int
    evaluation_role: Literal["validation", "test"]
    window_token_length: int
    calibration_window_ids: tuple[str, ...]
    calibration_window_count: int
    calibration_token_sha256: str
    evaluation_window_ids: tuple[str, ...]
    evaluation_window_count: int
    evaluation_token_sha256: str
    consumed: bool

    # Complete identities needed to audit the reconstruction later.
    status: str
    protocol_date: str
    model_id: str
    model_snapshot_commit: str
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_snapshot_commit: str
    dataset_name: str
    dataset_config: str
    dataset_fingerprints: tuple[tuple[str, str], ...]
    dataset_local_cache_only: bool
    dataset_fallback_allowed: bool
    tokenizer_add_special_tokens: bool
    tokenization_input: str
    source_row_reuse_allowed: bool
    manifest_seeds: tuple[int, ...]
    epsilon_grid: tuple[float, ...]
    local_fit_positive_epsilons: tuple[float, ...]
    validation_window_ids: tuple[str, ...]
    validation_token_sha256: str
    test_window_ids: tuple[str, ...]
    test_token_sha256: str
    all_calibration_window_ids: tuple[str, ...]
    calibration_token_sha256_by_seed: tuple[tuple[int, str], ...]
    all_window_count: int
    all_window_token_sha256: str
    allocated_source_row_count: int
    allocated_source_row_ids_sha256: str


@dataclass(frozen=True)
class ConfirmatoryProtocolSelection:
    selected_calibration_windows: tuple[ProtocolWindow, ...]
    evaluation_windows: tuple[ProtocolWindow, ...]
    validation_windows: tuple[ProtocolWindow, ...]
    test_windows: tuple[ProtocolWindow, ...]
    provenance: ProtocolProvenance

    @property
    def calibration_windows(self) -> tuple[ProtocolWindow, ...]:
        """Compatibility spelling for the selected calibration windows."""

        return self.selected_calibration_windows


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash one ordered token array using canonical compact JSON."""

    return hashlib.sha256(_canonical_json_bytes([int(value) for value in token_ids])).hexdigest()


def digest_protocol_windows(windows: Sequence[ProtocolWindow]) -> str:
    """Hash ordered window boundaries and token IDs for run provenance."""

    payload = [
        {"window_id": window.window_id, "token_ids": list(window.token_ids)}
        for window in windows
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return " ".join(normalized.split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ProtocolValidationError(f"manifest contains forbidden JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"manifest contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ProtocolValidationError(f"{context} keys must be strings")
    return value


def _as_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolValidationError(f"{context} must be an array")
    return value


def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"{context} must be a string")
    return value


def _as_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{context} must be an integer")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolValidationError(
            f"{context} keys do not match v2 schema; missing={missing}, extra={extra}"
        )


def _expect_equal(actual: object, expected: object, context: str) -> None:
    # Equality alone would accept True == 1.  Identity-sensitive scalar types
    # are checked here before comparing their values.
    if isinstance(expected, bool):
        matches = isinstance(actual, bool) and actual is expected
    elif expected is None:
        matches = actual is None
    elif isinstance(expected, int):
        matches = not isinstance(actual, bool) and isinstance(actual, int) and actual == expected
    else:
        matches = type(actual) is type(expected) and actual == expected
    if not matches:
        raise ProtocolValidationError(f"{context} mismatch: expected {expected!r}, got {actual!r}")


def _normalize_fingerprints(
    fingerprints: Mapping[str, str],
    context: str,
) -> tuple[tuple[str, str], ...]:
    if set(fingerprints) != set(_SPLITS):
        raise ProtocolValidationError(
            f"{context} must contain exactly train, validation, and test fingerprints"
        )
    normalized: list[tuple[str, str]] = []
    for split in sorted(_SPLITS):
        value = fingerprints[split]
        if not isinstance(value, str) or not value:
            raise ProtocolValidationError(f"{context}.{split} must be a nonempty string")
        normalized.append((split, value))
    return tuple(normalized)


def _load_local_dataset(
    dataset_name: str,
    dataset_config: str,
    *,
    cache_dir: Path | None,
) -> LoadedProtocolDataset:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - experiment-host dependency
        raise ProtocolValidationError("datasets is required to consume the protocol") from exc

    resolved_cache = None if cache_dir is None else str(cache_dir.expanduser().resolve())
    try:
        loaded = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=resolved_cache,
            download_config=DownloadConfig(
                cache_dir=resolved_cache,
                local_files_only=True,
            ),
            download_mode="reuse_dataset_if_exists",
        )
    except Exception as exc:  # pragma: no cover - depends on host cache
        raise ProtocolValidationError(
            "exact local WikiText cache is unavailable; download and fallback are forbidden"
        ) from exc

    if set(loaded) != set(_SPLITS):
        raise ProtocolValidationError("loaded dataset must contain exactly train, validation, and test")
    splits = {split: loaded[split] for split in _SPLITS}
    fingerprints = {
        split: str(getattr(splits[split], "_fingerprint", ""))
        for split in _SPLITS
    }
    return LoadedProtocolDataset(
        name=dataset_name,
        config=dataset_config,
        splits=splits,
        fingerprints=fingerprints,
        fallback_used=False,
    )


def _read_bound_manifest(path: Path, expected_sha256: str) -> tuple[Mapping[str, Any], str, Path]:
    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ProtocolValidationError("expected manifest SHA256 must be exactly 64 hexadecimal characters")
    try:
        resolved = path.expanduser().resolve(strict=True)
        raw = resolved.read_bytes()
    except OSError as exc:
        raise ProtocolValidationError(f"cannot read protocol manifest: {path}") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
        raise ProtocolValidationError(
            f"manifest SHA256 mismatch: expected {expected_sha256.lower()}, got {actual_sha256}"
        )
    try:
        manifest = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError("manifest is not strict UTF-8 JSON") from exc
    return _as_mapping(manifest, "manifest"), actual_sha256, resolved


def consume_confirmatory_protocol(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    experiment_seed: int,
    tokenizer: object,
    evaluation_role: Literal["validation", "test"] = "test",
    dataset_loader: DatasetLoader | None = None,
    dataset_cache_dir: str | Path | None = None,
    expected_model_id: str = MODEL_ID,
    expected_model_snapshot_commit: str = MODEL_SNAPSHOT_COMMIT,
    expected_tokenizer_class: str = TOKENIZER_CLASS,
    expected_tokenizer_vocab_size: int = TOKENIZER_VOCAB_SIZE,
    expected_tokenizer_snapshot_commit: str = MODEL_SNAPSHOT_COMMIT,
    expected_dataset_name: str = DATASET_NAME,
    expected_dataset_config: str = DATASET_CONFIG,
    expected_dataset_fingerprints: Mapping[str, str] | None = None,
    expected_seeds: Sequence[int] = SEEDS,
) -> ConfirmatoryProtocolSelection:
    """Validate and reconstruct all v2 windows, then select one seed/role.

    The expected manifest hash is mandatory and external to the JSON.  Tests
    and CPU-only callers can inject ``dataset_loader``; production loading is
    local-cache-only and has no retry or fallback branch.
    """

    manifest, manifest_sha256, resolved_manifest = _read_bound_manifest(
        Path(manifest_path), expected_sha256
    )
    _exact_keys(
        manifest,
        {
            "allocation_counts",
            "audits",
            "dataset",
            "epsilon_grid",
            "local_linear_quadratic_fit_positive_epsilons",
            "model",
            "protocol_date",
            "schema_version",
            "seeds",
            "selection",
            "status",
            "tokenization",
            "windows",
        },
        "manifest",
    )
    _expect_equal(manifest["schema_version"], SCHEMA_VERSION, "schema_version")
    _expect_equal(manifest["status"], PROTOCOL_STATUS, "status")
    _expect_equal(manifest["protocol_date"], PROTOCOL_DATE, "protocol_date")
    _expect_equal(manifest["epsilon_grid"], list(EPSILON_GRID), "epsilon_grid")
    _expect_equal(
        manifest["local_linear_quadratic_fit_positive_epsilons"],
        list(LOCAL_FIT_EPSILONS),
        "local fit epsilon grid",
    )

    model = _as_mapping(manifest["model"], "model")
    _exact_keys(model, {"model_id", "snapshot_commit", "weights_loaded_by_this_script"}, "model")
    _expect_equal(model["model_id"], expected_model_id, "model.model_id")
    _expect_equal(
        model["snapshot_commit"],
        expected_model_snapshot_commit,
        "model.snapshot_commit",
    )
    _expect_equal(model["weights_loaded_by_this_script"], False, "model weights-loaded flag")

    dataset = _as_mapping(manifest["dataset"], "dataset")
    _exact_keys(
        dataset,
        {
            "config",
            "fallback_allowed",
            "fingerprints",
            "local_cache_only",
            "name",
            "native_split_roles",
        },
        "dataset",
    )
    _expect_equal(dataset["name"], expected_dataset_name, "dataset.name")
    _expect_equal(dataset["config"], expected_dataset_config, "dataset.config")
    _expect_equal(dataset["local_cache_only"], True, "dataset.local_cache_only")
    _expect_equal(dataset["fallback_allowed"], False, "dataset.fallback_allowed")
    _expect_equal(
        dataset["native_split_roles"],
        {
            "train": "per-seed calibration",
            "validation": "fixed protocol-selection/evaluation windows",
            "test": "fixed confirmatory endpoint windows",
        },
        "dataset.native_split_roles",
    )
    manifest_fingerprints = _normalize_fingerprints(
        _as_mapping(dataset["fingerprints"], "dataset.fingerprints"),
        "dataset.fingerprints",
    )
    external_fingerprints = _normalize_fingerprints(
        dict(DATASET_FINGERPRINTS)
        if expected_dataset_fingerprints is None
        else expected_dataset_fingerprints,
        "expected dataset fingerprints",
    )
    if manifest_fingerprints != external_fingerprints:
        raise ProtocolValidationError(
            "dataset fingerprint mismatch between manifest and external expectation"
        )

    tokenization = _as_mapping(manifest["tokenization"], "tokenization")
    _exact_keys(
        tokenization,
        {
            "add_special_tokens",
            "deduplication_identity",
            "input_text",
            "snapshot_commit",
            "source_row_reuse_allowed",
            "text_or_token_ids_stored",
            "tokenizer_class",
            "vocab_size",
            "window_token_length",
        },
        "tokenization",
    )
    _expect_equal(tokenization["tokenizer_class"], expected_tokenizer_class, "tokenizer class")
    _expect_equal(tokenization["vocab_size"], expected_tokenizer_vocab_size, "tokenizer vocab size")
    _expect_equal(
        tokenization["snapshot_commit"],
        expected_tokenizer_snapshot_commit,
        "tokenizer snapshot",
    )
    _expect_equal(tokenization["add_special_tokens"], False, "tokenizer special-token flag")
    _expect_equal(
        tokenization["input_text"],
        "native raw dataset text without normalization",
        "tokenizer input-text rule",
    )
    _expect_equal(
        tokenization["deduplication_identity"],
        "SHA256 of NFKC/lowercase/whitespace-collapsed text",
        "deduplication identity",
    )
    _expect_equal(tokenization["source_row_reuse_allowed"], False, "source-row reuse flag")
    _expect_equal(tokenization["text_or_token_ids_stored"], False, "stored-text flag")
    _expect_equal(tokenization["window_token_length"], WINDOW_TOKEN_LENGTH, "window token length")

    observed_tokenizer_class = type(tokenizer).__name__
    if observed_tokenizer_class != expected_tokenizer_class:
        raise ProtocolValidationError(
            f"runtime tokenizer class mismatch: expected {expected_tokenizer_class!r}, "
            f"got {observed_tokenizer_class!r}"
        )
    observed_vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(observed_vocab_size, bool) or observed_vocab_size != expected_tokenizer_vocab_size:
        raise ProtocolValidationError(
            f"runtime tokenizer vocab mismatch: expected {expected_tokenizer_vocab_size}, "
            f"got {observed_vocab_size!r}"
        )
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise ProtocolValidationError("runtime tokenizer must expose encode")

    normalized_expected_seeds = tuple(
        _as_int(seed, f"expected_seeds[{index}]")
        for index, seed in enumerate(expected_seeds)
    )
    if len(normalized_expected_seeds) != len(set(normalized_expected_seeds)):
        raise ProtocolValidationError("expected seeds contain a duplicate")
    manifest_seed_values = _as_list(manifest["seeds"], "seeds")
    manifest_seeds = tuple(
        _as_int(seed, f"seeds[{index}]")
        for index, seed in enumerate(manifest_seed_values)
    )
    if manifest_seeds != normalized_expected_seeds:
        raise ProtocolValidationError(
            f"seed manifest mismatch: expected {normalized_expected_seeds}, got {manifest_seeds}"
        )
    selected_seed = _as_int(experiment_seed, "experiment_seed")
    if selected_seed not in manifest_seeds:
        raise ProtocolValidationError(f"experiment seed {selected_seed} is absent from the protocol")
    if evaluation_role not in ("validation", "test"):
        raise ProtocolValidationError("evaluation_role must be exactly 'validation' or 'test'")

    selection_rules = _as_mapping(manifest["selection"], "selection")
    _exact_keys(
        selection_rules,
        {
            "calibration_selection_seed",
            "heldout_priority",
            "row_consumption",
            "test_selection_seed",
            "validation_selection_seed",
        },
        "selection",
    )
    _expect_equal(
        selection_rules["heldout_priority"],
        ["test", "validation", "calibration"],
        "selection.heldout_priority",
    )
    _expect_equal(
        selection_rules["test_selection_seed"],
        2026071401,
        "selection.test_selection_seed",
    )
    _expect_equal(
        selection_rules["validation_selection_seed"],
        2026071402,
        "selection.validation_selection_seed",
    )
    _expect_equal(
        selection_rules["calibration_selection_seed"],
        "the corresponding experiment seed",
        "selection.calibration_selection_seed",
    )
    _expect_equal(
        selection_rules["row_consumption"],
        "a source row is consumed by one candidate only; unused tail tokens are discarded",
        "selection.row_consumption",
    )

    cache_dir = None if dataset_cache_dir is None else Path(dataset_cache_dir)
    if dataset_loader is None:
        loaded = _load_local_dataset(
            expected_dataset_name,
            expected_dataset_config,
            cache_dir=cache_dir,
        )
    else:
        if dataset_cache_dir is not None:
            raise ProtocolValidationError(
                "dataset_cache_dir cannot be combined with an injected dataset_loader"
            )
        loaded = dataset_loader(expected_dataset_name, expected_dataset_config)
    if not isinstance(loaded, LoadedProtocolDataset):
        raise ProtocolValidationError("dataset loader must return LoadedProtocolDataset")
    _expect_equal(loaded.name, expected_dataset_name, "loaded dataset name")
    _expect_equal(loaded.config, expected_dataset_config, "loaded dataset config")
    _expect_equal(loaded.fallback_used, False, "loaded dataset fallback flag")
    if set(loaded.splits) != set(_SPLITS):
        raise ProtocolValidationError("loaded dataset must contain exactly train, validation, and test")
    loaded_fingerprints = _normalize_fingerprints(
        loaded.fingerprints,
        "loaded dataset fingerprints",
    )
    if loaded_fingerprints != manifest_fingerprints:
        raise ProtocolValidationError("loaded dataset fingerprint mismatch")

    allocation_counts = _as_mapping(manifest["allocation_counts"], "allocation_counts")
    _exact_keys(
        allocation_counts,
        {
            "calibration_window_total",
            "calibration_windows_per_seed",
            "test_windows",
            "total_windows",
            "validation_windows",
        },
        "allocation_counts",
    )
    expected_count_values = {
        "calibration_windows_per_seed": CALIBRATION_WINDOWS_PER_SEED,
        "calibration_window_total": CALIBRATION_WINDOWS_PER_SEED * len(manifest_seeds),
        "validation_windows": VALIDATION_WINDOWS,
        "test_windows": TEST_WINDOWS,
        "total_windows": (
            CALIBRATION_WINDOWS_PER_SEED * len(manifest_seeds)
            + VALIDATION_WINDOWS
            + TEST_WINDOWS
        ),
    }
    for key, expected in expected_count_values.items():
        _expect_equal(allocation_counts[key], expected, f"allocation_counts.{key}")

    windows = _as_mapping(manifest["windows"], "windows")
    _exact_keys(windows, {"calibration_by_seed", "validation", "test"}, "windows")
    calibration_by_seed = _as_mapping(
        windows["calibration_by_seed"], "windows.calibration_by_seed"
    )
    expected_seed_keys = {str(seed) for seed in manifest_seeds}
    if set(calibration_by_seed) != expected_seed_keys:
        raise ProtocolValidationError(
            "calibration_by_seed keys do not exactly match the preregistered seeds"
        )

    row_id_pattern = re.compile(
        rf"{re.escape(expected_dataset_config)}/(train|validation|test)/(0|[1-9][0-9]*)\Z"
    )
    seen_row_ids: set[str] = set()
    seen_normalized_hashes: set[str] = set()
    seen_window_ids: set[str] = set()
    source_row_ids_in_order: list[str] = []

    def reconstruct_group(
        raw_windows: object,
        *,
        role: Literal["calibration", "validation", "test"],
        seed: int | None,
        split: Literal["train", "validation", "test"],
        expected_count: int,
    ) -> tuple[ProtocolWindow, ...]:
        entries = _as_list(raw_windows, f"windows.{role}")
        if len(entries) != expected_count:
            raise ProtocolValidationError(
                f"{role} window count mismatch: expected {expected_count}, got {len(entries)}"
            )
        reconstructed: list[ProtocolWindow] = []
        seed_label = f"seed-{seed}" if seed is not None else "fixed"
        for window_index, raw_window in enumerate(entries):
            context = f"{role}[{window_index}]"
            window = _as_mapping(raw_window, context)
            _exact_keys(window, {"allocation", "sources", "token_length", "window_id"}, context)
            expected_window_id = f"{role}/{seed_label}/{window_index:03d}"
            window_id = _as_string(window["window_id"], f"{context}.window_id")
            if window_id != expected_window_id:
                raise ProtocolValidationError(
                    f"window id mismatch: expected {expected_window_id!r}, got {window_id!r}"
                )
            if window_id in seen_window_ids:
                raise ProtocolValidationError(f"duplicate window id {window_id!r}")
            seen_window_ids.add(window_id)
            _expect_equal(window["token_length"], WINDOW_TOKEN_LENGTH, f"{context}.token_length")

            window_allocation = _as_mapping(window["allocation"], f"{context}.allocation")
            _exact_keys(
                window_allocation,
                {"native_split", "role", "seed", "window_index"},
                f"{context}.allocation",
            )
            _expect_equal(window_allocation["native_split"], split, f"{context} native split")
            _expect_equal(window_allocation["role"], role, f"{context} role")
            _expect_equal(window_allocation["seed"], seed, f"{context} seed")
            _expect_equal(window_allocation["window_index"], window_index, f"{context} index")

            raw_sources = _as_list(window["sources"], f"{context}.sources")
            if not raw_sources:
                raise ProtocolValidationError(f"{context} has no source rows")
            cursor = 0
            reconstructed_tokens: list[int] = []
            for source_index, raw_source in enumerate(raw_sources):
                source_context = f"{context}.sources[{source_index}]"
                source = _as_mapping(raw_source, source_context)
                _exact_keys(
                    source,
                    {"allocation", "normalized_sha256", "row_id", "token_length"},
                    source_context,
                )
                row_id = _as_string(source["row_id"], f"{source_context}.row_id")
                match = row_id_pattern.fullmatch(row_id)
                if match is None:
                    raise ProtocolValidationError(f"malformed source row id {row_id!r}")
                row_split = match.group(1)
                row_index = int(match.group(2))
                if row_split != split:
                    raise ProtocolValidationError(
                        f"source row {row_id!r} does not belong to required {split!r} split"
                    )
                if row_id in seen_row_ids:
                    raise ProtocolValidationError(f"duplicate source row {row_id!r}")
                seen_row_ids.add(row_id)
                source_row_ids_in_order.append(row_id)

                declared_hash = _as_string(
                    source["normalized_sha256"],
                    f"{source_context}.normalized_sha256",
                )
                if _SHA256_RE.fullmatch(declared_hash) is None or declared_hash != declared_hash.lower():
                    raise ProtocolValidationError(
                        f"{source_context}.normalized_sha256 must be lowercase SHA256"
                    )
                if declared_hash in seen_normalized_hashes:
                    raise ProtocolValidationError(
                        f"duplicate normalized source content hash {declared_hash}"
                    )
                seen_normalized_hashes.add(declared_hash)

                split_rows = loaded.splits[split]
                try:
                    split_length = len(split_rows)
                except TypeError as exc:
                    raise ProtocolValidationError(f"loaded {split} split has no stable length") from exc
                if row_index < 0 or row_index >= split_length:
                    raise ProtocolValidationError(f"source row {row_id!r} is out of bounds")
                raw_row = split_rows[row_index]
                if not isinstance(raw_row, Mapping) or "text" not in raw_row:
                    raise ProtocolValidationError(f"source row {row_id!r} has no raw text field")
                raw_text = raw_row["text"]
                if not isinstance(raw_text, str):
                    raise ProtocolValidationError(f"source row {row_id!r} text must be a string")

                actual_normalized_hash = _sha256_text(_normalize_text(raw_text))
                if not hmac.compare_digest(actual_normalized_hash, declared_hash):
                    raise ProtocolValidationError(
                        f"normalized row hash mismatch for {row_id!r}"
                    )
                try:
                    encoded = encode(raw_text, add_special_tokens=False)
                except Exception as exc:
                    raise ProtocolValidationError(f"tokenization failed for {row_id!r}") from exc
                if isinstance(encoded, (str, bytes)):
                    raise ProtocolValidationError(f"tokenizer returned invalid IDs for {row_id!r}")
                try:
                    full_row_tokens = tuple(encoded)
                except TypeError as exc:
                    raise ProtocolValidationError(f"tokenizer returned non-iterable IDs for {row_id!r}") from exc
                checked_tokens: list[int] = []
                for token_position, token_id in enumerate(full_row_tokens):
                    if isinstance(token_id, bool) or not isinstance(token_id, numbers.Integral):
                        raise ProtocolValidationError(
                            f"token {token_position} for {row_id!r} is not an integer"
                        )
                    value = int(token_id)
                    if value < 0 or value >= expected_tokenizer_vocab_size:
                        raise ProtocolValidationError(
                            f"token {token_position} for {row_id!r} is outside the tokenizer vocabulary"
                        )
                    checked_tokens.append(value)
                row_tokens = tuple(checked_tokens)
                declared_token_length = _as_int(
                    source["token_length"], f"{source_context}.token_length"
                )
                if declared_token_length != len(row_tokens):
                    raise ProtocolValidationError(
                        f"raw-token full length mismatch for {row_id!r}: "
                        f"manifest={declared_token_length}, reconstructed={len(row_tokens)}"
                    )

                allocation = _as_mapping(source["allocation"], f"{source_context}.allocation")
                _exact_keys(
                    allocation,
                    {
                        "row_token_start",
                        "row_token_stop",
                        "window_token_start",
                        "window_token_stop",
                    },
                    f"{source_context}.allocation",
                )
                row_start = _as_int(
                    allocation["row_token_start"], f"{source_context}.row_token_start"
                )
                row_stop = _as_int(
                    allocation["row_token_stop"], f"{source_context}.row_token_stop"
                )
                window_start = _as_int(
                    allocation["window_token_start"], f"{source_context}.window_token_start"
                )
                window_stop = _as_int(
                    allocation["window_token_stop"], f"{source_context}.window_token_stop"
                )
                if row_start != 0:
                    raise ProtocolValidationError(
                        f"{source_context} violates v2 row_token_start=0 rule"
                    )
                if not (0 <= row_start < row_stop <= len(row_tokens)):
                    raise ProtocolValidationError(f"{source_context} row allocation is out of bounds")
                if window_start != cursor:
                    raise ProtocolValidationError(
                        f"{source_context} window allocation has a gap or overlap: "
                        f"expected start {cursor}, got {window_start}"
                    )
                if not (0 <= window_start < window_stop <= WINDOW_TOKEN_LENGTH):
                    raise ProtocolValidationError(
                        f"{source_context} window allocation is out of bounds"
                    )
                if row_stop - row_start != window_stop - window_start:
                    raise ProtocolValidationError(
                        f"{source_context} row/window allocation lengths differ"
                    )
                if row_stop < len(row_tokens) and not (
                    source_index == len(raw_sources) - 1
                    and window_stop == WINDOW_TOKEN_LENGTH
                ):
                    raise ProtocolValidationError(
                        f"{source_context} discards a row tail before the window is complete"
                    )
                reconstructed_tokens.extend(row_tokens[row_start:row_stop])
                cursor = window_stop

            if cursor != WINDOW_TOKEN_LENGTH:
                raise ProtocolValidationError(
                    f"{context} allocation does not exactly cover {WINDOW_TOKEN_LENGTH} tokens"
                )
            token_ids = tuple(reconstructed_tokens)
            if len(token_ids) != WINDOW_TOKEN_LENGTH:
                raise ProtocolValidationError(
                    f"{context} reconstructed token length is {len(token_ids)}, "
                    f"expected {WINDOW_TOKEN_LENGTH}"
                )
            reconstructed.append(
                ProtocolWindow(
                    window_id=window_id,
                    role=role,
                    seed=seed,
                    token_ids=token_ids,
                    token_digest=token_ids_sha256(token_ids),
                )
            )
        return tuple(reconstructed)

    # Reconstruct every manifest window, not only the selected seed.  This is
    # required for the global source-row uniqueness guarantee.
    test_windows = reconstruct_group(
        windows["test"],
        role="test",
        seed=None,
        split="test",
        expected_count=TEST_WINDOWS,
    )
    validation_windows = reconstruct_group(
        windows["validation"],
        role="validation",
        seed=None,
        split="validation",
        expected_count=VALIDATION_WINDOWS,
    )
    all_calibration: dict[int, tuple[ProtocolWindow, ...]] = {}
    for seed in manifest_seeds:
        all_calibration[seed] = reconstruct_group(
            calibration_by_seed[str(seed)],
            role="calibration",
            seed=seed,
            split="train",
            expected_count=CALIBRATION_WINDOWS_PER_SEED,
        )

    actual_total_windows = (
        len(test_windows)
        + len(validation_windows)
        + sum(len(seed_windows) for seed_windows in all_calibration.values())
    )
    _expect_equal(
        actual_total_windows,
        expected_count_values["total_windows"],
        "reconstructed total window count",
    )
    audits = _as_mapping(manifest["audits"], "audits")
    _exact_keys(
        audits,
        {
            "allocated_source_row_count",
            "allocated_source_row_id_unique_count",
            "exact_content",
            "near_duplicate",
            "near_duplicate_candidate_rejections",
        },
        "audits",
    )
    exact_content_audit = _as_mapping(audits["exact_content"], "audits.exact_content")
    _expect_equal(
        exact_content_audit.get("normalization"),
        "Unicode NFKC, lowercase, then collapse all whitespace runs to one ASCII space",
        "audits.exact_content.normalization",
    )
    _expect_equal(
        exact_content_audit.get("normalization_scope"),
        "exact-content SHA256 deduplication only",
        "audits.exact_content.normalization_scope",
    )
    _expect_equal(
        exact_content_audit.get("tokenization_input"),
        "native raw dataset text without normalization",
        "audits.exact_content.tokenization_input",
    )
    _expect_equal(
        exact_content_audit.get("deduplication_priority"),
        ["test", "validation", "train"],
        "audits.exact_content.deduplication_priority",
    )
    near_duplicate_audit = _as_mapping(
        audits["near_duplicate"], "audits.near_duplicate"
    )
    _expect_equal(
        near_duplicate_audit.get("token_ngram_n"), 5, "audits.near_duplicate.token_ngram_n"
    )
    _expect_equal(
        near_duplicate_audit.get("jaccard_threshold"),
        0.8,
        "audits.near_duplicate.jaccard_threshold",
    )
    _expect_equal(
        near_duplicate_audit.get("pairs_checked"),
        actual_total_windows * (actual_total_windows - 1) // 2,
        "audits.near_duplicate.pairs_checked",
    )
    _expect_equal(
        near_duplicate_audit.get("violation_count"),
        0,
        "audits.near_duplicate.violation_count",
    )
    _expect_equal(
        near_duplicate_audit.get("violations"), [], "audits.near_duplicate.violations"
    )
    allocated_count = len(source_row_ids_in_order)
    _expect_equal(
        audits.get("allocated_source_row_count"),
        allocated_count,
        "audits.allocated_source_row_count",
    )
    _expect_equal(
        audits.get("allocated_source_row_id_unique_count"),
        allocated_count,
        "audits.allocated_source_row_id_unique_count",
    )

    selected_calibration = all_calibration[selected_seed]
    evaluation_windows = validation_windows if evaluation_role == "validation" else test_windows
    calibration_sha256 = digest_protocol_windows(selected_calibration)
    validation_sha256 = digest_protocol_windows(validation_windows)
    test_sha256 = digest_protocol_windows(test_windows)
    evaluation_sha256 = (
        validation_sha256 if evaluation_role == "validation" else test_sha256
    )
    all_calibration_windows = tuple(
        window
        for seed in manifest_seeds
        for window in all_calibration[seed]
    )
    all_windows = (*test_windows, *validation_windows, *all_calibration_windows)
    provenance = ProtocolProvenance(
        manifest_path=str(resolved_manifest),
        manifest_sha256=manifest_sha256,
        schema_version=SCHEMA_VERSION,
        selected_seed=selected_seed,
        evaluation_role=evaluation_role,
        window_token_length=WINDOW_TOKEN_LENGTH,
        calibration_window_ids=tuple(window.window_id for window in selected_calibration),
        calibration_window_count=len(selected_calibration),
        calibration_token_sha256=calibration_sha256,
        evaluation_window_ids=tuple(window.window_id for window in evaluation_windows),
        evaluation_window_count=len(evaluation_windows),
        evaluation_token_sha256=evaluation_sha256,
        consumed=True,
        status=PROTOCOL_STATUS,
        protocol_date=PROTOCOL_DATE,
        model_id=expected_model_id,
        model_snapshot_commit=expected_model_snapshot_commit,
        tokenizer_class=expected_tokenizer_class,
        tokenizer_vocab_size=expected_tokenizer_vocab_size,
        tokenizer_snapshot_commit=expected_tokenizer_snapshot_commit,
        dataset_name=expected_dataset_name,
        dataset_config=expected_dataset_config,
        dataset_fingerprints=manifest_fingerprints,
        dataset_local_cache_only=True,
        dataset_fallback_allowed=False,
        tokenizer_add_special_tokens=False,
        tokenization_input="native raw dataset text without normalization",
        source_row_reuse_allowed=False,
        manifest_seeds=manifest_seeds,
        epsilon_grid=EPSILON_GRID,
        local_fit_positive_epsilons=LOCAL_FIT_EPSILONS,
        validation_window_ids=tuple(window.window_id for window in validation_windows),
        validation_token_sha256=validation_sha256,
        test_window_ids=tuple(window.window_id for window in test_windows),
        test_token_sha256=test_sha256,
        all_calibration_window_ids=tuple(
            window.window_id for window in all_calibration_windows
        ),
        calibration_token_sha256_by_seed=tuple(
            (seed, digest_protocol_windows(all_calibration[seed]))
            for seed in manifest_seeds
        ),
        all_window_count=actual_total_windows,
        all_window_token_sha256=digest_protocol_windows(all_windows),
        allocated_source_row_count=allocated_count,
        allocated_source_row_ids_sha256=hashlib.sha256(
            _canonical_json_bytes(source_row_ids_in_order)
        ).hexdigest(),
    )
    return ConfirmatoryProtocolSelection(
        selected_calibration_windows=selected_calibration,
        evaluation_windows=evaluation_windows,
        validation_windows=validation_windows,
        test_windows=test_windows,
        provenance=provenance,
    )


# Short, explicit alias for runner integrations.
load_protocol_windows = consume_confirmatory_protocol
