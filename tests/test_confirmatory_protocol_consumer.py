from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_script(name: str, relative_path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _import_script(
    "build_confirmatory_hessian_protocol_for_consumer_tests",
    "scripts/build_confirmatory_hessian_protocol.py",
)
CONSUMER = _import_script(
    "confirmatory_protocol_windows",
    "scripts/confirmatory_protocol_windows.py",
)


class FixtureTokenizer:
    """Byte tokenizer that makes leading/trailing raw whitespace observable."""

    vocab_size = 256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


FINGERPRINTS = {
    "train": "fixture-train-fingerprint",
    "validation": "fixture-validation-fingerprint",
    "test": "fixture-test-fingerprint",
}


def _fixture_rows(split: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row_index in range(count):
        digest = hashlib.sha256(f"{split}/{row_index}".encode("ascii")).hexdigest()
        # Exactly 140 raw bytes.  Each 256-token window consequently needs two
        # rows, which gives the gap/overlap tests a real source boundary.
        raw_text = " " + (digest * 3)[:137] + " \n"
        assert len(raw_text.encode("utf-8")) == 140
        rows.append({"text": raw_text})
    return rows


@pytest.fixture(scope="module")
def protocol_fixture() -> tuple[dict[str, object], dict[str, list[dict[str, str]]]]:
    splits = {
        "train": _fixture_rows("train", 530),
        "validation": _fixture_rows("validation", 72),
        "test": _fixture_rows("test", 136),
    }
    manifest = BUILDER.build_protocol(
        splits,
        FixtureTokenizer(),
        dataset_fingerprints=FINGERPRINTS,
    )
    return manifest, splits


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> tuple[Path, str]:
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = tmp_path / "protocol.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _consume(
    tmp_path: Path,
    manifest: dict[str, object],
    splits: dict[str, list[dict[str, str]]],
    *,
    expected_sha256: str | None = None,
    seed: int = 17,
    evaluation_role: str = "test",
    fallback_used: bool = False,
) -> object:
    path, actual_sha256 = _write_manifest(tmp_path, manifest)

    def loader(name: str, config: str) -> object:
        return CONSUMER.LoadedProtocolDataset(
            name=name,
            config=config,
            splits=splits,
            fingerprints=FINGERPRINTS,
            fallback_used=fallback_used,
        )

    return CONSUMER.consume_confirmatory_protocol(
        path,
        expected_sha256=actual_sha256 if expected_sha256 is None else expected_sha256,
        experiment_seed=seed,
        tokenizer=FixtureTokenizer(),
        evaluation_role=evaluation_role,
        dataset_loader=loader,
        expected_tokenizer_class="FixtureTokenizer",
        expected_tokenizer_vocab_size=256,
        expected_dataset_fingerprints=FINGERPRINTS,
    )


def _mutated(
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
    mutate: Callable[[dict[str, object]], None],
) -> tuple[dict[str, object], dict[str, list[dict[str, str]]]]:
    manifest, splits = protocol_fixture
    changed = copy.deepcopy(manifest)
    mutate(changed)
    return changed, splits


def _first_test_window(manifest: dict[str, object]) -> dict[str, object]:
    return manifest["windows"]["test"][0]  # type: ignore[index,return-value]


def test_exact_reconstruction_selection_digests_and_immutability(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    manifest, splits = protocol_fixture
    selection = _consume(tmp_path, manifest, splits, evaluation_role="test")

    assert len(selection.selected_calibration_windows) == 32
    assert len(selection.validation_windows) == 32
    assert len(selection.test_windows) == 64
    assert selection.evaluation_windows == selection.test_windows
    assert selection.calibration_windows == selection.selected_calibration_windows

    protocol_window = manifest["windows"]["calibration_by_seed"]["17"][0]  # type: ignore[index]
    expected_tokens: list[int] = []
    for source in protocol_window["sources"]:
        split, row_index_text = source["row_id"].rsplit("/", 2)[-2:]
        raw_text = splits[split][int(row_index_text)]["text"]
        full_tokens = FixtureTokenizer().encode(raw_text, add_special_tokens=False)
        allocation = source["allocation"]
        expected_tokens.extend(
            full_tokens[allocation["row_token_start"] : allocation["row_token_stop"]]
        )
    first = selection.selected_calibration_windows[0]
    assert first.window_id == "calibration/seed-17/000"
    assert first.role == "calibration"
    assert first.seed == 17
    assert first.token_ids == tuple(expected_tokens)
    assert first.token_ids[0] == ord(" ")  # proves native text was not stripped
    assert first.token_digest == CONSUMER.token_ids_sha256(first.token_ids)

    provenance = selection.provenance
    assert provenance.schema_version == "confirmatory_hessian_protocol.v2"
    assert provenance.selected_seed == 17
    assert provenance.evaluation_role == "test"
    assert provenance.window_token_length == 256
    assert provenance.calibration_window_count == 32
    assert provenance.evaluation_window_count == 64
    assert provenance.calibration_window_ids == tuple(
        window.window_id for window in selection.selected_calibration_windows
    )
    assert provenance.evaluation_window_ids == tuple(
        window.window_id for window in selection.evaluation_windows
    )
    assert provenance.calibration_token_sha256 == CONSUMER.digest_protocol_windows(
        selection.selected_calibration_windows
    )
    assert provenance.evaluation_token_sha256 == CONSUMER.digest_protocol_windows(
        selection.evaluation_windows
    )
    assert provenance.consumed is True
    with pytest.raises(FrozenInstanceError):
        first.seed = 29


def test_wrong_external_manifest_sha_is_rejected_before_json_use(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    manifest, splits = protocol_fixture
    with pytest.raises(CONSUMER.ProtocolValidationError, match="manifest SHA256 mismatch"):
        _consume(tmp_path, manifest, splits, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest.__setitem__(
                "schema_version", "confirmatory_hessian_protocol.v1"
            ),
            "schema_version",
        ),
        (lambda manifest: manifest.__setitem__("seeds", [19, 29, 43, 59, 71, 89, 101, 113]), "seed manifest"),
        (lambda manifest: manifest["model"].__setitem__("model_id", "other/model"), "model.model_id"),
        (lambda manifest: manifest["model"].__setitem__("snapshot_commit", "bad-snapshot"), "model.snapshot_commit"),
        (
            lambda manifest: manifest["tokenization"].__setitem__(
                "tokenizer_class", "OtherTokenizer"
            ),
            "tokenizer class",
        ),
        (lambda manifest: manifest["tokenization"].__setitem__("vocab_size", 257), "tokenizer vocab"),
        (
            lambda manifest: manifest["tokenization"].__setitem__(
                "snapshot_commit", "bad-snapshot"
            ),
            "tokenizer snapshot",
        ),
        (lambda manifest: manifest["dataset"].__setitem__("name", "other"), "dataset.name"),
        (lambda manifest: manifest["dataset"].__setitem__("config", "other"), "dataset.config"),
        (lambda manifest: manifest["dataset"]["fingerprints"].__setitem__("train", "wrong"), "dataset fingerprint"),
    ],
    ids=[
        "schema",
        "seed",
        "model-id",
        "model-snapshot",
        "tokenizer-class",
        "tokenizer-vocab",
        "tokenizer-snapshot",
        "dataset-name",
        "dataset-config",
        "dataset-fingerprint",
    ],
)
def test_identity_tampering_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    manifest, splits = _mutated(protocol_fixture, mutate)
    with pytest.raises(CONSUMER.ProtocolValidationError, match=message):
        _consume(tmp_path, manifest, splits)


def test_normalized_row_hash_tamper_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    manifest, splits = _mutated(
        protocol_fixture,
        lambda value: _first_test_window(value)["sources"][0].__setitem__(
            "normalized_sha256", "0" * 64
        ),
    )
    with pytest.raises(CONSUMER.ProtocolValidationError, match="normalized row hash mismatch"):
        _consume(tmp_path, manifest, splits)


def test_raw_token_full_length_tamper_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        source = _first_test_window(manifest)["sources"][0]
        source["token_length"] += 1

    manifest, splits = _mutated(protocol_fixture, mutate)
    with pytest.raises(CONSUMER.ProtocolValidationError, match="raw-token full length mismatch"):
        _consume(tmp_path, manifest, splits)


@pytest.mark.parametrize(
    ("delta", "message"),
    [(1, "gap or overlap"), (-1, "gap or overlap")],
    ids=["gap", "overlap"],
)
def test_window_source_gap_or_overlap_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
    delta: int,
    message: str,
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        second_allocation = _first_test_window(manifest)["sources"][1]["allocation"]
        second_allocation["window_token_start"] += delta
        second_allocation["window_token_stop"] += delta

    manifest, splits = _mutated(protocol_fixture, mutate)
    with pytest.raises(CONSUMER.ProtocolValidationError, match=message):
        _consume(tmp_path, manifest, splits)


def test_out_of_bounds_row_allocation_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        allocation = _first_test_window(manifest)["sources"][0]["allocation"]
        allocation["row_token_stop"] = 999

    manifest, splits = _mutated(protocol_fixture, mutate)
    with pytest.raises(CONSUMER.ProtocolValidationError, match="row allocation is out of bounds"):
        _consume(tmp_path, manifest, splits)


def test_duplicate_source_row_anywhere_in_manifest_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        sources = _first_test_window(manifest)["sources"]
        sources[1]["row_id"] = sources[0]["row_id"]

    manifest, splits = _mutated(protocol_fixture, mutate)
    with pytest.raises(CONSUMER.ProtocolValidationError, match="duplicate source row"):
        _consume(tmp_path, manifest, splits)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("role", "validation", "role mismatch"),
        ("seed", 17, "seed mismatch"),
        ("window_index", 1, "index mismatch"),
    ],
)
def test_window_allocation_identity_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, splits = _mutated(
        protocol_fixture,
        lambda item: _first_test_window(item)["allocation"].__setitem__(field, value),
    )
    with pytest.raises(CONSUMER.ProtocolValidationError, match=message):
        _consume(tmp_path, manifest, splits)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("window_id", "test/fixed/999", "window id mismatch"),
        ("token_length", 255, "token_length"),
    ],
)
def test_window_id_and_length_tampering_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, splits = _mutated(
        protocol_fixture,
        lambda item: _first_test_window(item).__setitem__(field, value),
    )
    with pytest.raises(CONSUMER.ProtocolValidationError, match=message):
        _consume(tmp_path, manifest, splits)


def test_fallback_dataset_is_rejected(
    tmp_path: Path,
    protocol_fixture: tuple[dict[str, object], dict[str, list[dict[str, str]]]],
) -> None:
    manifest, splits = protocol_fixture
    with pytest.raises(CONSUMER.ProtocolValidationError, match="fallback flag"):
        _consume(tmp_path, manifest, splits, fallback_used=True)
