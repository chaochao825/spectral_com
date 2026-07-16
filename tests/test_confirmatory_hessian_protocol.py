from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_confirmatory_hessian_protocol",
    REPO_ROOT / "scripts" / "build_confirmatory_hessian_protocol.py",
)
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOCOL
SPEC.loader.exec_module(PROTOCOL)


class FixtureTokenizer:
    vocab_size = 50_304

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [
            int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
            for token in text.split()
        ]


def _fixture_rows(split: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in range(count):
        # Every row has enough unique tokens for one complete window.  The raw
        # marker lets the test prove that source text is absent from JSON.
        tokens = [f"rawpayload_{split}_{row}_{token}" for token in range(260)]
        rows.append({"text": " ".join(tokens)})
    return rows


def _fixture_splits() -> dict[str, list[dict[str, str]]]:
    return {
        "train": _fixture_rows("train", 280),
        "validation": _fixture_rows("validation", 40),
        "test": _fixture_rows("test", 72),
    }


@pytest.fixture(scope="module")
def fixture_manifest() -> dict[str, object]:
    return PROTOCOL.build_protocol(
        _fixture_splits(),
        FixtureTokenizer(),
        dataset_fingerprints={"train": "train-fp", "validation": "val-fp", "test": "test-fp"},
    )


def test_normalization_and_exact_content_hashing() -> None:
    left = PROTOCOL.normalize_text("  ＨＥＬＬＯ\tStraße\n")
    right = PROTOCOL.normalize_text("hello straße")
    assert left == right == "hello straße"
    assert PROTOCOL.normalized_sha256(left) == PROTOCOL.normalized_sha256(right)
    assert PROTOCOL.normalize_text(None) == ""


def test_token_five_gram_jaccard_threshold_is_inclusive() -> None:
    left = PROTOCOL.token_ngrams(tuple(range(20)))
    right_values = list(range(20))
    right_values[-1] = 999
    right = PROTOCOL.token_ngrams(tuple(right_values))
    score = PROTOCOL.ngram_jaccard(left, right)
    assert score == pytest.approx(15 / 17)
    assert score >= PROTOCOL.NEAR_DUPLICATE_THRESHOLD


def test_exact_deduplication_prioritizes_heldout_rows() -> None:
    splits = {
        "train": [{"text": "Ａ   repeated row"}, {"text": "unique train row"}],
        "validation": [{"text": "a repeated\trow"}],
        "test": [{"text": "unique test row"}],
    }
    prepared, audit = PROTOCOL.prepare_unique_rows(splits, FixtureTokenizer())
    assert [row.row_id for row in prepared["validation"]] == [
        "wikitext-2-raw-v1/validation/0"
    ]
    assert [row.row_id for row in prepared["train"]] == ["wikitext-2-raw-v1/train/1"]
    assert audit["exact_duplicate_removed_count"]["train"] == 1


def test_dedup_normalization_does_not_change_tokenizer_input() -> None:
    class RecordingTokenizer:
        vocab_size = 1

        def __init__(self) -> None:
            self.inputs: list[str] = []

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            assert add_special_tokens is False
            self.inputs.append(text)
            return [len(text)]

    tokenizer = RecordingTokenizer()
    raw = "  ＨＥＬＬＯ\tStraße\n"
    splits = {
        "train": [{"text": "unique train"}],
        "validation": [{"text": "unique validation"}],
        "test": [{"text": raw}],
    }
    prepared, audit = PROTOCOL.prepare_unique_rows(splits, tokenizer)
    assert prepared["test"][0].normalized_text == "hello straße"
    assert tokenizer.inputs[0] == raw
    assert audit["normalization_scope"] == "exact-content SHA256 deduplication only"
    assert audit["tokenization_input"] == "native raw dataset text without normalization"


def test_protocol_has_fixed_disjoint_allocations_and_no_raw_text(
    tmp_path: Path,
    fixture_manifest: dict[str, object],
) -> None:
    manifest = fixture_manifest
    assert manifest["schema_version"] == "confirmatory_hessian_protocol.v2"
    assert manifest["status"] == "preregistered_data_split_manifest"
    assert manifest["model"] == {
        "model_id": "EleutherAI/pythia-70m",
        "snapshot_commit": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "weights_loaded_by_this_script": False,
    }
    assert manifest["seeds"] == [17, 29, 43, 59, 71, 89, 101, 113]
    assert manifest["epsilon_grid"] == [
        0.0,
        1 / 32,
        1 / 16,
        3 / 32,
        1 / 8,
        3 / 16,
        1 / 4,
        3 / 8,
        1 / 2,
        5 / 8,
        3 / 4,
        7 / 8,
        1.0,
    ]
    assert manifest["local_linear_quadratic_fit_positive_epsilons"] == [1 / 32, 1 / 16, 3 / 32, 1 / 8]
    assert manifest["tokenization"]["input_text"] == "native raw dataset text without normalization"

    windows = manifest["windows"]
    assert len(windows["validation"]) == 32
    assert len(windows["test"]) == 64
    assert {seed: len(items) for seed, items in windows["calibration_by_seed"].items()} == {
        str(seed): 32 for seed in manifest["seeds"]
    }

    all_windows = [*windows["validation"], *windows["test"]]
    for seed_windows in windows["calibration_by_seed"].values():
        all_windows.extend(seed_windows)
    assert len(all_windows) == 352
    assert all(window["token_length"] == 256 for window in all_windows)
    row_ids = [source["row_id"] for window in all_windows for source in window["sources"]]
    assert len(row_ids) == len(set(row_ids))
    assert manifest["audits"]["near_duplicate"]["pairs_checked"] == 352 * 351 // 2
    assert manifest["audits"]["near_duplicate"]["violation_count"] == 0

    serialized = PROTOCOL.render_protocol_json(manifest)
    assert "rawpayload_" not in serialized
    decoded = json.loads(serialized)
    allowed_source_keys = {"row_id", "normalized_sha256", "token_length", "allocation"}
    assert all(set(source) == allowed_source_keys for window in all_windows for source in window["sources"])
    assert all(len(source["normalized_sha256"]) == 64 for window in all_windows for source in window["sources"])

    paths = PROTOCOL.write_outputs(tmp_path, manifest)
    assert {path.name for path in paths} == {"protocol.json", "summary.md"}
    PROTOCOL.write_outputs(tmp_path, manifest, check=True)


def test_protocol_refuses_insufficient_data_instead_of_falling_back() -> None:
    splits = {
        "train": _fixture_rows("train", 2),
        "validation": _fixture_rows("validation", 2),
        "test": _fixture_rows("test", 2),
    }
    with pytest.raises(RuntimeError, match="insufficient unique local test data"):
        PROTOCOL.build_protocol(splits, FixtureTokenizer())


def test_check_detects_manifest_drift(tmp_path: Path, fixture_manifest: dict[str, object]) -> None:
    PROTOCOL.write_outputs(tmp_path, fixture_manifest)
    (tmp_path / "summary.md").write_text("stale\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="drift detected"):
        PROTOCOL.write_outputs(tmp_path, fixture_manifest, check=True)
