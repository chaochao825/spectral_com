from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_raw_artifact_index",
    REPO_ROOT / "scripts" / "build_raw_artifact_index.py",
)
assert SPEC is not None and SPEC.loader is not None
INDEX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INDEX
SPEC.loader.exec_module(INDEX)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_payload_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "artifact_path",
                "artifact_file_bytes",
                "artifact_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object], list[dict[str, object]]]:
    root = tmp_path / "suite"
    job = root / "jobs" / "model_seed17_rate0258"
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    reference = artifacts / "reference_fp16.hrc"
    strategy = artifacts / "q_plus_l.hrc"
    reference.write_bytes(b"reference-payload")
    strategy.write_bytes(b"strategy")
    strategy_rows: list[dict[str, object]] = [
        {
            "strategy": "Q+L",
            "artifact_path": "artifacts/q_plus_l.hrc",
            "artifact_file_bytes": strategy.stat().st_size,
            "artifact_sha256": _sha256(strategy),
        }
    ]
    manifest: dict[str, object] = {
        "format": "llm_spectral_dynamics_research_codec",
        "reference": {
            "path": "artifacts/reference_fp16.hrc",
            "file_bytes": reference.stat().st_size,
            "sha256": _sha256(reference),
        },
        "strategies": strategy_rows,
    }
    (job / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_payload_csv(job / "artifact_payloads.csv", strategy_rows)
    commit = "0123456789abcdef0123456789abcdef01234567"
    (job / "run_config.json").write_text(
        json.dumps({"git": {"commit": commit, "dirty": False}}), encoding="utf-8"
    )
    (root / "suite_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "large_scale_hessian_suite_manifest.v1",
                "suite_id": "unit_raw_suite",
                "git": {"commit": commit, "dirty": False},
                "jobs": [
                    {"job_id": job.name, "status": "completed_valid", "exit_code": 0}
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, reference, strategy, manifest, strategy_rows


def test_builds_json_and_csv_index_without_copying_raw_artifacts(tmp_path: Path) -> None:
    root, reference, strategy, _, _ = _fixture(tmp_path)
    original = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (reference, strategy)}

    index = INDEX.build_raw_artifact_index(
        root,
        verify_content=True,
        verified_at="2026-07-16T01:02:03Z",
    )
    json_path, csv_path = INDEX.write_raw_artifact_index(index, root)

    assert index["schema_version"] == INDEX.SCHEMA_VERSION
    assert index["suite"] == "unit_raw_suite"
    assert index["artifact_count"] == 2
    assert index["total_bytes"] == reference.stat().st_size + strategy.stat().st_size
    assert {row["strategy"] for row in index["artifacts"]} == {
        INDEX.REFERENCE_STRATEGY,
        "Q+L",
    }
    assert all(row["content_sha256_verified"] is True for row in index["artifacts"])
    assert all(row["absolute_root"] == str(root.resolve()) for row in index["artifacts"])
    assert all(not Path(row["relative_path"]).is_absolute() for row in index["artifacts"])
    assert len(index["generation_sha256"]) == 64
    assert json.loads(json_path.read_text(encoding="utf-8"))["artifact_count"] == 2
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert {row["content_sha256_verified"] for row in csv_rows} == {"true"}
    assert {row["index_generation_sha256"] for row in csv_rows} == {
        index["generation_sha256"]
    }
    assert set(root.rglob("*.hrc")) == {reference, strategy}
    for path, (content, modified_ns) in original.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == modified_ns


def test_verify_content_detects_same_size_tampering(tmp_path: Path) -> None:
    root, _, strategy, _, _ = _fixture(tmp_path)
    strategy.write_bytes(b"tampered")

    metadata_index = INDEX.build_raw_artifact_index(root, verify_content=False)
    assert metadata_index["verification_mode"] == "metadata_and_size"
    assert all(row["content_sha256_verified"] is False for row in metadata_index["artifacts"])
    with pytest.raises(INDEX.ArtifactIndexError, match="SHA-256 differs"):
        INDEX.build_raw_artifact_index(root, verify_content=True)


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.hrc", "/absolute.hrc", "C:/drive.hrc", "artifacts\\x.hrc"],
)
def test_rejects_unsafe_artifact_paths(tmp_path: Path, unsafe_path: str) -> None:
    root, _, _, manifest, rows = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    manifest_strategy = manifest["strategies"][0]
    assert isinstance(manifest_strategy, dict)
    manifest_strategy["artifact_path"] = unsafe_path
    rows[0]["artifact_path"] = unsafe_path
    (job / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_payload_csv(job / "artifact_payloads.csv", rows)

    with pytest.raises(INDEX.ArtifactIndexError, match="relative path|escapes"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_manifest_csv_hash_disagreement(tmp_path: Path) -> None:
    root, _, _, _, rows = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    rows[0]["artifact_sha256"] = "f" * 64
    _write_payload_csv(job / "artifact_payloads.csv", rows)

    with pytest.raises(INDEX.ArtifactIndexError, match="manifest/CSV artifact evidence differs"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_duplicate_artifact_reference_and_unknown_job(tmp_path: Path) -> None:
    root, _, _, manifest, rows = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    reference = manifest["reference"]
    assert isinstance(reference, dict)
    strategy = manifest["strategies"][0]
    assert isinstance(strategy, dict)
    strategy["artifact_path"] = reference["path"]
    strategy["artifact_file_bytes"] = reference["file_bytes"]
    strategy["artifact_sha256"] = reference["sha256"]
    rows[0].update(strategy)
    (job / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_payload_csv(job / "artifact_payloads.csv", rows)

    with pytest.raises(INDEX.ArtifactIndexError, match="referenced more than once"):
        INDEX.build_raw_artifact_index(root)

    suite_manifest = json.loads((root / "suite_manifest.json").read_text(encoding="utf-8"))
    suite_manifest["jobs"] = []
    (root / "suite_manifest.json").write_text(json.dumps(suite_manifest), encoding="utf-8")
    with pytest.raises(INDEX.ArtifactIndexError, match="absent from suite_manifest"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_physical_byte_mismatch_and_invalid_source_commit(tmp_path: Path) -> None:
    root, _, strategy, _, _ = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    strategy.write_bytes(strategy.read_bytes() + b"!")
    with pytest.raises(INDEX.ArtifactIndexError, match="byte count differs"):
        INDEX.build_raw_artifact_index(root)

    _fixture(tmp_path)
    (job / "run_config.json").write_text(
        json.dumps({"git": {"commit": "not-a-commit"}}), encoding="utf-8"
    )
    with pytest.raises(INDEX.ArtifactIndexError, match="source commit"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_noncompleted_jobs_and_wrong_suite_schema(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    manifest_path = root / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(INDEX.ArtifactIndexError, match="not completed_valid"):
        INDEX.build_raw_artifact_index(root)

    manifest["jobs"][0].update({"status": "completed_valid", "exit_code": 1})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(INDEX.ArtifactIndexError, match="successful exit code"):
        INDEX.build_raw_artifact_index(root)

    manifest["schema_version"] = "wrong.v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(INDEX.ArtifactIndexError, match="schema differs"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_unbound_raw_artifact_and_invalid_calendar_timestamp(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    orphan = root / "jobs" / "model_seed17_rate0258" / "artifacts" / "orphan.hrc"
    orphan.write_bytes(b"unbound")
    with pytest.raises(INDEX.ArtifactIndexError, match="unbound .hrc"):
        INDEX.build_raw_artifact_index(root)

    orphan.unlink()
    with pytest.raises(INDEX.ArtifactIndexError, match="valid UTC calendar"):
        INDEX.build_raw_artifact_index(root, verified_at="2026-99-99T99:99:99Z")


def test_validates_both_index_views_before_replacing_either(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    index = INDEX.build_raw_artifact_index(
        root,
        verified_at="2026-07-16T01:02:03Z",
    )
    index["artifacts"][0]["unexpected"] = "field"

    with pytest.raises(INDEX.ArtifactIndexError, match="fields differ"):
        INDEX.write_raw_artifact_index(index, root)
    assert not (root / "raw_artifact_index.json").exists()
    assert not (root / "raw_artifact_index.csv").exists()


def test_rejects_missing_completed_job_artifact_closure(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    path = root / "suite_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["jobs"].append(
        {"job_id": "ghost_seed17_rate0258", "status": "completed_valid", "exit_code": 0}
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(INDEX.ArtifactIndexError, match="job closure differs.*ghost"):
        INDEX.build_raw_artifact_index(root)


@pytest.mark.parametrize("dirty", [True, None, "false"])
def test_rejects_dirty_or_unaudited_job_source(tmp_path: Path, dirty: object) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    commit = "0123456789abcdef0123456789abcdef01234567"
    (job / "run_config.json").write_text(
        json.dumps({"git": {"commit": commit, "dirty": dirty}}), encoding="utf-8"
    )
    with pytest.raises(INDEX.ArtifactIndexError, match="dirty or unaudited"):
        INDEX.build_raw_artifact_index(root)


@pytest.mark.parametrize("commit", ["a" * 7, "a" * 39, "a" * 41, "A" * 40])
def test_rejects_abbreviated_or_nonlowercase_job_commit(
    tmp_path: Path, commit: str
) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    (job / "run_config.json").write_text(
        json.dumps({"git": {"commit": commit, "dirty": False}}), encoding="utf-8"
    )
    with pytest.raises(INDEX.ArtifactIndexError, match="full lowercase source commit"):
        INDEX.build_raw_artifact_index(root)


def test_rejects_suite_job_commit_mismatch(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    job = root / "jobs" / "model_seed17_rate0258"
    (job / "run_config.json").write_text(
        json.dumps({"git": {"commit": "f" * 40, "dirty": False}}), encoding="utf-8"
    )
    with pytest.raises(INDEX.ArtifactIndexError, match="differs from suite source commit"):
        INDEX.build_raw_artifact_index(root)


def test_writer_rejects_stale_generation_and_boolean_type_pollution(tmp_path: Path) -> None:
    root, _, _, _, _ = _fixture(tmp_path)
    index = INDEX.build_raw_artifact_index(root, verify_content=True)
    index["artifacts"][0]["bytes"] += 1
    with pytest.raises(INDEX.ArtifactIndexError, match="total_bytes is inconsistent"):
        INDEX.write_raw_artifact_index(index, root)

    index = INDEX.build_raw_artifact_index(root, verify_content=True)
    index["artifacts"][0]["content_sha256_verified"] = "false"
    with pytest.raises(INDEX.ArtifactIndexError, match="verification flag is invalid"):
        INDEX.write_raw_artifact_index(index, root)

    index = INDEX.build_raw_artifact_index(root, verify_content=True)
    index["artifacts"][0]["sha256"] = "f" * 64
    with pytest.raises(INDEX.ArtifactIndexError, match="generation SHA-256 is stale"):
        INDEX.write_raw_artifact_index(index, root)


def test_cli_defaults_to_content_verification() -> None:
    assert INDEX._parser().parse_args(["suite"]).verify_content is True
    assert INDEX._parser().parse_args(["suite", "--metadata-only"]).verify_content is False
