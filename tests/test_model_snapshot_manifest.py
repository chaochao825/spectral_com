from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_model_snapshot_manifest",
    REPO_ROOT / "scripts" / "build_model_snapshot_manifest.py",
)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MANIFEST
SPEC.loader.exec_module(MANIFEST)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_manifest_selects_reproducibility_files_and_has_stable_aggregate(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "config.json", b'{"model_type":"tiny"}\n')
    _write(model_dir / "weights" / "model-00001-of-00002.safetensors", b"weights-a")
    _write(model_dir / "weights" / "model-00002-of-00002.safetensors", b"weights-b")
    _write(model_dir / "model.safetensors.index.json", b"{}\n")
    _write(model_dir / "tokenizer.json", b'{"version":"1"}\n')
    _write(model_dir / "modeling_tiny.py", b"class Tiny: pass\n")
    _write(model_dir / "helper.py", b"VALUE = 1\n")
    _write(model_dir / "README.md", b"not part of the checkpoint identity\n")
    first_output = tmp_path / "manifests" / "first.json"
    second_output = tmp_path / "manifests" / "second.json"

    first = MANIFEST.build_manifest(
        model_dir,
        first_output,
        "tiny-a",
        generated_at="2026-07-16T00:00:00Z",
    )
    second = MANIFEST.build_manifest(
        model_dir,
        second_output,
        "renamed-label",
        generated_at="2026-07-17T00:00:00Z",
    )

    assert first["schema_version"] == "model_snapshot_manifest.v1"
    assert first["file_count"] == 7
    assert first["total_bytes"] == sum(record["bytes"] for record in first["files"])
    assert len(first["aggregate_sha256"]) == 64
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    assert first["generated_at"].endswith("Z")
    assert [record["path"] for record in first["files"]] == sorted(
        record["path"] for record in first["files"]
    )
    assert "README.md" not in {record["path"] for record in first["files"]}
    assert next(record for record in first["files"] if record["path"] == "helper.py")[
        "role"
    ] == "model_code"
    config = next(record for record in first["files"] if record["path"] == "config.json")
    assert config == {
        "path": "config.json",
        "role": "config",
        "bytes": len(b'{"model_type":"tiny"}\n'),
        "sha256": hashlib.sha256(b'{"model_type":"tiny"}\n').hexdigest(),
        "is_symlink": False,
        "symlink_target": None,
    }

    MANIFEST.write_manifest(first, first_output)
    persisted = json.loads(first_output.read_text(encoding="utf-8"))
    assert persisted == first

    _write(model_dir / "tokenizer.json", b'{"version":"2"}\n')
    changed = MANIFEST.build_manifest(model_dir, tmp_path / "changed.json", "tiny-a")
    assert changed["aggregate_sha256"] != first["aggregate_sha256"]


@pytest.mark.skipif(
    os.name == "nt",
    reason="unprivileged symlink creation is not portable on Windows",
)
def test_manifest_records_file_symlink(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    blob = model_dir / "blobs" / "weight-blob"
    _write(blob, b"linked weights")
    (model_dir / "model.safetensors").symlink_to(Path("blobs") / blob.name)
    result = MANIFEST.build_manifest(model_dir, tmp_path / "manifest.json", "linked")

    assert result["file_count"] == 1
    assert result["files"][0]["is_symlink"] is True
    assert result["files"][0]["symlink_target"] == "blobs/weight-blob"
    assert result["files"][0]["bytes"] == len(b"linked weights")


@pytest.mark.skipif(
    os.name == "nt",
    reason="unprivileged symlink creation is not portable on Windows",
)
def test_manifest_rejects_directory_symlink_instead_of_hiding_content(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "config.json", b"{}")
    external = tmp_path / "external"
    _write(external / "model.safetensors", b"outside weights")
    (model_dir / "linked-weights").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="directory symlink.*hide snapshot content"):
        MANIFEST.build_manifest(model_dir, tmp_path / "manifest.json", "linked-dir")


@pytest.mark.skipif(
    os.name == "nt",
    reason="unprivileged symlink creation is not portable on Windows",
)
def test_manifest_rejects_output_aliased_by_a_selected_symlink(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    output = tmp_path / "manifest.json"
    output.write_text("old manifest", encoding="utf-8")
    (model_dir / "config.json").symlink_to(output)

    with pytest.raises(ValueError, match="target of a selected model-tree symlink"):
        MANIFEST.build_manifest(model_dir, output, "self-link")
    assert output.read_text(encoding="utf-8") == "old manifest"


@pytest.mark.skipif(
    os.name == "nt",
    reason="unprivileged symlink creation is not portable on Windows",
)
def test_manifest_rejects_broken_selected_symlink_cleanly(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").symlink_to("missing-blob")

    with pytest.raises(ValueError, match="selected path is a broken symlink"):
        MANIFEST.build_manifest(model_dir, tmp_path / "manifest.json", "broken")


def test_manifest_rejects_empty_unmatched_and_self_referential_output(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no files"):
        MANIFEST.build_manifest(empty, tmp_path / "empty.json", "empty")

    unmatched = tmp_path / "unmatched"
    _write(unmatched / "README.md", b"documentation only")
    with pytest.raises(ValueError, match="no matching checkpoint"):
        MANIFEST.build_manifest(unmatched, tmp_path / "unmatched.json", "unmatched")

    model_dir = tmp_path / "model"
    _write(model_dir / "config.json", b"{}")
    with pytest.raises(ValueError, match="outside the model directory"):
        MANIFEST.build_manifest(model_dir, model_dir / "snapshot.json", "self")
    assert not (model_dir / "snapshot.json").exists()

    config_only = tmp_path / "config-only"
    _write(config_only / "config.json", b"{}")
    with pytest.raises(ValueError, match="no selected checkpoint"):
        MANIFEST.build_manifest(config_only, tmp_path / "config-only.json", "config-only")


def test_write_rejects_destination_other_than_the_validated_output(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "pytorch_model.bin", b"weights")
    output = tmp_path / "snapshot.json"
    result = MANIFEST.build_manifest(model_dir, output, "tiny")

    with pytest.raises(ValueError, match="does not match"):
        MANIFEST.write_manifest(result, tmp_path / "different.json")
    assert not (tmp_path / "different.json").exists()


def test_write_rejects_stale_aggregate_and_derived_fields(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "pytorch_model.bin", b"weights")
    output = tmp_path / "snapshot.json"
    result = MANIFEST.build_manifest(model_dir, output, "tiny")

    result["files"][0]["bytes"] += 1
    with pytest.raises(ValueError, match="total_bytes is inconsistent"):
        MANIFEST.write_manifest(result, output)
    assert not output.exists()


def test_write_rejects_non_boolean_symlink_flag(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "pytorch_model.bin", b"weights")
    output = tmp_path / "snapshot.json"
    result = MANIFEST.build_manifest(model_dir, output, "tiny")
    result["files"][0]["is_symlink"] = "false"

    with pytest.raises(ValueError, match="symlink flag is invalid"):
        MANIFEST.write_manifest(result, output)


def test_manifest_rejects_a_file_that_changes_while_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write(model_dir / "pytorch_model.bin", b"weights")
    real_fstat = MANIFEST.os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls != 2:
            return observed
        values = {
            name: getattr(observed, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
                "st_mode",
            )
        }
        values["st_mtime_ns"] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(MANIFEST.os, "fstat", changing_fstat)
    with pytest.raises(ValueError, match="changed while hashing"):
        MANIFEST.build_manifest(model_dir, tmp_path / "manifest.json", "mutating")


def test_verify_manifest_rehashes_the_current_model_tree_and_binds_both_hashes(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    weights = model_dir / "model.safetensors"
    _write(weights, b"frozen weights")
    _write(model_dir / "config.json", b'{"model_type":"tiny"}\n')
    output = tmp_path / "snapshot.json"
    manifest = MANIFEST.build_manifest(
        model_dir,
        output,
        "tiny",
        generated_at="2026-07-16T00:00:00Z",
    )
    MANIFEST.write_manifest(manifest, output)
    manifest_sha = hashlib.sha256(output.read_bytes()).hexdigest()

    evidence = MANIFEST.verify_model_snapshot_manifest(
        output,
        model_dir,
        expected_manifest_sha256=manifest_sha,
        expected_aggregate_sha256=manifest["aggregate_sha256"],
    )
    assert evidence["verified_current_tree"] is True
    assert evidence["manifest_sha256"] == manifest_sha
    assert evidence["aggregate_sha256"] == manifest["aggregate_sha256"]
    assert evidence["model_dir"] == str(model_dir.resolve())

    with pytest.raises(ValueError, match="manifest file SHA-256 differs"):
        MANIFEST.verify_model_snapshot_manifest(
            output,
            model_dir,
            expected_manifest_sha256="0" * 64,
            expected_aggregate_sha256=manifest["aggregate_sha256"],
        )

    weights.write_bytes(b"mutated weights")
    with pytest.raises(ValueError, match="current tree differs"):
        MANIFEST.verify_model_snapshot_manifest(
            output,
            model_dir,
            expected_manifest_sha256=manifest_sha,
            expected_aggregate_sha256=manifest["aggregate_sha256"],
        )


def test_committed_model_snapshot_raw_hashes_are_lf_pinned_and_match_v3_config() -> None:
    config = json.loads(
        (REPO_ROOT / "configs" / "large_model_global_controls_v3_20260716.json").read_text(
            encoding="utf-8"
        )
    )
    contracts = {
        stage["model_snapshot_manifest"]: stage["model_snapshot_manifest_sha256"]
        for stage in config["stages"]
    }
    assert len(contracts) == 3
    for relative, expected_sha in contracts.items():
        path = REPO_ROOT / relative
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw).hexdigest() == expected_sha
        completed = subprocess.run(
            ["git", "check-attr", "eol", "--", relative],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.rstrip().endswith("eol: lf")
