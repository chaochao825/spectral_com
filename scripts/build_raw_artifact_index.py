#!/usr/bin/env python3
"""Build a small, verifiable index for suite-local raw ``.hrc`` artifacts.

The index contains paths and cryptographic metadata only.  It never copies,
rewrites, or opens an artifact for writing.  The command-line interface
recomputes every artifact SHA-256 by default.  ``--metadata-only`` is an
explicitly weaker diagnostic mode that checks metadata and file sizes only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "raw_artifact_index.v1"
SUITE_MANIFEST_SCHEMA_VERSION = "large_scale_hessian_suite_manifest.v1"
REFERENCE_STRATEGY = "__fp16_reference__"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SAFE_SUITE_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
UTC_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
ARTIFACT_ROW_COLUMNS = (
    "suite",
    "job",
    "strategy",
    "role",
    "relative_path",
    "absolute_root",
    "bytes",
    "sha256",
    "source_commit",
    "verified_at",
    "content_sha256_verified",
)
CSV_COLUMNS = (
    *ARTIFACT_ROW_COLUMNS,
    "index_generation_sha256",
)
GENERATION_SCOPE = (
    "SHA-256 of the canonical index before generation fields; the same "
    "digest is repeated in every CSV row to detect mixed JSON/CSV views"
)
INDEX_FIELDS = {
    "schema_version",
    "suite",
    "absolute_root",
    "verification_mode",
    "verified_at",
    "artifact_count",
    "total_bytes",
    "source_commits",
    "artifacts",
    "generation_scope",
    "generation_sha256",
}


class ArtifactIndexError(RuntimeError):
    """Raised when raw artifact evidence is absent, unsafe, or inconsistent."""


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _stable_regular_bytes(path: Path, context: str) -> bytes:
    """Read one non-symlink regular file through a stable descriptor."""

    try:
        link_before = path.lstat()
        if stat.S_ISLNK(link_before.st_mode):
            raise ArtifactIndexError(f"{context} must not be a symlink: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except ArtifactIndexError:
        raise
    except OSError as exc:
        raise ArtifactIndexError(f"cannot open {context}: {path}: {exc}") from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ArtifactIndexError(f"{context} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        link_after = path.lstat()
    except OSError as exc:
        raise ArtifactIndexError(f"{context} changed after reading: {path}") from exc
    identity = _stat_identity(descriptor_before)
    if (
        identity != _stat_identity(descriptor_after)
        or identity != _stat_identity(link_before)
        or identity != _stat_identity(link_after)
    ):
        raise ArtifactIndexError(f"{context} changed while reading: {path}")
    payload = b"".join(chunks)
    if len(payload) != descriptor_after.st_size:
        raise ArtifactIndexError(f"{context} size changed while reading: {path}")
    return payload


def _stable_artifact_measure(
    path: Path, *, verify_content: bool
) -> tuple[int, str | None, tuple[int, int]]:
    """Measure/hash an artifact using lstat + O_NOFOLLOW + fstat checks."""

    try:
        link_before = path.lstat()
        if stat.S_ISLNK(link_before.st_mode):
            raise ArtifactIndexError(f"raw artifact must not be a symlink: {path}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except ArtifactIndexError:
        raise
    except OSError as exc:
        raise ArtifactIndexError(f"cannot open raw artifact: {path}: {exc}") from exc
    digest = hashlib.sha256() if verify_content else None
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise ArtifactIndexError(f"raw artifact is not a regular file: {path}")
        if digest is not None:
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        link_after = path.lstat()
    except OSError as exc:
        raise ArtifactIndexError(f"raw artifact changed after inspection: {path}") from exc
    identity = _stat_identity(descriptor_before)
    if (
        identity != _stat_identity(descriptor_after)
        or identity != _stat_identity(link_before)
        or identity != _stat_identity(link_after)
    ):
        raise ArtifactIndexError(f"artifact changed while hashing: {path}")
    inode = (int(descriptor_after.st_dev), int(descriptor_after.st_ino))
    return int(descriptor_after.st_size), None if digest is None else digest.hexdigest(), inode


def _iter_files_without_directory_symlinks(root: Path) -> Iterable[Path]:
    """Yield a closed output tree and reject directories hidden by links."""

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ArtifactIndexError(
                    f"suite jobs tree contains a directory symlink: {directory_path / name}"
                )
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            yield directory_path / name


def _tree_identity_snapshot(root: Path) -> tuple[list[Path], dict[str, tuple[Any, ...]]]:
    """Capture every jobs-tree path identity to require a quiescent suite."""

    files = list(_iter_files_without_directory_symlinks(root))
    snapshot: dict[str, tuple[Any, ...]] = {}
    for path in files:
        try:
            link = path.lstat()
            is_symlink = stat.S_ISLNK(link.st_mode)
            target = path.stat()
            link_target = os.readlink(path) if is_symlink else None
        except OSError as exc:
            raise ArtifactIndexError(f"jobs-tree path is unreadable: {path}: {exc}") from exc
        snapshot[path.relative_to(root).as_posix()] = (
            _stat_identity(link),
            _stat_identity(target),
            is_symlink,
            link_target,
        )
    return files, snapshot


def _canonical_utc_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not isinstance(value, str) or UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ArtifactIndexError("verified_at must be a second-resolution UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ArtifactIndexError("verified_at is not a valid UTC calendar timestamp") from exc
    canonical = parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if canonical != value:
        raise ArtifactIndexError("verified_at must use canonical second-resolution UTC form")
    return canonical


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactIndexError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        text = _stable_regular_bytes(path, context).decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except ArtifactIndexError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactIndexError(f"cannot read {context}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactIndexError(f"{context} must contain a JSON object: {path}")
    return payload


def _require_within(path: Path, root: Path, context: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactIndexError(f"{context} does not exist: {path}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactIndexError(f"{context} escapes its allowed root: {path}") from exc
    return resolved


def _require_regular_evidence_file(path: Path, root: Path, context: str) -> Path:
    resolved = _require_within(path, root, context)
    if path.is_symlink() or not resolved.is_file():
        raise ArtifactIndexError(f"{context} must be a regular, non-symlink file: {path}")
    return resolved


def _strict_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise ArtifactIndexError(f"{context} must be a positive integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        number = int(value)
    else:
        raise ArtifactIndexError(f"{context} must be a positive integer")
    if number <= 0:
        raise ArtifactIndexError(f"{context} must be a positive integer")
    return number


def _strict_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ArtifactIndexError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _strict_strategy(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ArtifactIndexError(f"{context} is not a valid non-empty strategy name")
    if value == REFERENCE_STRATEGY:
        raise ArtifactIndexError(f"{context} uses the reserved reference strategy name")
    return value


def _artifact_path(job_dir: Path, jobs_root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ArtifactIndexError(f"{context} must be a non-empty POSIX relative path")
    relative = PurePosixPath(value)
    windows_relative = PureWindowsPath(value)
    if (
        relative.is_absolute()
        or bool(windows_relative.drive)
        or relative.as_posix() != value
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise ArtifactIndexError(f"{context} must be a normalized relative path: {value!r}")
    lexical = job_dir.joinpath(*relative.parts)
    cursor = job_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ArtifactIndexError(f"{context} traverses a symlink: {value!r}")
    resolved = _require_within(lexical, job_dir, context)
    try:
        resolved.relative_to(jobs_root)
    except ValueError as exc:
        raise ArtifactIndexError(f"{context} escapes the suite jobs root: {value!r}") from exc
    if lexical.is_symlink() or not resolved.is_file():
        raise ArtifactIndexError(f"{context} must identify a regular, non-symlink file")
    if resolved.suffix != ".hrc":
        raise ArtifactIndexError(f"{context} is not an .hrc artifact: {value!r}")
    return resolved


def _read_payload_rows(path: Path, job_id: str) -> dict[str, dict[str, str]]:
    try:
        payload = _stable_regular_bytes(
            path, f"artifact_payloads.csv for {job_id}"
        ).decode("utf-8")
        with io.StringIO(payload, newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None or len(fieldnames) != len(set(fieldnames)):
                raise ArtifactIndexError(f"artifact_payloads.csv has absent/duplicate columns: {job_id}")
            required = {
                "strategy",
                "artifact_path",
                "artifact_file_bytes",
                "artifact_sha256",
            }
            if not required.issubset(fieldnames):
                missing = sorted(required.difference(fieldnames))
                raise ArtifactIndexError(
                    f"artifact_payloads.csv lacks required columns for {job_id}: {missing}"
                )
            rows: dict[str, dict[str, str]] = {}
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ArtifactIndexError(
                        f"artifact_payloads.csv has extra fields at {job_id}:{line_number}"
                    )
                strategy = _strict_strategy(
                    row.get("strategy"), f"artifact_payloads.csv strategy at {job_id}:{line_number}"
                )
                if strategy in rows:
                    raise ArtifactIndexError(
                        f"duplicate artifact_payloads.csv strategy for {job_id}: {strategy}"
                    )
                rows[strategy] = row
    except ArtifactIndexError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ArtifactIndexError(f"cannot read artifact_payloads.csv for {job_id}: {exc}") from exc
    if not rows:
        raise ArtifactIndexError(f"artifact_payloads.csv is empty for {job_id}")
    return rows


def _source_commit(job_dir: Path, jobs_root: Path, job_id: str) -> str:
    path = job_dir / "run_config.json"
    _require_regular_evidence_file(path, jobs_root, f"run_config.json for {job_id}")
    payload = _read_json_object(path, f"run_config.json for {job_id}")
    git = payload.get("git")
    commit = git.get("commit") if isinstance(git, dict) else None
    if not isinstance(commit, str) or GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ArtifactIndexError(f"run_config.json has no full lowercase source commit for {job_id}")
    if git.get("dirty") is not False:
        raise ArtifactIndexError(f"run_config.json source tree is dirty or unaudited for {job_id}")
    return commit


def _suite_identity(
    root: Path,
) -> tuple[str, str, dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = root / "suite_manifest.json"
    _require_regular_evidence_file(manifest_path, root, "suite_manifest.json")
    manifest = _read_json_object(manifest_path, "suite_manifest.json")
    if manifest.get("schema_version") != SUITE_MANIFEST_SCHEMA_VERSION:
        raise ArtifactIndexError(
            "suite_manifest.json schema differs from "
            f"{SUITE_MANIFEST_SCHEMA_VERSION!r}"
        )
    suite_id = manifest.get("suite_id")
    if not isinstance(suite_id, str) or SAFE_SUITE_ID_RE.fullmatch(suite_id) is None:
        raise ArtifactIndexError("suite_manifest.json has no safe suite_id")
    git = manifest.get("git")
    suite_commit = git.get("commit") if isinstance(git, dict) else None
    if not isinstance(suite_commit, str) or GIT_COMMIT_RE.fullmatch(suite_commit) is None:
        raise ArtifactIndexError("suite_manifest.json has no full lowercase source commit")
    if git.get("dirty") is not False:
        raise ArtifactIndexError("suite_manifest.json source tree is dirty or unaudited")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ArtifactIndexError("suite_manifest.json jobs must be a list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in jobs:
        job_id = entry.get("job_id") if isinstance(entry, dict) else None
        if (
            not isinstance(job_id, str)
            or SAFE_SUITE_ID_RE.fullmatch(job_id) is None
            or job_id in entries
        ):
            raise ArtifactIndexError("suite_manifest.json has invalid or duplicate job ids")
        assert isinstance(entry, dict)
        entries[job_id] = entry
    return suite_id, suite_commit, manifest, entries


def _manifest_artifacts(
    manifest: Mapping[str, Any],
    payload_rows: Mapping[str, Mapping[str, str]],
    job_id: str,
) -> list[tuple[str, str, str, int, str]]:
    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise ArtifactIndexError(f"artifact manifest lacks reference evidence for {job_id}")
    records: list[tuple[str, str, str, int, str]] = [
        (
            REFERENCE_STRATEGY,
            "reference",
            str(reference.get("path")) if isinstance(reference.get("path"), str) else "",
            _strict_positive_integer(reference.get("file_bytes"), f"reference bytes for {job_id}"),
            _strict_sha256(reference.get("sha256"), f"reference sha256 for {job_id}"),
        )
    ]
    raw_strategies = manifest.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        raise ArtifactIndexError(f"artifact manifest has no strategies for {job_id}")
    manifest_names: set[str] = set()
    for index, raw in enumerate(raw_strategies):
        if not isinstance(raw, dict):
            raise ArtifactIndexError(f"artifact manifest strategy {index} is invalid for {job_id}")
        strategy = _strict_strategy(raw.get("strategy"), f"manifest strategy {index} for {job_id}")
        if strategy in manifest_names:
            raise ArtifactIndexError(f"duplicate manifest strategy for {job_id}: {strategy}")
        manifest_names.add(strategy)
        path = raw.get("artifact_path")
        size = _strict_positive_integer(
            raw.get("artifact_file_bytes"), f"manifest bytes for {job_id}/{strategy}"
        )
        sha256 = _strict_sha256(
            raw.get("artifact_sha256"), f"manifest sha256 for {job_id}/{strategy}"
        )
        row = payload_rows.get(strategy)
        if row is None:
            raise ArtifactIndexError(f"artifact_payloads.csv lacks {job_id}/{strategy}")
        csv_size = _strict_positive_integer(
            row.get("artifact_file_bytes"), f"CSV bytes for {job_id}/{strategy}"
        )
        csv_sha256 = _strict_sha256(
            row.get("artifact_sha256"), f"CSV sha256 for {job_id}/{strategy}"
        )
        if row.get("artifact_path") != path or csv_size != size or csv_sha256 != sha256:
            raise ArtifactIndexError(
                f"manifest/CSV artifact evidence differs for {job_id}/{strategy}"
            )
        records.append((strategy, "strategy", path if isinstance(path, str) else "", size, sha256))
    if manifest_names != set(payload_rows):
        extras = sorted(set(payload_rows).difference(manifest_names))
        raise ArtifactIndexError(f"artifact_payloads.csv has unbound strategies for {job_id}: {extras}")
    return records


def build_raw_artifact_index(
    output_root: Path | str,
    *,
    verify_content: bool = False,
    verified_at: str | None = None,
) -> dict[str, Any]:
    """Validate a suite's raw artifacts and return an in-memory index."""

    try:
        root = Path(output_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ArtifactIndexError(f"suite output root does not exist: {output_root}: {exc}") from exc
    if not root.is_dir():
        raise ArtifactIndexError(f"suite output root is not a directory: {root}")
    suite_manifest_path = root / "suite_manifest.json"
    suite_manifest_digest_before = hashlib.sha256(
        _stable_regular_bytes(suite_manifest_path, "suite_manifest.json")
    ).hexdigest()
    suite_id, suite_commit, suite_manifest, suite_jobs = _suite_identity(root)
    jobs_root = _require_within(root / "jobs", root, "suite jobs directory")
    if not jobs_root.is_dir():
        raise ArtifactIndexError(f"suite jobs path is not a directory: {jobs_root}")

    tree_files, tree_identity_before = _tree_identity_snapshot(jobs_root)
    manifests = sorted(path for path in tree_files if path.name == "artifact_manifest.json")
    payloads = sorted(path for path in tree_files if path.name == "artifact_payloads.csv")
    discovered_artifacts: set[Path] = set()
    for path in sorted(path for path in tree_files if path.suffix == ".hrc"):
        if path.is_symlink():
            raise ArtifactIndexError(f"raw artifact must not be a symlink: {path}")
        resolved = _require_within(path, jobs_root, "raw .hrc artifact")
        if not resolved.is_file():
            raise ArtifactIndexError(f"raw artifact must be a regular file: {path}")
        if resolved in discovered_artifacts:
            raise ArtifactIndexError(f"raw artifact is reachable by more than one path: {path}")
        discovered_artifacts.add(resolved)
    manifest_parents = {path.parent.resolve() for path in manifests}
    payload_parents = {path.parent.resolve() for path in payloads}
    if not manifest_parents:
        raise ArtifactIndexError(f"no artifact manifests found below {jobs_root}")
    if manifest_parents != payload_parents:
        raise ArtifactIndexError("artifact manifest/CSV job directories differ")

    known_job_ids = set(suite_jobs)
    expected_completed_jobs = {
        job_id
        for job_id, entry in suite_jobs.items()
        if entry.get("status") == "completed_valid" and entry.get("exit_code") == 0
    }
    observed_job_ids = {
        path.parent.resolve().relative_to(jobs_root).as_posix() for path in manifests
    }
    for job_id in sorted(observed_job_ids):
        if job_id not in suite_jobs:
            raise ArtifactIndexError(
                f"artifact directory is absent from suite_manifest.json: {job_id}"
            )
        if suite_jobs[job_id].get("status") != "completed_valid":
            raise ArtifactIndexError(
                f"artifact job is not completed_valid in suite_manifest.json: {job_id}"
            )
        if suite_jobs[job_id].get("exit_code") != 0:
            raise ArtifactIndexError(
                f"artifact job has no successful exit code in suite_manifest.json: {job_id}"
            )
    if observed_job_ids != expected_completed_jobs:
        missing = sorted(expected_completed_jobs.difference(observed_job_ids))
        unexpected = sorted(observed_job_ids.difference(expected_completed_jobs))
        raise ArtifactIndexError(
            "artifact job closure differs from completed_valid suite jobs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    timestamp = _canonical_utc_timestamp(verified_at)
    rows: list[dict[str, Any]] = []
    seen_resolved_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for manifest_path in manifests:
        job_dir = manifest_path.parent.resolve()
        _require_regular_evidence_file(manifest_path, jobs_root, "artifact_manifest.json")
        payload_path = job_dir / "artifact_payloads.csv"
        _require_regular_evidence_file(payload_path, jobs_root, "artifact_payloads.csv")
        job_id = job_dir.relative_to(jobs_root).as_posix()
        if job_id not in known_job_ids:
            raise ArtifactIndexError(f"artifact directory is absent from suite_manifest.json: {job_id}")
        if suite_jobs[job_id].get("status") != "completed_valid":
            raise ArtifactIndexError(
                f"artifact job is not completed_valid in suite_manifest.json: {job_id}"
            )
        if suite_jobs[job_id].get("exit_code") != 0:
            raise ArtifactIndexError(
                f"artifact job has no successful exit code in suite_manifest.json: {job_id}"
            )
        source_commit = _source_commit(job_dir, jobs_root, job_id)
        if source_commit != suite_commit:
            raise ArtifactIndexError(
                f"job source commit differs from suite source commit for {job_id}"
            )
        manifest = _read_json_object(manifest_path, f"artifact_manifest.json for {job_id}")
        payload_rows = _read_payload_rows(payload_path, job_id)
        for strategy, role, raw_path, expected_size, expected_sha256 in _manifest_artifacts(
            manifest, payload_rows, job_id
        ):
            artifact = _artifact_path(
                job_dir,
                jobs_root,
                raw_path,
                f"artifact path for {job_id}/{strategy}",
            )
            if artifact in seen_resolved_paths:
                raise ArtifactIndexError(f"artifact path is referenced more than once: {artifact}")
            seen_resolved_paths.add(artifact)
            actual_size, actual_sha256, inode = _stable_artifact_measure(
                artifact, verify_content=verify_content
            )
            if inode in seen_inodes:
                raise ArtifactIndexError(
                    f"raw artifact is a hardlink alias of another indexed file: {artifact}"
                )
            seen_inodes.add(inode)
            if actual_size != expected_size:
                raise ArtifactIndexError(
                    f"artifact byte count differs for {job_id}/{strategy}: "
                    f"expected {expected_size}, found {actual_size}"
                )
            if verify_content:
                if actual_sha256 != expected_sha256:
                    raise ArtifactIndexError(f"artifact SHA-256 differs for {job_id}/{strategy}")
            rows.append(
                {
                    "suite": suite_id,
                    "job": job_id,
                    "strategy": strategy,
                    "role": role,
                    "relative_path": artifact.relative_to(root).as_posix(),
                    "absolute_root": str(root),
                    "bytes": actual_size,
                    "sha256": expected_sha256,
                    "source_commit": source_commit,
                    "verified_at": timestamp,
                    "content_sha256_verified": bool(verify_content),
                }
            )
    missing_from_index = sorted(discovered_artifacts.difference(seen_resolved_paths))
    missing_from_tree = sorted(seen_resolved_paths.difference(discovered_artifacts))
    if missing_from_index:
        rendered = [path.relative_to(root).as_posix() for path in missing_from_index]
        raise ArtifactIndexError(f"unbound .hrc artifacts exist below jobs/: {rendered}")
    if missing_from_tree:
        rendered = [path.relative_to(root).as_posix() for path in missing_from_tree]
        raise ArtifactIndexError(f"manifest-bound artifacts escaped tree discovery: {rendered}")
    _final_tree_files, tree_identity_after = _tree_identity_snapshot(jobs_root)
    if tree_identity_after != tree_identity_before:
        raise ArtifactIndexError("suite jobs tree changed while building the raw artifact index")
    suite_manifest_digest_after = hashlib.sha256(
        _stable_regular_bytes(suite_manifest_path, "suite_manifest.json")
    ).hexdigest()
    if suite_manifest_digest_after != suite_manifest_digest_before:
        raise ArtifactIndexError("suite_manifest.json changed while building the raw artifact index")
    rows.sort(key=lambda row: (str(row["job"]), str(row["role"]), str(row["strategy"])))
    result = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite_id,
        "absolute_root": str(root),
        "verification_mode": "content_sha256" if verify_content else "metadata_and_size",
        "verified_at": timestamp,
        "artifact_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "source_commits": [suite_commit],
        "artifacts": rows,
    }
    generation_payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    result["generation_scope"] = GENERATION_SCOPE
    result["generation_sha256"] = hashlib.sha256(generation_payload).hexdigest()
    return result


def _canonical_generation_sha256(index: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in index.items()
        if key not in {"generation_scope", "generation_sha256"}
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_canonical_index(
    index: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze and fully self-check both index views before publication."""

    try:
        frozen = json.loads(
            json.dumps(index, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactIndexError(f"artifact index is not canonical JSON data: {exc}") from exc
    if not isinstance(frozen, dict) or set(frozen) != INDEX_FIELDS:
        fields = set(frozen) if isinstance(frozen, dict) else set()
        raise ArtifactIndexError(
            "artifact index top-level fields differ from schema: "
            f"missing={sorted(INDEX_FIELDS - fields)}, unexpected={sorted(fields - INDEX_FIELDS)}"
        )
    if frozen.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactIndexError("artifact index schema_version is invalid")
    suite = frozen.get("suite")
    if not isinstance(suite, str) or SAFE_SUITE_ID_RE.fullmatch(suite) is None:
        raise ArtifactIndexError("artifact index suite id is invalid")
    if frozen.get("absolute_root") != str(root):
        raise ArtifactIndexError("index absolute_root differs from the requested output root")
    mode = frozen.get("verification_mode")
    if mode not in {"content_sha256", "metadata_and_size"}:
        raise ArtifactIndexError("artifact index verification_mode is invalid")
    timestamp = frozen.get("verified_at")
    if not isinstance(timestamp, str) or _canonical_utc_timestamp(timestamp) != timestamp:
        raise ArtifactIndexError("artifact index verified_at is invalid")
    if frozen.get("generation_scope") != GENERATION_SCOPE:
        raise ArtifactIndexError("artifact index generation_scope is invalid")
    raw_rows = frozen.get("artifacts")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ArtifactIndexError("cannot write an empty or invalid artifact index")

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    reference_counts: dict[str, int] = {}
    for index_number, raw in enumerate(raw_rows):
        if not isinstance(raw, dict) or set(raw) != set(ARTIFACT_ROW_COLUMNS):
            fields = set(raw) if isinstance(raw, dict) else set()
            raise ArtifactIndexError(
                "artifact index row fields differ from the CSV schema: "
                f"row={index_number}, missing={sorted(set(ARTIFACT_ROW_COLUMNS) - fields)}, "
                f"unexpected={sorted(fields - set(ARTIFACT_ROW_COLUMNS))}"
            )
        if raw.get("suite") != suite or raw.get("absolute_root") != str(root):
            raise ArtifactIndexError(f"artifact index row {index_number} suite/root is inconsistent")
        job = raw.get("job")
        if not isinstance(job, str) or SAFE_SUITE_ID_RE.fullmatch(job) is None:
            raise ArtifactIndexError(f"artifact index row {index_number} job is invalid")
        role = raw.get("role")
        strategy = raw.get("strategy")
        if role == "reference":
            if strategy != REFERENCE_STRATEGY:
                raise ArtifactIndexError(f"artifact index row {index_number} reference is invalid")
            reference_counts[job] = reference_counts.get(job, 0) + 1
        elif role == "strategy":
            _strict_strategy(strategy, f"artifact index row {index_number} strategy")
        else:
            raise ArtifactIndexError(f"artifact index row {index_number} role is invalid")
        key = (job, str(strategy))
        if key in seen_keys:
            raise ArtifactIndexError(f"artifact index contains duplicate job/strategy: {key}")
        seen_keys.add(key)
        relative_value = raw.get("relative_path")
        if not isinstance(relative_value, str) or "\\" in relative_value:
            raise ArtifactIndexError(f"artifact index row {index_number} path is invalid")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or PureWindowsPath(relative_value).drive
            or relative.as_posix() != relative_value
            or any(part in ("", ".", "..") for part in relative.parts)
            or len(relative.parts) < 4
            or relative.parts[0] != "jobs"
            or relative.parts[1] != job
            or relative.suffix != ".hrc"
            or relative_value in seen_paths
        ):
            raise ArtifactIndexError(f"artifact index row {index_number} path is invalid")
        seen_paths.add(relative_value)
        byte_count = raw.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ArtifactIndexError(f"artifact index row {index_number} bytes is invalid")
        _strict_sha256(raw.get("sha256"), f"artifact index row {index_number} SHA-256")
        commit = raw.get("source_commit")
        if not isinstance(commit, str) or GIT_COMMIT_RE.fullmatch(commit) is None:
            raise ArtifactIndexError(f"artifact index row {index_number} commit is invalid")
        if raw.get("verified_at") != timestamp:
            raise ArtifactIndexError(f"artifact index row {index_number} timestamp is inconsistent")
        verified = raw.get("content_sha256_verified")
        if type(verified) is not bool:
            raise ArtifactIndexError(
                f"artifact index row {index_number} content verification flag is invalid"
            )
        if verified != (mode == "content_sha256"):
            raise ArtifactIndexError(
                f"artifact index row {index_number} verification flag differs from mode"
            )
        rows.append(dict(raw))

    expected_order = sorted(
        rows, key=lambda row: (str(row["job"]), str(row["role"]), str(row["strategy"]))
    )
    if rows != expected_order:
        raise ArtifactIndexError("artifact index rows are not in canonical order")
    if any(count != 1 for count in reference_counts.values()) or set(reference_counts) != {
        str(row["job"]) for row in rows
    }:
        raise ArtifactIndexError("each indexed job must have exactly one reference artifact")
    if frozen.get("artifact_count") != len(rows):
        raise ArtifactIndexError("artifact index artifact_count is inconsistent")
    if frozen.get("total_bytes") != sum(int(row["bytes"]) for row in rows):
        raise ArtifactIndexError("artifact index total_bytes is inconsistent")
    expected_commits = sorted({str(row["source_commit"]) for row in rows})
    if len(expected_commits) != 1 or frozen.get("source_commits") != expected_commits:
        raise ArtifactIndexError("artifact index source_commits is inconsistent")
    generation = frozen.get("generation_sha256")
    if not isinstance(generation, str) or SHA256_RE.fullmatch(generation) is None:
        raise ArtifactIndexError("artifact index has no valid generation SHA-256")
    if generation != _canonical_generation_sha256(frozen):
        raise ArtifactIndexError("artifact index generation SHA-256 is stale or inconsistent")
    return frozen, rows


def _stage_text(path: Path, text: str) -> Path:
    """Durably stage one text file without replacing its destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staged = temporary
        temporary = None
        return staged
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_raw_artifact_index(index: Mapping[str, Any], output_root: Path | str) -> tuple[Path, Path]:
    """Publish two validated views with JSON-last commit-marker semantics.

    Each replacement is atomic, but the pair cannot be atomically replaced on
    an ordinary filesystem.  CSV is published first and JSON last; readers use
    the JSON generation as the committed view and reject a CSV generation that
    differs.  Thus a crash is detectable rather than silently mixed.
    """

    root = Path(output_root).expanduser().resolve(strict=True)
    frozen, raw_rows = _validate_canonical_index(index, root)
    generation = str(frozen["generation_sha256"])
    json_path = root / "raw_artifact_index.json"
    csv_path = root / "raw_artifact_index.csv"
    json_text = json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n"

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ArtifactIndexError("artifact index contains a non-object row")
        unexpected = set(raw).difference(ARTIFACT_ROW_COLUMNS)
        missing = set(ARTIFACT_ROW_COLUMNS).difference(raw)
        if unexpected or missing:
            raise ArtifactIndexError(
                "artifact index row fields differ from the CSV schema: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        row = dict(raw)
        row["content_sha256_verified"] = (
            "true" if row["content_sha256_verified"] is True else "false"
        )
        row["index_generation_sha256"] = generation
        writer.writerow(row)
    csv_text = buffer.getvalue()

    staged_json: Path | None = None
    staged_csv: Path | None = None
    try:
        # Both complete views exist and have been fsynced before publication.
        # JSON is the commit marker, so publish CSV first and JSON last.
        staged_json = _stage_text(json_path, json_text)
        staged_csv = _stage_text(csv_path, csv_text)
        os.replace(staged_csv, csv_path)
        staged_csv = None
        _fsync_directory(root)
        os.replace(staged_json, json_path)
        staged_json = None
        _fsync_directory(root)
    finally:
        for staged in (staged_json, staged_csv):
            if staged is not None and staged.exists():
                staged.unlink()
    return json_path, csv_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        help="Suite output root containing suite_manifest.json and jobs/",
    )
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--verify-content",
        dest="verify_content",
        action="store_true",
        help="Recompute every .hrc SHA-256 (the default; retained for explicit scripts).",
    )
    verification.add_argument(
        "--metadata-only",
        dest="verify_content",
        action="store_false",
        help="Weaker diagnostic mode: validate redundant metadata and file sizes only.",
    )
    parser.set_defaults(verify_content=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index = build_raw_artifact_index(args.output_root, verify_content=args.verify_content)
        json_path, csv_path = write_raw_artifact_index(index, args.output_root)
    except ArtifactIndexError as exc:
        raise SystemExit(f"raw artifact index failed closed: {exc}") from exc
    print(
        json.dumps(
            {
                "artifact_count": index["artifact_count"],
                "total_bytes": index["total_bytes"],
                "verification_mode": index["verification_mode"],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
