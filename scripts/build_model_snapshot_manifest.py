#!/usr/bin/env python3
"""Build a read-only, content-addressed manifest for a local model snapshot.

Only checkpoint, configuration, tokenizer, processor, and custom model-code
files are included.  The source tree is never written.  The aggregate digest
is computed from canonical, path-sorted file records and deliberately excludes
the absolute model path, label, generation time, and output path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "model_snapshot_manifest.v1"
HASH_ALGORITHM = "sha256"
READ_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")
AGGREGATE_SCOPE = (
    "schema version plus newline-delimited canonical file records "
    "sorted by path; excludes label, locations, and generated_at"
)
MANIFEST_FIELDS = {
    "schema_version",
    "label",
    "model_dir",
    "output",
    "generated_at",
    "hash_algorithm",
    "aggregate_scope",
    "file_count",
    "total_bytes",
    "aggregate_sha256",
    "files",
}

_CHECKPOINT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".msgpack",
    ".h5",
    ".npy",
    ".npz",
    ".onnx",
    ".gguf",
)
_TOKENIZER_STEMS = (
    "tokenizer",
    "tokenization",
    "vocab",
    "merges",
    "added_tokens",
    "special_tokens",
    "sentencepiece",
    "spiece",
    "tiktoken",
    "chat_template",
)
_CONFIG_STEMS = (
    "config",
    "configuration",
    "generation_config",
    "quantization_config",
    "adapter_config",
    "processor_config",
    "preprocessor_config",
    "feature_extractor",
    "image_processor",
    "model_args",
    "hyperparameters",
    "params.json",
)
def _file_role(relative_path: Path) -> str | None:
    """Return the reproducibility role of a selected model file."""

    name = relative_path.name.lower()
    if (
        name.endswith(_CHECKPOINT_SUFFIXES)
        or name.endswith(".index.json")
        or name.endswith(".ckpt.index")
        or ".ckpt.data-" in name
    ):
        return "checkpoint"
    # Remote-code models routinely import snapshot-local helpers whose names do
    # not start with ``modeling_``.  Keeping every Python source file is a small
    # price for a closed, reproducible code snapshot.
    if name.endswith(".py"):
        return "model_code"
    if any(stem in name for stem in _TOKENIZER_STEMS):
        return "tokenizer"
    if any(stem in name for stem in _CONFIG_STEMS):
        return "config"
    return None


def _iter_tree_without_following_directory_symlinks(root: Path) -> Iterable[Path]:
    """Yield descendants and fail closed on a linked directory."""

    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        linked = sorted(
            name for name in directory_names if (directory_path / name).is_symlink()
        )
        if linked:
            raise ValueError(
                "model directory contains a directory symlink that would hide "
                f"snapshot content: {directory_path / linked[0]}"
            )
        directory_names[:] = sorted(directory_names)
        for name in sorted(file_names):
            yield directory_path / name


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _tree_identity_snapshot(root: Path) -> tuple[list[Path], dict[str, tuple[Any, ...]]]:
    """Capture a cheap point-in-time identity for every path in the tree."""

    paths = list(_iter_tree_without_following_directory_symlinks(root))
    snapshot: dict[str, tuple[Any, ...]] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            link = path.lstat()
            is_symlink = stat.S_ISLNK(link.st_mode)
            link_target = os.readlink(path) if is_symlink else None
            try:
                target = path.stat()
            except OSError as exc:
                if is_symlink:
                    raise ValueError(
                        f"selected path is a broken symlink: {relative}"
                    ) from exc
                raise
        except OSError as exc:
            raise ValueError(f"model snapshot path is unreadable: {path}") from exc
        snapshot[relative] = (
            _stat_identity(link),
            _stat_identity(target),
            is_symlink,
            link_target,
        )
    return paths, snapshot


def _stable_sha256_and_size(path: Path) -> tuple[str, int, bool, str | None]:
    """Hash one regular file and reject path/target changes during the read."""

    try:
        link_before = path.lstat()
        is_symlink = stat.S_ISLNK(link_before.st_mode)
        symlink_target = os.readlink(path) if is_symlink else None
        target_before = path.stat()
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as handle:
            descriptor_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise ValueError(f"selected path is not a regular file: {path}")
            for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
                digest.update(chunk)
                byte_count += len(chunk)
            descriptor_after = os.fstat(handle.fileno())
        link_after = path.lstat()
        target_after = path.stat()
        symlink_target_after = os.readlink(path) if is_symlink else None
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"selected file changed or became unreadable while hashing: {path}") from exc

    stable_descriptor = _stat_identity(descriptor_before) == _stat_identity(descriptor_after)
    stable_link = _stat_identity(link_before) == _stat_identity(link_after)
    stable_target = (
        _stat_identity(target_before)
        == _stat_identity(target_after)
        == _stat_identity(descriptor_after)
    )
    stable_symlink = path.is_symlink() == is_symlink and symlink_target_after == symlink_target
    if not (stable_descriptor and stable_link and stable_target and stable_symlink):
        raise ValueError(f"selected file changed while hashing: {path}")
    if byte_count != descriptor_after.st_size:
        raise ValueError(f"selected file size changed while hashing: {path}")
    return digest.hexdigest(), byte_count, is_symlink, symlink_target


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_paths(model_dir: Path, output: Path) -> tuple[Path, Path]:
    try:
        root = model_dir.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"model directory does not exist: {model_dir}") from exc
    if not root.is_dir():
        raise ValueError(f"model directory is not a directory: {model_dir}")

    resolved_output = output.expanduser().resolve(strict=False)
    if _is_within(resolved_output, root):
        raise ValueError(
            "output must be outside the model directory; an in-tree manifest "
            "would make the snapshot self-referential"
        )
    return root, resolved_output


def _canonical_record(record: dict[str, Any]) -> bytes:
    covered = {
        "path": record["path"],
        "role": record["role"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "is_symlink": record["is_symlink"],
        "symlink_target": record["symlink_target"],
    }
    return json.dumps(
        covered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _aggregate_sha256(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update((SCHEMA_VERSION + "\n").encode("ascii"))
    for record in records:
        digest.update(_canonical_record(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _generated_at_utc(value: str | None) -> str:
    if value is None:
        instant = datetime.now(timezone.utc)
    else:
        try:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"generated_at is not an ISO-8601 timestamp: {value!r}") from exc
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("generated_at must include a UTC offset")
    return instant.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_manifest(
    model_dir: Path | str,
    output: Path | str,
    label: str,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Inspect ``model_dir`` and return its selected-file manifest.

    ``output`` participates only in safety validation and is never opened by
    this function.  Call :func:`write_manifest` to persist the returned data.
    The optional ``generated_at`` injection is intended for deterministic unit
    tests; command-line use always records the current UTC time.
    """

    clean_label = str(label).strip()
    if not clean_label:
        raise ValueError("label must not be empty")

    root, resolved_output = _validate_paths(Path(model_dir), Path(output))
    all_files, tree_identity_before = _tree_identity_snapshot(root)
    if not all_files:
        raise ValueError(f"model directory contains no files: {root}")

    selected: list[tuple[str, Path, str]] = []
    for path in all_files:
        relative = path.relative_to(root)
        role = _file_role(relative)
        if role is not None:
            selected.append((relative.as_posix(), path, role))
    selected.sort(key=lambda item: item[0])
    if not selected:
        raise ValueError(
            "model directory contains no matching checkpoint, config, "
            f"tokenizer, processor, or model-code files: {root}"
        )
    for relative, path, _role in selected:
        if path.resolve(strict=False) == resolved_output:
            raise ValueError(
                "output is the target of a selected model-tree symlink; writing "
                f"it would make the snapshot self-referential: {relative}"
            )
    if not any(role == "checkpoint" for _relative, _path, role in selected):
        raise ValueError(
            "model directory contains no selected checkpoint file; refusing a "
            "config/tokenizer-only snapshot manifest"
        )

    records: list[dict[str, Any]] = []
    for relative, path, role in selected:
        if not path.is_file():
            kind = "broken symlink" if path.is_symlink() else "non-regular file"
            raise ValueError(f"selected path is a {kind}: {relative}")
        sha256, byte_count, is_symlink, symlink_target = _stable_sha256_and_size(path)
        records.append(
            {
                "path": relative,
                "role": role,
                "bytes": byte_count,
                "sha256": sha256,
                "is_symlink": is_symlink,
                "symlink_target": symlink_target,
            }
        )

    _final_files, tree_identity_after = _tree_identity_snapshot(root)
    if tree_identity_after != tree_identity_before:
        raise ValueError("model directory changed while building the snapshot manifest")

    timestamp = _generated_at_utc(generated_at)
    return {
        "schema_version": SCHEMA_VERSION,
        "label": clean_label,
        "model_dir": str(root),
        "output": str(resolved_output),
        "generated_at": timestamp,
        "hash_algorithm": HASH_ALGORITHM,
        "aggregate_scope": AGGREGATE_SCOPE,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "aggregate_sha256": _aggregate_sha256(records),
        "files": records,
    }


def _validate_manifest_object(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recompute all derived fields before a manifest can be persisted."""

    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("model manifest top-level fields are invalid")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("model manifest schema_version is invalid")
    if manifest.get("hash_algorithm") != HASH_ALGORITHM:
        raise ValueError("model manifest hash_algorithm is invalid")
    if manifest.get("aggregate_scope") != AGGREGATE_SCOPE:
        raise ValueError("model manifest aggregate_scope is invalid")
    label = manifest.get("label")
    if not isinstance(label, str) or not label or label.strip() != label:
        raise ValueError("model manifest label is invalid")
    generated_at = manifest.get("generated_at")
    if not isinstance(generated_at, str) or _generated_at_utc(generated_at) != generated_at:
        raise ValueError("model manifest generated_at is not canonical UTC")
    model_dir_value = manifest.get("model_dir")
    output_value = manifest.get("output")
    if not isinstance(model_dir_value, str) or not isinstance(output_value, str):
        raise ValueError("model manifest locations are invalid")
    model_dir = Path(model_dir_value).expanduser().resolve(strict=False)
    output = Path(output_value).expanduser().resolve(strict=False)
    if _is_within(output, model_dir):
        raise ValueError("model manifest output is inside the model directory")
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("model manifest has no file records")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_fields = {
        "path",
        "role",
        "bytes",
        "sha256",
        "is_symlink",
        "symlink_target",
    }
    allowed_roles = {"checkpoint", "config", "tokenizer", "model_code"}
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise ValueError(f"model manifest record {index} fields are invalid")
        path = raw.get("path")
        relative = Path(path) if isinstance(path, str) else None
        if (
            relative is None
            or not path
            or "\\" in path
            or relative.is_absolute()
            or relative.as_posix() != path
            or any(part in ("", ".", "..") for part in relative.parts)
            or path in seen
        ):
            raise ValueError(f"model manifest record {index} path is invalid")
        seen.add(path)
        if raw.get("role") not in allowed_roles:
            raise ValueError(f"model manifest record {index} role is invalid")
        byte_count = raw.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"model manifest record {index} byte count is invalid")
        if not isinstance(raw.get("sha256"), str) or SHA256_RE.fullmatch(raw["sha256"]) is None:
            raise ValueError(f"model manifest record {index} SHA-256 is invalid")
        is_symlink = raw.get("is_symlink")
        if type(is_symlink) is not bool:
            raise ValueError(f"model manifest record {index} symlink flag is invalid")
        symlink_target = raw.get("symlink_target")
        if (is_symlink and not isinstance(symlink_target, str)) or (
            not is_symlink and symlink_target is not None
        ):
            raise ValueError(f"model manifest record {index} symlink target is invalid")
        records.append(dict(raw))
    if [record["path"] for record in records] != sorted(seen):
        raise ValueError("model manifest records are not path-sorted")
    if not any(record["role"] == "checkpoint" for record in records):
        raise ValueError("model manifest contains no checkpoint record")
    if manifest.get("file_count") != len(records):
        raise ValueError("model manifest file_count is inconsistent")
    if manifest.get("total_bytes") != sum(record["bytes"] for record in records):
        raise ValueError("model manifest total_bytes is inconsistent")
    expected_aggregate = _aggregate_sha256(records)
    if manifest.get("aggregate_sha256") != expected_aggregate:
        raise ValueError("model manifest aggregate_sha256 is stale or inconsistent")
    return records


def _fsync_directory(directory: Path) -> None:
    """Persist a directory entry update on platforms that support it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_manifest(manifest: Mapping[str, Any], output: Path | str) -> None:
    """Atomically write a manifest outside its inspected model tree."""

    _validate_manifest_object(manifest)
    destination = Path(output).expanduser().resolve(strict=False)
    recorded_destination = Path(str(manifest.get("output", ""))).resolve(strict=False)
    if destination != recorded_destination:
        raise ValueError(
            "write destination does not match the output path validated by build_manifest"
        )
    model_dir = Path(str(manifest.get("model_dir", ""))).resolve(strict=False)
    if _is_within(destination, model_dir):
        raise ValueError("refusing to write a manifest inside the model directory")

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def verify_model_snapshot_manifest(
    manifest_path: Path | str,
    model_dir: Path | str,
    *,
    expected_manifest_sha256: str | None = None,
    expected_aggregate_sha256: str | None = None,
) -> dict[str, Any]:
    """Rehash ``model_dir`` and bind it to one committed manifest file."""

    path = Path(manifest_path).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("model snapshot manifest file SHA-256 differs from the contract")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model snapshot manifest is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("model snapshot manifest must contain an object")
    _validate_manifest_object(manifest)
    actual_root = Path(model_dir).expanduser().resolve(strict=True)
    recorded_root = Path(str(manifest["model_dir"])).expanduser().resolve(strict=True)
    if actual_root != recorded_root:
        raise ValueError(
            f"model directory differs from snapshot manifest: {actual_root} != {recorded_root}"
        )
    aggregate = str(manifest["aggregate_sha256"])
    if expected_aggregate_sha256 is not None and aggregate != expected_aggregate_sha256:
        raise ValueError("model snapshot aggregate SHA-256 differs from the contract")

    probe_output = path.parent / f".{path.name}.verification-probe"
    observed = build_manifest(
        actual_root,
        probe_output,
        str(manifest["label"]),
        generated_at=str(manifest["generated_at"]),
    )
    for field in ("file_count", "total_bytes", "aggregate_sha256", "files"):
        if observed[field] != manifest[field]:
            raise ValueError(f"model snapshot current tree differs in {field}")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "model_dir": str(actual_root),
        "aggregate_sha256": aggregate,
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "verified_current_tree": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.model_dir, args.output, args.label)
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": manifest["output"],
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "aggregate_sha256": manifest["aggregate_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
