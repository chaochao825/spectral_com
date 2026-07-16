#!/usr/bin/env python3
"""Build fail-closed ICML artifacts for the three-job scaling pilot.

The source of truth is the declared pilot suite plus its complete job
directories.  The script deliberately revalidates identities, source hashes,
physical files, held-out endpoints, paired fixed windows, and 13-point path
probes before writing any derived result.  It never pools rows across models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "scaling_pilot_report.v1"
SUITE_SCHEMA = "large_scale_hessian_suite.v1"
SUITE_MANIFEST_SCHEMA = "large_scale_hessian_suite_manifest.v1"
JOB_RECORD_SCHEMA = "large_scale_hessian_job_record.v1"
SUITE_ID = "large_scale_hessian_pilot_20260714"
PAYLOAD_SCOPE = "selected_linear_weights_only"
EVIDENCE_ROLE = "scalability_smoke"
INTERVAL_SEMANTICS = (
    "fixed-window descriptive mean +/- 1.96 sample standard errors; "
    "not an independence-based confidence interval or significance test"
)
FLOAT32_EPSILON = 2.0**-23
# Aggregate Hessian totals and the separately accumulated self/cross ledger use
# different float32 reduction trees across up to 12 tensors and six terms.
# Sixty-four float32 epsilons times the observed reduction scale is a
# conservative closure bound for these recorded reduction trees; materially
# different ledgers remain fail-closed.
HESSIAN_DECOMPOSITION_ULP_FACTOR = 64.0
PSD_REJECTION_RTOL = 1e-7
FLOAT32_PSD_FLOOR_RTOL = 8.0 * FLOAT32_EPSILON
PSD_FLOAT32_CLOSURE_RTOL = 0.5 * FLOAT32_EPSILON
CODEC_MAGIC = b"HRCODEC1"
CODEC_VERSION = 1
CODEC_ALIGNMENT = 64
CODEC_PREFIX = struct.Struct("<8sIIQ")
STRICT = "Q+S+L_QL_budget_component_scale"
QL = "Q+L"
EXPECTED_STRATEGIES = (
    "Q",
    "Q_global_scale",
    "Q_block_scale",
    "Q+S",
    "Q+S_OBS",
    "Q+L",
    "Q+S+L_QL_budget",
    "Q+S+L_QL_budget_component_scale",
    "Q+S+L",
    "Q+S_OBS+L",
    "Q+S+L_component_scale",
)
EXPECTED_STAGES = (
    "pythia70m_full_mlp_pilot",
    "opt125m_depth_mlp_pilot",
    "qwen3_06b_depth_mlp_pilot",
)
EXPECTED_OUTPUTS = (
    "run_config.json",
    "candidate_ablation.csv",
    "strategy_endpoints.csv",
    "endpoint_window_nll.csv",
    "comfort_sweep.csv",
    "comfort_summary.csv",
    "covariance_psd_audit.csv",
    "artifact_manifest.json",
    "artifact_payloads.csv",
    "summary.md",
    "figures/pretrained_hessian_repair_probe.pdf",
    "figures/pretrained_hessian_repair_probe.png",
)
EXPECTED_COMFORT_EPSILONS = (
    0.0,
    0.03125,
    0.0625,
    0.09375,
    0.125,
    0.1875,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)
EXPECTED_COMFORT_STRATEGIES = (
    "Q",
    "Q_block_scale",
    "Q+S_OBS",
    "Q+L",
    STRICT,
    "Q+S+L_component_scale",
)
EXPECTED_SOURCE_PATHS = {
    "runner": "scripts/run_pretrained_hessian_repair.py",
    "codec": "src/llm_spectral_dynamics/structured/codec_artifact.py",
    "hessian_repair": "src/llm_spectral_dynamics/structured/hessian_repair.py",
    "base_runner": "scripts/run_pretrained_llm_orthogonality.py",
}
EXPECTED_STAGE_MATERIAL = {
    "pythia70m_full_mlp_pilot": {
        "model": "EleutherAI/pythia-70m",
        "model_scale": "70M",
        "model_availability": "required",
        "model_override_env": "COMPRESSION_PYTHIA70M_MODEL",
        "resolved_commit": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "revision": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
        "scope_id": "full_mlp_weights",
        "claim_scope": "all transformer-block MLP projection weights; embeddings, attention, and output head remain dense",
        "module_types": ["dense_h_to_4h", "dense_4h_to_h"],
        "layers": [0, 1, 2, 3, 4, 5],
        "selected_tensors": 12,
    },
    "opt125m_depth_mlp_pilot": {
        "model": "facebook/opt-125m",
        "model_scale": "125M",
        "model_availability": "optional",
        "model_override_env": "COMPRESSION_OPT125M_MODEL",
        "resolved_commit": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
        "revision": "",
        "scope_id": "five_depth_mlp_weights",
        "claim_scope": "five depth-stratified transformer blocks; only fc1/fc2 MLP projection weights are charged and perturbed",
        "module_types": ["fc1", "fc2"],
        "layers": [0, 3, 6, 9, 11],
        "selected_tensors": 10,
    },
    "qwen3_06b_depth_mlp_pilot": {
        "model": "Qwen/Qwen3-0.6B",
        "model_scale": "0.6B",
        "model_availability": "optional",
        "model_override_env": "COMPRESSION_QWEN3_06B_MODEL",
        "resolved_commit": "c1899de289a04d12100db370d81485cdf75e47ca",
        "revision": "",
        "scope_id": "five_depth_mlp_weights",
        "claim_scope": "five depth-stratified transformer blocks; only up/down MLP projection weights are charged and perturbed",
        "module_types": ["up_proj", "down_proj"],
        "layers": [0, 7, 14, 21, 27],
        "selected_tensors": 10,
    },
}
COMPARISONS = (
    ("global_scale_vs_q", "Q_global_scale", "Q"),
    ("block_scale_vs_q", "Q_block_scale", "Q"),
    ("obs_vs_qs", "Q+S_OBS", "Q+S"),
    ("strict_qsl_vs_ql", STRICT, QL),
)
MODEL_LABELS = {
    "pythia70m_full_mlp_pilot": "Pythia-70M",
    "opt125m_depth_mlp_pilot": "OPT-125M",
    "qwen3_06b_depth_mlp_pilot": "Qwen3-0.6B",
}
MODEL_MACROS = {
    "pythia70m_full_mlp_pilot": "Pythia",
    "opt125m_depth_mlp_pilot": "OPT",
    "qwen3_06b_depth_mlp_pilot": "Qwen",
}
SCOPE_LABELS = {
    "full_mlp_weights": "full MLP",
    "five_depth_mlp_weights": "5-depth MLP",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LAYER_PATTERNS = (
    re.compile(r"\.layers\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.block\.(\d+)\."),
    re.compile(r"\.blocks\.(\d+)\."),
)


class ReportError(RuntimeError):
    """An input cannot support the declared scaling-smoke report."""


@dataclass(frozen=True)
class PairStats:
    comparison_id: str
    left: str
    right: str
    count: int
    mean: float
    standard_error: float
    interval_low: float
    interval_high: float
    left_wins: int


@dataclass(frozen=True)
class ContainerInspection:
    manifest: dict[str, Any]
    kind: str
    file_bytes: int
    natural_file_bytes: int
    logical_payload_bits: int
    stream_bytes: int
    container_bytes: int
    alignment_padding_bytes: int
    tail_padding_bytes: int
    alignment_bytes: int
    sha256: str
    layer_shapes: dict[str, tuple[int, int]]
    q_scale_count: int
    sparse_nnz: int
    lowrank_rank_sum: int
    component_sha256: dict[str, str]


@dataclass
class ValidatedJob:
    stage: dict[str, Any]
    manifest_entry: dict[str, Any]
    run_config: dict[str, Any]
    endpoint_rows: list[dict[str, str]]
    endpoint_by_strategy: dict[str, dict[str, str]]
    artifact_manifest: dict[str, Any]
    windows: dict[str, dict[int, tuple[int, float, float, int, int]]]
    comfort_rows: list[dict[str, str]]
    comfort_summary: dict[str, dict[str, str]]
    pairs: list[PairStats]
    input_hashes: dict[str, str]

    @property
    def stage_id(self) -> str:
        return str(self.stage["id"])

    @property
    def job_id(self) -> str:
        return str(self.manifest_entry["job_id"])

    @property
    def label(self) -> str:
        return MODEL_LABELS[self.stage_id]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReportError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"expected JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise ReportError(f"cannot read CSV from {path}: {exc}") from exc
    if not rows:
        raise ReportError(f"empty CSV evidence: {path}")
    return rows


def _safe_file(root: Path, relative: object, *, nonempty: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ReportError(f"unsafe relative evidence path: {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ReportError(f"evidence path escapes its root: {relative!r}") from exc
    if not path.is_file() or (nonempty and path.stat().st_size <= 0):
        raise ReportError(f"missing or empty evidence file: {path}")
    return path


def _require_sha(value: object, where: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ReportError(f"{where} is not a lowercase SHA-256")
    return value


def _integer(value: object, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ReportError(f"{where} is not an integer")
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ReportError(f"{where} is not an integer: {value!r}") from exc
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise ReportError(f"{where} is not integral: {value!r}")
    parsed = int(decimal_value)
    if positive and parsed <= 0:
        raise ReportError(f"{where} must be positive")
    return parsed


def _finite(value: object, where: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{where} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ReportError(f"{where} must be finite: {value!r}")
    return parsed


def _optional_float(value: object, where: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{where} is not numeric: {value!r}") from exc
    if math.isinf(parsed):
        raise ReportError(f"{where} cannot be infinite")
    return parsed


def _boolean(value: object, where: str) -> bool:
    if value is True or (isinstance(value, str) and value.lower() == "true"):
        return True
    if value is False or (isinstance(value, str) and value.lower() == "false"):
        return False
    raise ReportError(f"{where} is not boolean: {value!r}")


def _close(
    actual: float,
    expected: float,
    where: str,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise ReportError(f"{where} mismatch: {actual!r} != {expected!r}")


def _match(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _match(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _rho_kind(value: float, threshold: float) -> str:
    if math.isnan(value):
        return "inactive"
    if abs(value) <= threshold:
        return "hessian_orthogonal"
    if value < -threshold:
        return "repair_cancellation"
    return "positive_conflict"


def _rho_from_geometry(cross: float, self_left: float, self_right: float) -> float:
    denominator = math.sqrt(max(2.0 * self_left, 0.0) * max(2.0 * self_right, 0.0))
    if denominator <= 1e-12:
        return float("nan")
    return max(-1.0, min(1.0, cross / denominator))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ReportError("Pearson correlation needs paired vectors of length at least two")
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= 0.0:
        raise ReportError("Pearson correlation is undefined for a constant path")
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def _fit_linear_quadratic(epsilons: Sequence[float], values: Sequence[float]) -> tuple[float, float]:
    if len(epsilons) != len(values) or len(epsilons) < 2:
        raise ReportError("Taylor fit needs at least two paired points")
    s2 = sum(epsilon * epsilon for epsilon in epsilons)
    s3 = sum(epsilon**3 for epsilon in epsilons)
    s4 = sum(epsilon**4 for epsilon in epsilons)
    t1 = sum(epsilon * value for epsilon, value in zip(epsilons, values))
    t2 = sum(epsilon * epsilon * value for epsilon, value in zip(epsilons, values))
    determinant = s2 * s4 - s3 * s3
    if abs(determinant) <= 1e-30:
        raise ReportError("Taylor fit design is singular")
    linear = (t1 * s4 - t2 * s3) / determinant
    quadratic = (s2 * t2 - s3 * t1) / determinant
    return linear, quadratic


def _layer_index(name: str) -> int | None:
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _verify_file_record(root: Path, record: Mapping[str, Any], *, prefix: str = "") -> tuple[Path, int, str]:
    path_key = f"{prefix}path" if prefix else "path"
    bytes_key = f"{prefix}file_bytes" if prefix else "file_bytes"
    sha_key = f"{prefix}sha256" if prefix else "sha256"
    path = _safe_file(root, record.get(path_key))
    expected_bytes = _integer(record.get(bytes_key), f"{path_key}.{bytes_key}", positive=True)
    expected_sha = _require_sha(record.get(sha_key), f"{path_key}.{sha_key}")
    if path.stat().st_size != expected_bytes:
        raise ReportError(f"artifact size mismatch for {path}")
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha:
        raise ReportError(f"artifact SHA-256 mismatch for {path}")
    return path, expected_bytes, expected_sha


def _codec_int(value: object, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{where} is not an integer in the codec header")
    if positive and value <= 0:
        raise ReportError(f"{where} must be positive in the codec header")
    return value


def _codec_shape(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ReportError(f"{where} is not a non-empty codec shape")
    shape = tuple(_codec_int(item, f"{where} dimension", positive=True) for item in value)
    return shape


def _read_file_range(handle: Any, start: int, length: int, where: str) -> bytes:
    handle.seek(start)
    payload = handle.read(length)
    if len(payload) != length:
        raise ReportError(f"truncated codec range for {where}")
    return payload


def _require_zero_range(handle: Any, start: int, length: int, where: str) -> None:
    handle.seek(start)
    remaining = length
    while remaining:
        block = handle.read(min(1024 * 1024, remaining))
        if not block or any(block):
            raise ReportError(f"nonzero or truncated codec padding in {where}")
        remaining -= len(block)


def _hash_file_range(handle: Any, start: int, length: int, where: str) -> str:
    handle.seek(start)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = handle.read(min(1024 * 1024, remaining))
        if not block:
            raise ReportError(f"truncated codec stream in {where}")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _expected_codec_layout(
    manifest: Mapping[str, Any], kind: str
) -> tuple[dict[str, tuple[int, int]], int, int, int, list[dict[str, object]]]:
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ReportError("codec header has no layers")
    layer_shapes: dict[str, tuple[int, int]] = {}
    expected_streams: list[dict[str, object]] = []
    sparse_total = 0
    rank_total = 0
    q_scale_total = 0
    layer_names: list[str] = []

    def add_stream(
        *,
        name: str,
        layer: str,
        component: str,
        encoding: str,
        dtype: str,
        shape: tuple[int, ...],
        nbytes: int,
        logical_bits: int,
    ) -> None:
        expected_streams.append(
            {
                "name": name,
                "layer": layer,
                "component": component,
                "encoding": encoding,
                "dtype": dtype,
                "shape": list(shape),
                "nbytes": nbytes,
                "logical_bits": logical_bits,
            }
        )

    for raw_layer in raw_layers:
        if not isinstance(raw_layer, dict):
            raise ReportError("codec header contains a non-object layer")
        name = raw_layer.get("name")
        if not isinstance(name, str) or not name:
            raise ReportError("codec layer name is missing")
        shape = _codec_shape(raw_layer.get("shape"), f"{name} shape")
        if len(shape) != 2:
            raise ReportError(f"{name}: codec layer is not a matrix")
        rows, cols = shape
        if name in layer_shapes:
            raise ReportError(f"duplicate codec layer: {name}")
        layer_names.append(name)
        layer_shapes[name] = (rows, cols)
        components = raw_layer.get("components")
        if not isinstance(components, dict):
            raise ReportError(f"{name}: codec components are missing")

        if kind == "fp16_selected_linear_reference":
            if set(raw_layer) != {"name", "shape", "components"}:
                raise ReportError(f"{name}: reference layer header fields changed")
            if set(components) != {"dense_fp16"}:
                raise ReportError(f"{name}: reference component set changed")
            stream_name = components.get("dense_fp16")
            if stream_name != f"{name}/dense_fp16":
                raise ReportError(f"{name}: reference stream name changed")
            add_stream(
                name=stream_name,
                layer=name,
                component="dense_fp16",
                encoding="raw_little_endian",
                dtype="float16",
                shape=(rows, cols),
                nbytes=rows * cols * 2,
                logical_bits=rows * cols * 16,
            )
            continue

        if set(raw_layer) != {
            "name",
            "shape",
            "q_bits",
            "q_col_block_size",
            "sparse_nnz",
            "lowrank_rank",
            "components",
        }:
            raise ReportError(f"{name}: endpoint layer header fields changed")
        expected_components = {
            "q_codes",
            "q_scales",
            "sparse_values",
            "sparse_row_ptr",
            "sparse_col_idx",
            "lowrank_left",
            "lowrank_right",
        }
        if set(components) != expected_components:
            raise ReportError(f"{name}: endpoint component set changed")
        bits = _codec_int(raw_layer.get("q_bits"), f"{name} q_bits", positive=True)
        if bits > 8:
            raise ReportError(f"{name}: q_bits exceeds the research codec")
        block = raw_layer.get("q_col_block_size")
        if block is not None:
            block = _codec_int(block, f"{name} q_col_block_size", positive=True)
        scale_shape = (rows,) if block is None else (rows, (cols + block - 1) // block)
        q_scale_total += math.prod(scale_shape)
        for component, encoding, dtype, stream_shape, nbytes, logical_bits in (
            (
                "q_codes",
                f"signed_twos_complement_lsb_bitpack_{bits}",
                "bitpack",
                (rows, cols),
                (rows * cols * bits + 7) // 8,
                rows * cols * bits,
            ),
            (
                "q_scales",
                "raw_little_endian",
                "float16",
                scale_shape,
                math.prod(scale_shape) * 2,
                math.prod(scale_shape) * 16,
            ),
        ):
            stream_name = components.get(component)
            if stream_name != f"{name}/{component}":
                raise ReportError(f"{name}: {component} stream name changed")
            add_stream(
                name=stream_name,
                layer=name,
                component=component,
                encoding=encoding,
                dtype=dtype,
                shape=stream_shape,
                nbytes=nbytes,
                logical_bits=logical_bits,
            )

        sparse_nnz = _codec_int(raw_layer.get("sparse_nnz"), f"{name} sparse_nnz")
        if sparse_nnz < 0 or sparse_nnz > rows * cols:
            raise ReportError(f"{name}: sparse_nnz is outside the layer extent")
        sparse_total += sparse_nnz
        sparse_names = ("sparse_values", "sparse_row_ptr", "sparse_col_idx")
        if sparse_nnz == 0:
            if any(components.get(component) is not None for component in sparse_names):
                raise ReportError(f"{name}: zero sparse_nnz has stored sparse streams")
        else:
            col_bytes = 1 if cols <= 2**8 else 2 if cols <= 2**16 else 4
            col_dtype = {1: "uint8", 2: "uint16", 4: "uint32"}[col_bytes]
            sparse_specs = (
                ("sparse_values", "csr_row_major_values", "float16", (sparse_nnz,), sparse_nnz * 2, sparse_nnz * 16),
                ("sparse_row_ptr", "csr_fixed", "uint32", (rows + 1,), (rows + 1) * 4, (rows + 1) * 32),
                ("sparse_col_idx", "csr_fixed", col_dtype, (sparse_nnz,), sparse_nnz * col_bytes, sparse_nnz * col_bytes * 8),
            )
            for component, encoding, dtype, stream_shape, nbytes, logical_bits in sparse_specs:
                stream_name = components.get(component)
                if stream_name != f"{name}/{component}":
                    raise ReportError(f"{name}: {component} stream name changed")
                add_stream(
                    name=stream_name,
                    layer=name,
                    component=component,
                    encoding=encoding,
                    dtype=dtype,
                    shape=stream_shape,
                    nbytes=nbytes,
                    logical_bits=logical_bits,
                )

        rank = _codec_int(raw_layer.get("lowrank_rank"), f"{name} lowrank_rank")
        if rank < 0 or rank > min(rows, cols):
            raise ReportError(f"{name}: lowrank_rank is outside the layer extent")
        rank_total += rank
        rank_names = ("lowrank_left", "lowrank_right")
        if rank == 0:
            if any(components.get(component) is not None for component in rank_names):
                raise ReportError(f"{name}: rank zero has stored low-rank streams")
        else:
            for component, stream_shape in (
                ("lowrank_left", (rows, rank)),
                ("lowrank_right", (rank, cols)),
            ):
                stream_name = components.get(component)
                if stream_name != f"{name}/{component}":
                    raise ReportError(f"{name}: {component} stream name changed")
                add_stream(
                    name=stream_name,
                    layer=name,
                    component=component,
                    encoding="raw_little_endian",
                    dtype="float16",
                    shape=stream_shape,
                    nbytes=math.prod(stream_shape) * 2,
                    logical_bits=math.prod(stream_shape) * 16,
                )

    if layer_names != sorted(layer_names):
        raise ReportError("codec layers are not in canonical name order")
    if len({str(item["name"]) for item in expected_streams}) != len(expected_streams):
        raise ReportError("codec stream names are not unique")
    return layer_shapes, q_scale_total, sparse_total, rank_total, expected_streams


def _inspect_codec_container(path: Path) -> ContainerInspection:
    file_bytes = path.stat().st_size
    if file_bytes < CODEC_PREFIX.size:
        raise ReportError(f"artifact is shorter than its codec prefix: {path}")
    with path.open("rb") as handle:
        prefix = _read_file_range(handle, 0, CODEC_PREFIX.size, f"{path} prefix")
        magic, version, alignment, header_size = CODEC_PREFIX.unpack(prefix)
        if magic != CODEC_MAGIC or version != CODEC_VERSION:
            raise ReportError(f"unsupported research-codec magic/version: {path}")
        if alignment != CODEC_ALIGNMENT:
            raise ReportError(f"research-codec alignment is not {CODEC_ALIGNMENT}: {path}")
        if header_size <= 0 or CODEC_PREFIX.size + header_size > file_bytes:
            raise ReportError(f"invalid research-codec header size: {path}")
        header = _read_file_range(
            handle, CODEC_PREFIX.size, header_size, f"{path} JSON header"
        )
        try:
            manifest = json.loads(header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReportError(f"invalid research-codec JSON header: {path}") from exc
        if not isinstance(manifest, dict) or _canonical_json_bytes(manifest) != header:
            raise ReportError(f"research-codec header is not canonical JSON: {path}")
        if set(manifest) != {
            "format",
            "version",
            "kind",
            "offset_semantics",
            "alignment_bytes",
            "layers",
            "streams",
            "transparent_compression",
        }:
            raise ReportError(f"research-codec top-level fields changed: {path}")
        if (
            manifest.get("format") != "llm_spectral_dynamics_research_codec"
            or manifest.get("version") != CODEC_VERSION
            or manifest.get("alignment_bytes") != CODEC_ALIGNMENT
            or manifest.get("offset_semantics") != "relative_to_aligned_payload_base"
            or manifest.get("transparent_compression") != "none"
        ):
            raise ReportError(f"research-codec format contract changed: {path}")
        kind = manifest.get("kind")
        if kind not in {"fp16_selected_linear_reference", "qsl_selected_linear_weights"}:
            raise ReportError(f"unexpected research-codec kind: {path}")
        (
            layer_shapes,
            q_scale_total,
            sparse_total,
            rank_total,
            expected_streams,
        ) = _expected_codec_layout(manifest, str(kind))
        streams = manifest.get("streams")
        if not isinstance(streams, list) or len(streams) != len(expected_streams):
            raise ReportError(f"research-codec stream count changed: {path}")
        payload_base = (
            (CODEC_PREFIX.size + header_size + alignment - 1) // alignment * alignment
        )
        _require_zero_range(
            handle,
            CODEC_PREFIX.size + header_size,
            payload_base - CODEC_PREFIX.size - header_size,
            f"{path} header alignment",
        )
        previous_end = 0
        internal_padding = 0
        stream_bytes = 0
        logical_bits = 0
        component_sha: dict[str, str] = {}
        stream_by_name: dict[str, dict[str, Any]] = {}
        required_record_keys = {
            "name",
            "layer",
            "component",
            "encoding",
            "dtype",
            "shape",
            "offset",
            "nbytes",
            "logical_bits",
            "sha256",
        }
        for raw, expected in zip(streams, expected_streams):
            if not isinstance(raw, dict) or set(raw) != required_record_keys:
                raise ReportError(f"research-codec stream fields changed: {path}")
            for key, value in expected.items():
                if raw.get(key) != value:
                    raise ReportError(f"research-codec stream {key} mismatch: {path}")
            offset = _codec_int(raw.get("offset"), f"{path} stream offset")
            nbytes = _codec_int(raw.get("nbytes"), f"{path} stream bytes", positive=True)
            expected_offset = (previous_end + alignment - 1) // alignment * alignment
            if offset != expected_offset:
                raise ReportError(f"research-codec stream layout is noncanonical: {path}")
            gap = offset - previous_end
            _require_zero_range(
                handle, payload_base + previous_end, gap, f"{path} internal alignment"
            )
            start = payload_base + offset
            stop = start + nbytes
            if stop > file_bytes:
                raise ReportError(f"research-codec stream exceeds the file: {path}")
            expected_sha = _require_sha(raw.get("sha256"), f"{path} stream SHA")
            actual_sha = _hash_file_range(handle, start, nbytes, f"{path}/{raw.get('name')}")
            if actual_sha != expected_sha:
                raise ReportError(f"research-codec stream checksum mismatch: {path}")
            name = str(raw["name"])
            component_sha[name] = actual_sha
            stream_by_name[name] = raw
            previous_end = offset + nbytes
            internal_padding += gap
            stream_bytes += nbytes
            logical_bits += _codec_int(raw.get("logical_bits"), f"{path} logical bits")

        natural_file_bytes = payload_base + previous_end
        if natural_file_bytes > file_bytes:
            raise ReportError(f"research-codec natural size exceeds file size: {path}")
        tail_padding = file_bytes - natural_file_bytes
        _require_zero_range(
            handle, natural_file_bytes, tail_padding, f"{path} tail padding"
        )

        # Parse only the compact CSR index streams.  This checks that metadata
        # nnz is backed by a valid row structure without materializing weights.
        for layer in manifest["layers"]:
            if kind != "qsl_selected_linear_weights" or int(layer["sparse_nnz"]) == 0:
                continue
            rows, cols = map(int, layer["shape"])
            components = layer["components"]
            row_record = stream_by_name[str(components["sparse_row_ptr"])]
            col_record = stream_by_name[str(components["sparse_col_idx"])]
            row_start = payload_base + int(row_record["offset"])
            col_start = payload_base + int(col_record["offset"])
            row_payload = _read_file_range(
                handle, row_start, int(row_record["nbytes"]), f"{path} CSR row pointers"
            )
            row_ptr = [item[0] for item in struct.iter_unpack("<I", row_payload)]
            nnz = int(layer["sparse_nnz"])
            if len(row_ptr) != rows + 1 or row_ptr[0] != 0 or row_ptr[-1] != nnz:
                raise ReportError(f"invalid research-codec CSR row pointers: {path}")
            if any(right < left for left, right in zip(row_ptr, row_ptr[1:])):
                raise ReportError(f"nonmonotone research-codec CSR row pointers: {path}")
            width = {"uint8": 1, "uint16": 2, "uint32": 4}[str(col_record["dtype"])]
            code = {1: "<B", 2: "<H", 4: "<I"}[width]
            col_payload = _read_file_range(
                handle, col_start, int(col_record["nbytes"]), f"{path} CSR columns"
            )
            columns = [item[0] for item in struct.iter_unpack(code, col_payload)]
            if len(columns) != nnz or any(column >= cols for column in columns):
                raise ReportError(f"invalid research-codec CSR column indices: {path}")
            for start, stop in zip(row_ptr, row_ptr[1:]):
                if columns[start:stop] != sorted(set(columns[start:stop])):
                    raise ReportError(f"noncanonical research-codec CSR row: {path}")

    return ContainerInspection(
        manifest=manifest,
        kind=str(kind),
        file_bytes=file_bytes,
        natural_file_bytes=natural_file_bytes,
        logical_payload_bits=logical_bits,
        stream_bytes=stream_bytes,
        container_bytes=payload_base,
        alignment_padding_bytes=(payload_base - CODEC_PREFIX.size - header_size)
        + internal_padding,
        tail_padding_bytes=tail_padding,
        alignment_bytes=alignment,
        sha256=_file_sha256(path),
        layer_shapes=layer_shapes,
        q_scale_count=q_scale_total,
        sparse_nnz=sparse_total,
        lowrank_rank_sum=rank_total,
        component_sha256=component_sha,
    )


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _format(value: float) -> str:
    if math.isnan(value):
        return ""
    return format(value, ".17g")


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _job_id(stage_id: str) -> str:
    return f"{stage_id}__seed17__rate0p258"


def _validate_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SUITE_SCHEMA or payload.get("suite_id") != SUITE_ID:
        raise ReportError("the report only accepts the declared three-job pilot suite")
    if tuple(payload.get("expected_strategies", ())) != EXPECTED_STRATEGIES:
        raise ReportError("pilot strategy order/set differs from the report contract")
    stages = payload.get("stages")
    if not isinstance(stages, list) or tuple(stage.get("id") for stage in stages if isinstance(stage, dict)) != EXPECTED_STAGES:
        raise ReportError("pilot stage order/set differs from the report contract")
    common = payload.get("common")
    if not isinstance(common, dict):
        raise ReportError("pilot config has no common argument contract")
    material_common = {
        "device": "auto",
        "svd_device": "auto",
        "torch_dtype": "float32",
        "local_files_only": True,
        "dataset": "wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "validation",
        "calib_limit": 4,
        "eval_limit": 8,
        "sequence_length": 128,
        "batch_size": 1,
        "texts_per_batch_window": 8,
        "selector_activation_sample_rows": 256,
        "bits": 4,
        "support_encoding": "csr_fixed",
        "emit_codec_artifacts": True,
        "enforce_serialized_rate_cap": True,
        "artifact_alignment": 64,
        "s_method": "wanda",
        "l_method": "whitened_svd",
        "repair_block_sizes": [32, 64, 128, 256, 512],
        "max_allocation_ranks": 32,
        "obs_rcond": 1e-10,
        "scale_min": 0.0,
        "scale_max": 2.0,
        "rho_threshold": 0.1,
        "rate_tolerance": 0.01,
        "comfort_relative_tolerance": 0.2,
        "comfort_absolute_tolerance": 0.0001,
    }
    for key, expected in material_common.items():
        if common.get(key) != expected:
            raise ReportError(f"pilot common.{key} differs from the unique pilot contract")
    expected_common_keys = set(material_common) | {
        "comfort_epsilons",
        "comfort_fit_max_epsilon",
        "comfort_strategies",
    }
    if set(common) != expected_common_keys:
        raise ReportError("pilot common argument key set differs from the unique pilot contract")
    if common.get("skip_comfort", False) is not False or common.get("proxy_only", False) is not False:
        raise ReportError("pilot must retain held-out endpoints and comfort paths")
    if common.get("comfort_fit_max_epsilon") != 0.125:
        raise ReportError("pilot Taylor fit interval must remain epsilon <= 0.125")
    epsilons = common.get("comfort_epsilons")
    if (tuple(epsilons) if isinstance(epsilons, list) else ()) != EXPECTED_COMFORT_EPSILONS:
        raise ReportError("pilot must retain the exact preregistered 13-point epsilon grid")
    comfort = common.get("comfort_strategies")
    if (tuple(comfort) if isinstance(comfort, list) else ()) != EXPECTED_COMFORT_STRATEGIES:
        raise ReportError("pilot comfort strategy contract is invalid")
    expected_outputs = payload.get("expected_outputs")
    if (tuple(expected_outputs) if isinstance(expected_outputs, list) else ()) != EXPECTED_OUTPUTS:
        raise ReportError("pilot expected output contract changed")
    for stage in stages:
        material = EXPECTED_STAGE_MATERIAL[str(stage["id"])]
        for key in (
            "model",
            "model_scale",
            "model_availability",
            "model_override_env",
            "revision",
        ):
            if stage.get(key, "") != material[key]:
                raise ReportError(f"{stage.get('id')}: material field {key} changed")
        if stage.get("lane") != "A_post_training_no_backward":
            raise ReportError(f"{stage.get('id')}: compression lane changed")
        if stage.get("evidence_role") != EVIDENCE_ROLE:
            raise ReportError(f"{stage.get('id')}: evidence role is not scalability_smoke")
        if stage.get("protocol_manifest_consumed") is not False:
            raise ReportError(f"{stage.get('id')}: protocol consumption claim is invalid")
        if stage.get("seed_aggregation_allowed") is not False:
            raise ReportError(f"{stage.get('id')}: seed aggregation must remain forbidden")
        if stage.get("data_window_independence") != "shared_sequential_windows_not_independent_across_seeds":
            raise ReportError(f"{stage.get('id')}: data-window dependence disclosure changed")
        if stage.get("seeds") != [17] or stage.get("rates") != [0.258]:
            raise ReportError(f"{stage.get('id')}: expected exactly seed 17 and rate 0.258")
        scope = stage.get("tensor_scope")
        if not isinstance(scope, dict):
            raise ReportError(f"{stage.get('id')}: missing tensor scope")
        scope_expected = {
            "id": material["scope_id"],
            "claim_scope": material["claim_scope"],
            "module_types": material["module_types"],
            "layers": material["layers"],
            "max_modules": 0,
            "expected_selected_tensors": material["selected_tensors"],
        }
        for key, expected in scope_expected.items():
            if scope.get(key) != expected:
                raise ReportError(f"{stage.get('id')}: tensor scope field {key} changed")


def _validate_source_snapshot(snapshot: object, repo_root: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(snapshot, dict) or not snapshot:
        raise ReportError("suite manifest has no numerical source snapshot")
    if set(snapshot) != set(EXPECTED_SOURCE_PATHS):
        raise ReportError("suite numerical source file set differs from the pilot contract")
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in snapshot.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise ReportError("invalid numerical source snapshot entry")
        relative = raw.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ReportError(f"numerical source {name} is not repository-relative")
        if relative != EXPECTED_SOURCE_PATHS[name]:
            raise ReportError(f"numerical source {name} path differs from the pilot contract")
        # The completed jobs are audited against this immutable recorded
        # source closure.  The current worktree may legitimately contain a
        # later runner revision; it must not be substituted for the recorded
        # hashes, nor should source drift make historical artifacts disappear.
        _safe_file(repo_root, relative)
        size = _integer(raw.get("size_bytes"), f"source {name} size", positive=True)
        sha = _require_sha(raw.get("sha256"), f"source {name} SHA")
        normalized[name] = {"path": relative, "size_bytes": size, "sha256": sha}
    return normalized


def _expected_job_hash(config: dict[str, Any], stage: dict[str, Any], entry: dict[str, Any]) -> str:
    effective = dict(config["common"])
    effective.update(
        {
            "model": entry["model_argument"],
            "revision": str(stage.get("revision", "")),
            "module_types": list(stage["tensor_scope"]["module_types"]),
            "layers": list(stage["tensor_scope"]["layers"]),
            "max_modules": int(stage["tensor_scope"]["max_modules"]),
            "target_ratios": [0.258],
            "endpoint_target": 0.258,
            "seed": 17,
        }
    )
    provisional = {
        "suite_id": SUITE_ID,
        "stage_id": stage["id"],
        "lane": stage["lane"],
        "evidence_role": stage["evidence_role"],
        "protocol_manifest_consumed": stage["protocol_manifest_consumed"],
        "seed_aggregation_allowed": stage["seed_aggregation_allowed"],
        "data_window_independence": stage["data_window_independence"],
        "model_declared": stage["model"],
        "model_argument": entry["model_argument"],
        "model_scale": stage["model_scale"],
        "model_availability": stage["model_availability"],
        "availability_note": stage["availability_note"],
        "revision": str(stage.get("revision", "")),
        "seed": 17,
        "target_rate": 0.258,
        "tensor_scope": stage["tensor_scope"],
        "effective_arguments": effective,
    }
    return _object_sha256(provisional)


def _validate_log(suite_root: Path, record: dict[str, Any], key: str, job_id: str) -> None:
    raw = record.get(key)
    if not isinstance(raw, dict):
        raise ReportError(f"{job_id}: missing {key} log record")
    path = _safe_file(suite_root, raw.get("path"), nonempty=False)
    expected = (suite_root / "_logs" / f"{job_id}.{key}.log").resolve()
    if path != expected:
        raise ReportError(f"{job_id}: {key} log path differs from the suite contract")
    expected_size = _integer(raw.get("size_bytes"), f"{job_id} {key} log size")
    expected_sha = _require_sha(raw.get("sha256"), f"{job_id} {key} log SHA")
    content = path.read_bytes()
    normalized = content.replace(b"\r\n", b"\n")
    raw_matches = len(content) == expected_size and hashlib.sha256(content).hexdigest() == expected_sha
    portable_matches = (
        len(normalized) == expected_size
        and hashlib.sha256(normalized).hexdigest() == expected_sha
    )
    if not raw_matches and not portable_matches:
        raise ReportError(f"{job_id}: {key} log size/SHA mismatch after CRLF normalization")


def _validate_endpoint_rows(
    job_dir: Path,
    run_config: dict[str, Any],
    expected_strategies: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any], dict[str, str]]:
    endpoint_path = job_dir / "strategy_endpoints.csv"
    artifact_path = job_dir / "artifact_manifest.json"
    rows = _read_csv(endpoint_path)
    if [row.get("strategy") for row in rows] != list(expected_strategies):
        raise ReportError(f"{job_dir.name}: endpoint strategy order/set differs from the suite contract")
    by_strategy = {str(row["strategy"]): row for row in rows}
    if len(by_strategy) != len(expected_strategies):
        raise ReportError(f"{job_dir.name}: duplicate endpoint strategy")

    manifest = _read_json(artifact_path)
    if (
        manifest.get("format") != "llm_spectral_dynamics_research_codec"
        or manifest.get("alignment_bytes") != CODEC_ALIGNMENT
        or manifest.get("scope") != PAYLOAD_SCOPE
        or manifest.get("production_backend") is not False
    ):
        raise ReportError(f"{job_dir.name}: artifact scope/backend disclosure is invalid")
    if manifest.get("serialized_rate_cap_enforced") is not True:
        raise ReportError(f"{job_dir.name}: serialized rate cap was not enforced")
    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise ReportError(f"{job_dir.name}: missing reference artifact")
    reference_file, reference_bytes, reference_sha = _verify_file_record(job_dir, reference)
    reference_inspection = _inspect_codec_container(reference_file)
    if reference_inspection.kind != "fp16_selected_linear_reference":
        raise ReportError(f"{job_dir.name}: reference has the wrong codec kind")
    if (
        reference_inspection.file_bytes != reference_bytes
        or reference_inspection.sha256 != reference_sha
        or reference_inspection.natural_file_bytes != reference_bytes
        or reference_inspection.tail_padding_bytes != 0
    ):
        raise ReportError(f"{job_dir.name}: reference codec ledger mismatch")
    if reference.get("roundtrip_exact_fp16") is not True:
        raise ReportError(f"{job_dir.name}: reference did not pass exact FP16 roundtrip")
    selected_parameters = _integer(
        run_config.get("selected_parameter_count"), f"{job_dir.name} selected parameters", positive=True
    )
    if _integer(reference.get("logical_payload_bits"), "reference logical bits", positive=True) != 16 * selected_parameters:
        raise ReportError(f"{job_dir.name}: reference logical bits do not equal selected FP16 parameters")
    if reference_inspection.logical_payload_bits != 16 * selected_parameters:
        raise ReportError(f"{job_dir.name}: decoded reference logical bits mismatch")
    selected_layers = run_config.get("selected_layers")
    if not isinstance(selected_layers, list) or set(reference_inspection.layer_shapes) != set(selected_layers):
        raise ReportError(f"{job_dir.name}: reference codec layers differ from selected tensors")
    if sum(math.prod(shape) for shape in reference_inspection.layer_shapes.values()) != selected_parameters:
        raise ReportError(f"{job_dir.name}: reference codec shapes do not close to selected parameters")

    strategies = manifest.get("strategies")
    if not isinstance(strategies, list) or [item.get("strategy") for item in strategies if isinstance(item, dict)] != list(expected_strategies):
        raise ReportError(f"{job_dir.name}: artifact strategy order/set differs from the suite contract")
    baseline = run_config.get("baseline_metrics")
    if not isinstance(baseline, dict):
        raise ReportError(f"{job_dir.name}: missing baseline metrics")
    baseline_nll = _finite(baseline.get("nll"), "baseline NLL")
    baseline_ppl = _finite(baseline.get("perplexity"), "baseline perplexity")
    _close(math.exp(baseline_nll), baseline_ppl, "baseline exp(NLL)", rel_tol=2e-9)
    eval_tokens = _integer(run_config.get("actual_eval_tokens"), "actual eval tokens", positive=True)
    run_arguments = run_config.get("arguments")
    if not isinstance(run_arguments, dict):
        raise ReportError(f"{job_dir.name}: run arguments are absent")
    rho_threshold = _finite(run_arguments.get("rho_threshold"), "rho threshold")
    quantization_bits = _integer(
        run_arguments.get("bits"), f"{job_dir.name} quantization bits", positive=True
    )
    if quantization_bits != 4:
        raise ReportError(f"{job_dir.name}: scaling pilot is not the declared 4-bit experiment")
    input_hashes = {
        endpoint_path.as_posix(): _file_sha256(endpoint_path),
        artifact_path.as_posix(): _file_sha256(artifact_path),
        reference_file.as_posix(): reference_sha,
    }
    payload_path = job_dir / "artifact_payloads.csv"
    payload_rows = _read_csv(payload_path)
    if [row.get("strategy") for row in payload_rows] != list(expected_strategies):
        raise ReportError(f"{job_dir.name}: artifact payload strategy order/set differs")
    input_hashes[payload_path.as_posix()] = _file_sha256(payload_path)

    artifact_by_strategy: dict[str, dict[str, Any]] = {}
    inspection_by_strategy: dict[str, ContainerInspection] = {}
    for raw, payload_row in zip(strategies, payload_rows):
        if not isinstance(raw, dict):
            raise ReportError(f"{job_dir.name}: non-object artifact strategy")
        strategy = str(raw["strategy"])
        artifact_by_strategy[strategy] = raw
        if not math.isclose(
            _finite(raw.get("target_ratio"), f"{strategy} manifest target"),
            0.258,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ReportError(f"{job_dir.name}/{strategy}: manifest target ratio changed")
        file_path, file_bytes, sha = _verify_file_record(job_dir, raw, prefix="artifact_")
        if file_path.suffix != ".hrc":
            raise ReportError(f"{job_dir.name}/{strategy}: artifact is not an HRC container")
        inspection = _inspect_codec_container(file_path)
        inspection_by_strategy[strategy] = inspection
        if inspection.kind != "qsl_selected_linear_weights":
            raise ReportError(f"{job_dir.name}/{strategy}: artifact has the wrong codec kind")
        if inspection.file_bytes != file_bytes or inspection.sha256 != sha:
            raise ReportError(f"{job_dir.name}/{strategy}: codec identity mismatch")
        if inspection.layer_shapes != reference_inspection.layer_shapes:
            raise ReportError(f"{job_dir.name}/{strategy}: codec tensor shapes differ from reference")
        for layer in inspection.manifest["layers"]:
            if _integer(layer.get("q_bits"), f"{strategy}/{layer.get('name')} q_bits", positive=True) != quantization_bits:
                raise ReportError(
                    f"{job_dir.name}/{strategy}: codec q_bits differ from run arguments"
                )
        input_hashes[file_path.as_posix()] = sha
        natural = _integer(raw.get("artifact_natural_file_bytes"), f"{strategy} natural bytes", positive=True)
        tail = _integer(raw.get("artifact_tail_padding_bytes"), f"{strategy} tail padding")
        if _integer(
            raw.get("reference_artifact_file_bytes"),
            f"{strategy} reference artifact bytes",
            positive=True,
        ) != reference_bytes:
            raise ReportError(
                f"{job_dir.name}/{strategy}: strategy reference bytes differ from the verified reference file"
            )
        if natural > file_bytes or natural + tail != file_bytes:
            raise ReportError(f"{job_dir.name}/{strategy}: natural bytes plus tail padding do not equal file bytes")
        codec_ledger = {
            "artifact_natural_file_bytes": inspection.natural_file_bytes,
            "artifact_logical_payload_bits": inspection.logical_payload_bits,
            "artifact_stream_bytes": inspection.stream_bytes,
            "artifact_container_bytes": inspection.container_bytes,
            "artifact_alignment_padding_bytes": inspection.alignment_padding_bytes,
            "artifact_tail_padding_bytes": inspection.tail_padding_bytes,
            # The runner defines this field against total logical payload
            # bits, so it also counts byte-rounding slack across bitstreams.
            "artifact_total_overhead_bytes": inspection.file_bytes
            - math.ceil(inspection.logical_payload_bits / 8.0),
        }
        for key, expected in codec_ledger.items():
            if _integer(raw.get(key), f"{strategy} manifest {key}") != expected:
                raise ReportError(f"{job_dir.name}/{strategy}: HRC {key} ledger mismatch")
        header_alignment = (
            inspection.container_bytes
            - CODEC_PREFIX.size
            - len(_canonical_json_bytes(inspection.manifest))
        )
        internal_alignment = inspection.alignment_padding_bytes - header_alignment
        serialized_overhead = inspection.file_bytes - inspection.stream_bytes
        if header_alignment < 0 or internal_alignment < 0 or serialized_overhead != (
            inspection.container_bytes + internal_alignment + inspection.tail_padding_bytes
        ):
            raise ReportError(
                f"{job_dir.name}/{strategy}: HRC serialized-overhead ledger does not close"
            )
        rate = file_bytes / reference_bytes
        _close(_finite(raw.get("artifact_to_reference_file_ratio"), f"{strategy} physical rate"), rate, f"{strategy} physical rate")
        _close(
            _finite(raw.get("artifact_physical_compression_ratio"), f"{strategy} compression ratio"),
            1.0 / rate,
            f"{strategy} physical compression ratio",
        )
        if raw.get("roundtrip_exact_fp16_endpoint") is not True:
            raise ReportError(f"{job_dir.name}/{strategy}: endpoint roundtrip is not exact")
        if raw.get("artifact_scope") != PAYLOAD_SCOPE or raw.get("production_backend") is not False:
            raise ReportError(f"{job_dir.name}/{strategy}: artifact scope/backend mismatch")

        row = by_strategy[strategy]
        if set(payload_row) != set(raw):
            raise ReportError(f"{job_dir.name}/{strategy}: payload CSV fields differ from manifest")
        for key, expected in raw.items():
            actual = payload_row.get(key)
            if isinstance(expected, bool):
                matched = _boolean(actual, f"{strategy} payload {key}") is expected
            elif isinstance(expected, int):
                matched = _integer(actual, f"{strategy} payload {key}") == expected
            elif isinstance(expected, float):
                matched = math.isclose(
                    _finite(actual, f"{strategy} payload {key}"),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            else:
                matched = str(actual) == str(expected)
            if not matched:
                raise ReportError(f"{job_dir.name}/{strategy}: payload CSV {key} mismatch")
        if not math.isclose(_finite(row.get("target_ratio"), f"{strategy} target"), 0.258, rel_tol=0.0, abs_tol=1e-12):
            raise ReportError(f"{job_dir.name}/{strategy}: target ratio changed")
        csv_manifest_pairs = (
            ("artifact_file_bytes", "artifact_file_bytes"),
            ("artifact_natural_file_bytes", "artifact_natural_file_bytes"),
            ("artifact_tail_padding_bytes", "artifact_tail_padding_bytes"),
            ("reference_artifact_file_bytes", "reference_artifact_file_bytes"),
        )
        for csv_key, manifest_key in csv_manifest_pairs:
            if _integer(row.get(csv_key), f"{strategy} CSV {csv_key}") != _integer(raw.get(manifest_key), f"{strategy} manifest {manifest_key}"):
                raise ReportError(f"{job_dir.name}/{strategy}: CSV/manifest {csv_key} mismatch")
        if row.get("artifact_path") != raw.get("artifact_path") or row.get("artifact_sha256") != raw.get("artifact_sha256"):
            raise ReportError(f"{job_dir.name}/{strategy}: CSV/manifest artifact identity mismatch")
        if _integer(row.get("artifact_logical_payload_bits"), f"{strategy} logical bits", positive=True) != _integer(raw.get("artifact_logical_payload_bits"), f"{strategy} manifest logical bits", positive=True):
            raise ReportError(f"{job_dir.name}/{strategy}: logical payload mismatch")
        if _integer(row.get("q_scale_count"), f"{strategy} q_scale_count", positive=True) != inspection.q_scale_count:
            raise ReportError(
                f"{job_dir.name}/{strategy}: q_scale_count does not match HRC q_scales"
            )
        if _integer(row.get("sparse_nnz"), f"{strategy} sparse_nnz") != inspection.sparse_nnz:
            raise ReportError(f"{job_dir.name}/{strategy}: sparse_nnz does not match HRC")
        if _integer(row.get("lowrank_rank_sum"), f"{strategy} lowrank rank") != inspection.lowrank_rank_sum:
            raise ReportError(
                f"{job_dir.name}/{strategy}: lowrank_rank_sum does not match HRC"
            )
        _close(_finite(row.get("artifact_to_reference_file_ratio"), f"{strategy} CSV rate"), rate, f"{strategy} CSV physical rate")
        _close(_finite(row.get("artifact_physical_compression_ratio"), f"{strategy} CSV compression"), 1.0 / rate, f"{strategy} CSV compression ratio")
        if _boolean(row.get("heldout_evaluated"), f"{strategy} heldout flag") is not True:
            raise ReportError(f"{job_dir.name}/{strategy}: held-out evaluation is absent")
        if _integer(row.get("heldout_tokens"), f"{strategy} heldout tokens", positive=True) != eval_tokens:
            raise ReportError(f"{job_dir.name}/{strategy}: endpoint token count mismatch")
        heldout_nll = _finite(row.get("heldout_nll"), f"{strategy} NLL")
        heldout_ppl = _finite(row.get("heldout_perplexity"), f"{strategy} perplexity")
        nll_delta = _finite(row.get("nll_delta"), f"{strategy} NLL delta")
        ppl_delta = _finite(row.get("perplexity_delta"), f"{strategy} perplexity delta")
        _close(heldout_nll - baseline_nll, nll_delta, f"{strategy} NLL delta")
        _close(math.exp(heldout_nll), heldout_ppl, f"{strategy} exp(NLL)", rel_tol=2e-9)
        _close(heldout_ppl - baseline_ppl, ppl_delta, f"{strategy} perplexity delta")
        normalized_hessian = _finite(
            row.get("normalized_hessian_cost"), f"{strategy} normalized_hessian_cost"
        )
        hessian_cost = _finite(row.get("hessian_cost"), f"{strategy} hessian_cost")
        baseline_hessian = _finite(
            row.get("baseline_hessian_energy"), f"{strategy} baseline_hessian_energy"
        )
        if baseline_hessian <= 0.0 or hessian_cost < 0.0 or normalized_hessian < 0.0:
            raise ReportError(
                f"{job_dir.name}/{strategy}: PSD-proxy energy/cost must be non-negative with positive baseline"
            )
        _close(
            normalized_hessian,
            2.0 * hessian_cost / baseline_hessian,
            f"{strategy} normalized Hessian formula",
            rel_tol=2e-8,
            abs_tol=1e-11,
        )
        self_terms = {
            suffix: _finite(row.get(f"hessian_self_{suffix}"), f"{strategy} self {suffix}")
            for suffix in ("q", "s", "l")
        }
        if any(value < 0.0 for value in self_terms.values()):
            raise ReportError(f"{job_dir.name}/{strategy}: PSD self-energy is negative")
        cross_terms = {
            suffix: _finite(row.get(f"hessian_cross_{suffix}"), f"{strategy} cross {suffix}")
            for suffix in ("qs", "ql", "sl")
        }
        decomposed = sum(self_terms.values()) + sum(cross_terms.values())
        decomposition_scale = abs(hessian_cost) + sum(
            abs(value) for value in (*self_terms.values(), *cross_terms.values())
        )
        decomposition_bound = (
            HESSIAN_DECOMPOSITION_ULP_FACTOR * FLOAT32_EPSILON * decomposition_scale
        )
        if abs(hessian_cost - decomposed) > decomposition_bound:
            raise ReportError(
                f"{job_dir.name}/{strategy}: Hessian self/cross decomposition exceeds "
                f"the {HESSIAN_DECOMPOSITION_ULP_FACTOR:g}-ulp float32 reduction bound"
            )
        if row.get("artifact_scope") != PAYLOAD_SCOPE or _boolean(row.get("production_backend"), f"{strategy} production flag"):
            raise ReportError(f"{job_dir.name}/{strategy}: endpoint scope/backend mismatch")
        if not _boolean(row.get("roundtrip_exact_fp16_endpoint"), f"{strategy} roundtrip"):
            raise ReportError(f"{job_dir.name}/{strategy}: CSV roundtrip flag is false")
        rho_geometry = {
            "qs": (cross_terms["qs"], self_terms["q"], self_terms["s"]),
            "ql": (cross_terms["ql"], self_terms["q"], self_terms["l"]),
            "sl": (cross_terms["sl"], self_terms["s"], self_terms["l"]),
        }
        for suffix in ("sl", "qs", "ql"):
            cross, self_left, self_right = rho_geometry[suffix]
            cauchy_bound = math.sqrt(
                max(2.0 * self_left, 0.0) * max(2.0 * self_right, 0.0)
            )
            cauchy_tolerance = max(1e-8, 2e-6 * cauchy_bound)
            if abs(cross) > cauchy_bound + cauchy_tolerance:
                raise ReportError(
                    f"{job_dir.name}/{strategy}: cross {suffix} violates the PSD Cauchy bound"
                )
            rho = _optional_float(row.get(f"rho_{suffix}"), f"{strategy} rho_{suffix}")
            recomputed_rho = _rho_from_geometry(*rho_geometry[suffix])
            if math.isnan(recomputed_rho):
                if not math.isnan(rho):
                    raise ReportError(
                        f"{job_dir.name}/{strategy}: rho_{suffix} must be inactive for zero self-energy"
                    )
            else:
                if math.isnan(rho):
                    raise ReportError(
                        f"{job_dir.name}/{strategy}: rho_{suffix} is inactive despite nonzero self-energy"
                    )
                _close(rho, recomputed_rho, f"{strategy} rho_{suffix}", abs_tol=1e-12)
            expected_kind = _rho_kind(rho, rho_threshold)
            if row.get(f"rho_{suffix}_kind") != expected_kind:
                raise ReportError(f"{job_dir.name}/{strategy}: rho_{suffix} classification mismatch")
        for field in ("sparse_nnz", "lowrank_rank_sum", "folded_repair_dof"):
            if _integer(row.get(field), f"{strategy} {field}") < 0:
                raise ReportError(f"{job_dir.name}/{strategy}: {field} cannot be negative")
        if _integer(row.get("sparse_nnz"), f"{strategy} sparse_nnz") != inspection.sparse_nnz:
            raise ReportError(f"{job_dir.name}/{strategy}: HRC sparse_nnz mismatch")
        if _integer(row.get("lowrank_rank_sum"), f"{strategy} rank sum") != inspection.lowrank_rank_sum:
            raise ReportError(f"{job_dir.name}/{strategy}: HRC low-rank sum mismatch")
        codec_layers = inspection.manifest["layers"]
        sparse_active = sum(int(layer["sparse_nnz"]) > 0 for layer in codec_layers)
        lowrank_active = sum(int(layer["lowrank_rank"]) > 0 for layer in codec_layers)
        both_active = sum(
            int(layer["sparse_nnz"]) > 0 and int(layer["lowrank_rank"]) > 0
            for layer in codec_layers
        )
        for key, expected_count in (
            ("layers_s_active", sparse_active),
            ("layers_l_active", lowrank_active),
            ("layers_both_s_l_active", both_active),
        ):
            if _integer(row.get(key), f"{strategy} {key}") != expected_count:
                raise ReportError(f"{job_dir.name}/{strategy}: {key} does not match HRC")
        if strategy == "Q_global_scale":
            expected_repair_dof = len(codec_layers)
        elif strategy == "Q_block_scale":
            expected_repair_dof = inspection.q_scale_count
        elif strategy in {"Q+S_OBS", "Q+S_OBS+L"}:
            expected_repair_dof = inspection.sparse_nnz
        elif strategy in {STRICT, "Q+S+L_component_scale"}:
            expected_repair_dof = len(codec_layers) + sparse_active + lowrank_active
        else:
            expected_repair_dof = 0
        if _integer(row.get("folded_repair_dof"), f"{strategy} repair DoF") != expected_repair_dof:
            raise ReportError(
                f"{job_dir.name}/{strategy}: folded_repair_dof does not match stored-state reuse"
            )

    ql = artifact_by_strategy[QL]
    strict = artifact_by_strategy[STRICT]
    if _integer(
        artifact_by_strategy["Q_global_scale"]["artifact_file_bytes"],
        "global-scale bytes",
        positive=True,
    ) != _integer(artifact_by_strategy["Q"]["artifact_file_bytes"], "Q bytes", positive=True):
        raise ReportError(f"{job_dir.name}: folded global scale changed final Q file bytes")
    if _integer(
        artifact_by_strategy["Q+S_OBS"]["artifact_file_bytes"],
        "OBS sparse bytes",
        positive=True,
    ) != _integer(artifact_by_strategy["Q+S"]["artifact_file_bytes"], "Q+S bytes", positive=True):
        raise ReportError(f"{job_dir.name}: OBS refit changed final Q+S file bytes")
    ql_bytes = _integer(ql["artifact_file_bytes"], "Q+L bytes", positive=True)
    strict_bytes = _integer(strict["artifact_file_bytes"], "strict bytes", positive=True)
    if strict_bytes != ql_bytes:
        raise ReportError(f"{job_dir.name}: strict QSL and Q+L final artifact bytes differ")
    if _integer(strict.get("ql_budget_file_bytes"), "strict Q+L budget", positive=True) != ql_bytes:
        raise ReportError(f"{job_dir.name}: strict recorded Q+L byte budget differs from Q+L")
    if strict.get("same_physical_bytes_as_ql") is not True or strict.get("under_ql_serialized_cap_before_padding") is not True:
        raise ReportError(f"{job_dir.name}: strict artifact does not satisfy the declared Q+L cap")
    if _integer(strict.get("artifact_natural_file_bytes"), "strict natural bytes", positive=True) > ql_bytes:
        raise ReportError(f"{job_dir.name}: strict natural artifact exceeds Q+L final bytes")
    if inspection_by_strategy[QL].tail_padding_bytes != 0:
        raise ReportError(f"{job_dir.name}: Q+L contains unexpected tail padding")
    if inspection_by_strategy[STRICT].natural_file_bytes > inspection_by_strategy[QL].natural_file_bytes:
        raise ReportError(f"{job_dir.name}: strict natural artifact exceeds natural Q+L bytes")
    strict_row = by_strategy[STRICT]
    if not _boolean(strict_row.get("same_physical_bytes_as_ql"), "strict equal-byte flag"):
        raise ReportError(f"{job_dir.name}: strict CSV equal-byte flag is false")
    if not _boolean(strict_row.get("rate_cap_satisfied"), "strict rate cap flag"):
        raise ReportError(f"{job_dir.name}: strict CSV rate-cap flag is false")
    if not _boolean(strict_row.get("under_ql_serialized_cap_before_padding"), "strict natural cap flag"):
        raise ReportError(f"{job_dir.name}: strict CSV natural-cap flag is false")
    for suffix in ("sl", "qs", "ql"):
        if math.isnan(_optional_float(strict_row.get(f"rho_{suffix}"), f"strict rho_{suffix}")):
            raise ReportError(f"{job_dir.name}: strict rho_{suffix} is inactive")
    support_components = ("q_codes", "q_scales", "sparse_row_ptr", "sparse_col_idx")
    qs_inspection = inspection_by_strategy["Q+S"]
    obs_inspection = inspection_by_strategy["Q+S_OBS"]
    for layer in sorted(reference_inspection.layer_shapes):
        for component in support_components:
            stream = f"{layer}/{component}"
            if qs_inspection.component_sha256.get(stream) != obs_inspection.component_sha256.get(stream):
                raise ReportError(
                    f"{job_dir.name}: OBS changed already-paid {component} state"
                )
    return rows, by_strategy, manifest, input_hashes


def _validate_covariance_psd_audit(
    job_dir: Path, run_config: Mapping[str, Any], selected_layers: Sequence[str]
) -> str:
    path = job_dir / "covariance_psd_audit.csv"
    rows = _read_csv(path)
    expected_fields = {
        "diagonal_shift",
        "diagonal_shift_relative",
        "final_min_eigenvalue",
        "final_spectral_scale",
        "float32_storage_floor_rtol",
        "layer",
        "original_min_eigenvalue",
        "original_min_relative",
        "original_spectral_scale",
        "psd_rejection_rtol",
        "repair_applied",
    }
    if any(set(row) != expected_fields for row in rows):
        raise ReportError(f"{job_dir.name}: covariance PSD audit fields changed")
    names = [row.get("layer") for row in rows]
    if names != list(selected_layers) or len(names) != len(set(names)):
        raise ReportError(f"{job_dir.name}: covariance PSD rows do not exactly cover selected tensors")

    audit = run_config.get("covariance_psd_audit")
    if not isinstance(audit, dict):
        raise ReportError(f"{job_dir.name}: run covariance PSD audit metadata is missing")
    if audit.get("path") != "covariance_psd_audit.csv":
        raise ReportError(f"{job_dir.name}: covariance PSD audit path changed")
    if _integer(audit.get("layer_count"), "PSD audit layer count", positive=True) != len(rows):
        raise ReportError(f"{job_dir.name}: covariance PSD layer count mismatch")
    _close(
        _finite(audit.get("psd_rejection_rtol"), "PSD rejection rtol"),
        PSD_REJECTION_RTOL,
        "PSD rejection rtol",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    _close(
        _finite(audit.get("float32_storage_floor_rtol"), "PSD storage floor"),
        FLOAT32_PSD_FLOOR_RTOL,
        "PSD storage floor",
        rel_tol=0.0,
        abs_tol=0.0,
    )
    if audit.get("all_consumers_share_prepared_covariance") is not True:
        raise ReportError(f"{job_dir.name}: covariance is not shared by every consumer")

    negative_relatives: list[float] = []
    shift_relatives: list[float] = []
    for row in rows:
        layer = str(row["layer"])
        original_min = _finite(row["original_min_eigenvalue"], f"{layer} original minimum")
        original_scale = _finite(row["original_spectral_scale"], f"{layer} original scale")
        original_relative = _finite(row["original_min_relative"], f"{layer} original relative")
        final_min = _finite(row["final_min_eigenvalue"], f"{layer} final minimum")
        final_scale = _finite(row["final_spectral_scale"], f"{layer} final scale")
        shift = _finite(row["diagonal_shift"], f"{layer} diagonal shift")
        shift_relative = _finite(row["diagonal_shift_relative"], f"{layer} shift relative")
        rejection = _finite(row["psd_rejection_rtol"], f"{layer} rejection rtol")
        storage_floor = _finite(row["float32_storage_floor_rtol"], f"{layer} storage floor")
        if original_scale < 0.0 or final_scale < 0.0 or shift < 0.0 or final_min < 0.0:
            raise ReportError(f"{job_dir.name}/{layer}: invalid PSD scale, shift, or final spectrum")
        spectral_tolerance = PSD_FLOAT32_CLOSURE_RTOL * max(original_scale, final_scale)
        if abs(original_min) > original_scale + spectral_tolerance:
            raise ReportError(f"{job_dir.name}/{layer}: original minimum exceeds the spectral scale")
        if final_min > final_scale + spectral_tolerance:
            raise ReportError(f"{job_dir.name}/{layer}: final minimum exceeds the spectral scale")
        denominator = original_scale if original_scale > 0.0 else 1.0
        _close(original_relative, original_min / denominator, f"{layer} original relative")
        _close(shift_relative, shift / denominator, f"{layer} shift relative")
        _close(rejection, PSD_REJECTION_RTOL, f"{layer} rejection rtol", rel_tol=0.0, abs_tol=0.0)
        _close(storage_floor, FLOAT32_PSD_FLOOR_RTOL, f"{layer} storage floor", rel_tol=0.0, abs_tol=0.0)
        if original_min < -PSD_REJECTION_RTOL * original_scale:
            raise ReportError(f"{job_dir.name}/{layer}: materially indefinite original covariance")
        if original_scale == 0.0:
            if any(value != 0.0 for value in (original_min, final_min, final_scale, shift, shift_relative)):
                raise ReportError(f"{job_dir.name}/{layer}: the zero covariance was changed")
        else:
            expected_shift_relative = max(
                0.0, FLOAT32_PSD_FLOOR_RTOL - original_relative
            )
            if abs(shift_relative - expected_shift_relative) > PSD_FLOAT32_CLOSURE_RTOL:
                raise ReportError(
                    f"{job_dir.name}/{layer}: PSD shift exceeds the declared float32 floor repair"
                )
            final_scale_residual = abs(final_scale - (original_scale + shift))
            if final_scale_residual > PSD_FLOAT32_CLOSURE_RTOL * original_scale:
                raise ReportError(
                    f"{job_dir.name}/{layer}: final spectral scale does not close after the diagonal shift"
                )
            if final_min > shift + max(original_min, 0.0) + PSD_FLOAT32_CLOSURE_RTOL * original_scale:
                raise ReportError(
                    f"{job_dir.name}/{layer}: final minimum is inconsistent with a scalar diagonal repair"
                )
        if _boolean(row["repair_applied"], f"{layer} repair flag") != (shift > 0.0):
            raise ReportError(f"{job_dir.name}/{layer}: PSD repair flag mismatch")
        negative_relatives.append(max(0.0, -original_relative))
        shift_relatives.append(max(0.0, shift_relative))

    _close(
        _finite(audit.get("maximum_original_negative_relative"), "maximum negative PSD relative"),
        max(negative_relatives, default=0.0),
        "maximum negative PSD relative",
    )
    _close(
        _finite(audit.get("maximum_diagonal_shift_relative"), "maximum PSD shift relative"),
        max(shift_relatives, default=0.0),
        "maximum PSD shift relative",
    )
    return _file_sha256(path)


def _validate_windows(
    job_dir: Path,
    run_config: dict[str, Any],
    endpoints: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, dict[int, tuple[int, float, float, int, int]]], list[PairStats], str]:
    path = job_dir / "endpoint_window_nll.csv"
    rows = _read_csv(path)
    expected_names = ("dense", *EXPECTED_STRATEGIES)
    grouped: dict[str, dict[int, tuple[int, float, float, int, int]]] = {
        name: {} for name in expected_names
    }
    data = run_config.get("data")
    if not isinstance(data, dict):
        raise ReportError(f"{job_dir.name}: missing data provenance")
    window_count = _integer(data.get("eval_window_count"), "eval window count", positive=True)
    if window_count != 8:
        raise ReportError(f"{job_dir.name}: expected eight fixed held-out windows")
    arguments = run_config.get("arguments")
    if not isinstance(arguments, dict):
        raise ReportError(f"{job_dir.name}: run arguments are absent")
    expected_window_tokens = (
        _integer(arguments.get("sequence_length"), "sequence length", positive=True) - 1
    ) * _integer(arguments.get("batch_size"), "batch size", positive=True)
    if expected_window_tokens <= 0:
        raise ReportError(f"{job_dir.name}: fixed window has no predicted tokens")
    for row in rows:
        strategy = str(row.get("strategy"))
        if strategy not in grouped:
            raise ReportError(f"{job_dir.name}: unexpected window strategy {strategy!r}")
        index = _integer(row.get("window_index"), f"{strategy} window index")
        if index in grouped[strategy]:
            raise ReportError(f"{job_dir.name}: duplicate ({strategy}, {index}) window")
        tokens = _integer(row.get("tokens"), f"{strategy} window tokens", positive=True)
        if tokens != expected_window_tokens:
            raise ReportError(
                f"{job_dir.name}: {strategy} window {index} does not contain "
                f"the declared {expected_window_tokens} predicted tokens"
            )
        batch_index = _integer(row.get("batch_index"), f"{strategy} batch index")
        sequence_index = _integer(row.get("sequence_index"), f"{strategy} sequence index")
        if batch_index != index or sequence_index != 0:
            raise ReportError(
                f"{job_dir.name}: ({strategy}, {index}) batch/sequence identity changed"
            )
        nll_sum = _finite(row.get("nll_sum"), f"{strategy} window NLL sum")
        nll = _finite(row.get("nll"), f"{strategy} window NLL")
        ppl = _finite(row.get("perplexity"), f"{strategy} window perplexity")
        _close(nll_sum / tokens, nll, f"{strategy} window {index} NLL")
        _close(math.exp(nll), ppl, f"{strategy} window {index} perplexity", rel_tol=2e-9)
        grouped[strategy][index] = (tokens, nll_sum, nll, batch_index, sequence_index)
    required_indices = set(range(window_count))
    dense_tokens = None
    dense_identity = None
    for strategy, group in grouped.items():
        if set(group) != required_indices:
            raise ReportError(f"{job_dir.name}: {strategy} does not have exactly windows 0..7")
        token_vector = tuple(group[index][0] for index in range(window_count))
        identity_vector = tuple(
            (group[index][3], group[index][4], index) for index in range(window_count)
        )
        if dense_tokens is None:
            dense_tokens = token_vector
            dense_identity = identity_vector
        elif token_vector != dense_tokens:
            raise ReportError(f"{job_dir.name}: {strategy} window tokens do not match dense")
        elif identity_vector != dense_identity:
            raise ReportError(f"{job_dir.name}: {strategy} window identities do not match dense")

    actual_eval_tokens = _integer(
        run_config.get("actual_eval_tokens"), f"{job_dir.name} actual eval tokens", positive=True
    )
    if dense_tokens is None or sum(dense_tokens) != actual_eval_tokens:
        raise ReportError(
            f"{job_dir.name}: fixed-window token total does not equal actual_eval_tokens"
        )

    baseline = run_config["baseline_metrics"]
    dense_nll = sum(item[1] for item in grouped["dense"].values()) / sum(
        item[0] for item in grouped["dense"].values()
    )
    _close(dense_nll, _finite(baseline.get("nll"), "baseline NLL"), f"{job_dir.name} dense aggregate")
    for strategy in EXPECTED_STRATEGIES:
        group = grouped[strategy]
        aggregate = sum(item[1] for item in group.values()) / sum(item[0] for item in group.values())
        endpoint = endpoints[strategy]
        _close(aggregate, _finite(endpoint.get("heldout_nll"), f"{strategy} heldout NLL"), f"{job_dir.name}/{strategy} window aggregate")
        deltas = [group[index][2] - grouped["dense"][index][2] for index in range(window_count)]
        mean = statistics.fmean(deltas)
        se = statistics.stdev(deltas) / math.sqrt(window_count)
        _close(mean, _finite(endpoint.get("paired_window_nll_delta_mean"), f"{strategy} paired mean"), f"{strategy} paired mean")
        _close(se, _finite(endpoint.get("paired_window_nll_delta_se"), f"{strategy} paired SE"), f"{strategy} paired SE")
        _close(mean - 1.96 * se, _finite(endpoint.get("paired_window_nll_delta_ci95_low"), f"{strategy} paired low"), f"{strategy} paired low")
        _close(mean + 1.96 * se, _finite(endpoint.get("paired_window_nll_delta_ci95_high"), f"{strategy} paired high"), f"{strategy} paired high")
        if _integer(endpoint.get("paired_window_count"), f"{strategy} paired count") != window_count:
            raise ReportError(f"{job_dir.name}/{strategy}: paired window count mismatch")

    pairs: list[PairStats] = []
    for comparison_id, left, right in COMPARISONS:
        differences = [
            grouped[left][index][2] - grouped[right][index][2]
            for index in range(window_count)
        ]
        mean = statistics.fmean(differences)
        se = statistics.stdev(differences) / math.sqrt(window_count)
        pairs.append(
            PairStats(
                comparison_id=comparison_id,
                left=left,
                right=right,
                count=window_count,
                mean=mean,
                standard_error=se,
                interval_low=mean - 1.96 * se,
                interval_high=mean + 1.96 * se,
                left_wins=sum(
                    grouped[left][index][2] < grouped[right][index][2]
                    for index in range(window_count)
                ),
            )
        )
    return grouped, pairs, _file_sha256(path)


def _validate_comfort(
    job_dir: Path,
    config: dict[str, Any],
    run_config: dict[str, Any],
    endpoints: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
    sweep_path = job_dir / "comfort_sweep.csv"
    summary_path = job_dir / "comfort_summary.csv"
    rows = _read_csv(sweep_path)
    summaries = _read_csv(summary_path)
    common = config["common"]
    strategies = list(common["comfort_strategies"])
    epsilons = [float(value) for value in common["comfort_epsilons"]]
    expected_pairs = [(strategy, epsilon) for strategy in strategies for epsilon in epsilons]
    observed_pairs = [
        (str(row.get("strategy")), _finite(row.get("epsilon"), "comfort epsilon"))
        for row in rows
    ]
    if observed_pairs != expected_pairs:
        raise ReportError(f"{job_dir.name}: comfort sweep strategy/epsilon grid or order changed")
    if [row.get("strategy") for row in summaries] != strategies:
        raise ReportError(f"{job_dir.name}: comfort summary strategy order/set changed")
    summary_by_strategy = {str(row["strategy"]): row for row in summaries}
    baseline_nll = _finite(run_config["baseline_metrics"]["nll"], "baseline NLL")
    baseline_ppl = _finite(run_config["baseline_metrics"]["perplexity"], "baseline perplexity")
    eval_tokens = _integer(run_config["actual_eval_tokens"], "eval tokens", positive=True)
    fit_max = float(common["comfort_fit_max_epsilon"])

    for strategy in strategies:
        group = [row for row in rows if row["strategy"] == strategy]
        proxy: list[float] = []
        task: list[float] = []
        for row, epsilon in zip(group, epsilons):
            if not math.isclose(_finite(row.get("target_ratio"), "comfort target"), 0.258, rel_tol=0.0, abs_tol=1e-12):
                raise ReportError(f"{job_dir.name}/{strategy}: comfort target changed")
            if _integer(row.get("tokens"), "comfort tokens", positive=True) != eval_tokens:
                raise ReportError(f"{job_dir.name}/{strategy}: comfort token count mismatch")
            nll = _finite(row.get("nll"), "comfort NLL")
            ppl = _finite(row.get("perplexity"), "comfort perplexity")
            nll_delta = _finite(row.get("nll_delta"), "comfort NLL delta")
            ppl_delta = _finite(row.get("perplexity_delta"), "comfort perplexity delta")
            _close(nll - baseline_nll, nll_delta, f"{strategy} comfort NLL delta")
            _close(math.exp(nll), ppl, f"{strategy} comfort exp(NLL)", rel_tol=2e-9)
            _close(ppl - baseline_ppl, ppl_delta, f"{strategy} comfort PPL delta")
            normalized = _finite(row.get("normalized_hessian_cost"), "comfort proxy cost")
            hessian_cost = _finite(row.get("hessian_cost"), "comfort Hessian cost")
            endpoint_normalized = _finite(
                endpoints[strategy].get("normalized_hessian_cost"),
                f"{strategy} endpoint normalized Hessian",
            )
            endpoint_hessian = _finite(
                endpoints[strategy].get("hessian_cost"), f"{strategy} endpoint Hessian cost"
            )
            _close(
                normalized,
                epsilon * epsilon * endpoint_normalized,
                f"{strategy} path normalized Hessian epsilon={epsilon}",
                rel_tol=2e-8,
                abs_tol=1e-11,
            )
            _close(
                hessian_cost,
                epsilon * epsilon * endpoint_hessian,
                f"{strategy} path Hessian epsilon={epsilon}",
                rel_tol=2e-8,
                abs_tol=1e-9,
            )
            proxy.append(normalized)
            task.append(nll_delta)
            if epsilon == 0.0:
                _close(nll_delta, 0.0, f"{strategy} epsilon-zero NLL", abs_tol=1e-11)
                _close(normalized, 0.0, f"{strategy} epsilon-zero proxy", abs_tol=1e-11)
            expected_kind = "codec_endpoint" if epsilon == 1.0 else "noncodec_interpolation"
            if row.get("path_kind") != expected_kind or _boolean(row.get("deployable"), "comfort deployable") != (epsilon == 1.0):
                raise ReportError(f"{job_dir.name}/{strategy}: path/deployability label mismatch")
        endpoint_delta = _finite(endpoints[strategy]["nll_delta"], f"{strategy} endpoint NLL")
        _close(task[-1], endpoint_delta, f"{strategy} epsilon-one endpoint")
        summary = summary_by_strategy[strategy]
        if not math.isclose(_finite(summary.get("small_epsilon_fit_max"), "fit maximum"), fit_max, rel_tol=0.0, abs_tol=1e-12):
            raise ReportError(f"{job_dir.name}/{strategy}: fit interval changed")
        fit_pairs = [(epsilon, value) for epsilon, value in zip(epsilons, task) if 0.0 < epsilon <= fit_max]
        linear, quadratic = _fit_linear_quadratic(
            [pair[0] for pair in fit_pairs], [pair[1] for pair in fit_pairs]
        )
        _close(linear, _finite(summary.get("taylor_linear_coefficient"), "Taylor linear"), f"{strategy} Taylor linear", rel_tol=2e-8)
        _close(quadratic, _finite(summary.get("taylor_quadratic_coefficient"), "Taylor quadratic"), f"{strategy} Taylor quadratic", rel_tol=2e-8)
        for row, epsilon in zip(group, epsilons):
            prediction = linear * epsilon + quadratic * epsilon * epsilon
            _close(prediction, _finite(row.get("taylor_fit_nll_delta"), "Taylor path value"), f"{strategy} Taylor path epsilon={epsilon}", rel_tol=2e-8)
            actual = _finite(row.get("nll_delta"), "Taylor path measured NLL")
            _close(
                _finite(
                    row.get("taylor_fit_absolute_error"),
                    "Taylor path absolute error",
                ),
                abs(actual - prediction),
                f"{strategy} Taylor absolute error epsilon={epsilon}",
                rel_tol=2e-8,
                abs_tol=1e-11,
            )
        correlation = _pearson(proxy, task)
        _close(correlation, _finite(summary.get("hessian_proxy_nll_correlation"), "proxy correlation"), f"{strategy} proxy correlation", rel_tol=2e-8)
        _close(endpoint_delta, _finite(summary.get("codec_endpoint_nll_delta"), "summary endpoint NLL"), f"{strategy} summary endpoint")

    return rows, summary_by_strategy, {
        sweep_path.as_posix(): _file_sha256(sweep_path),
        summary_path.as_posix(): _file_sha256(summary_path),
    }


def _runner_evidence(
    stage: dict[str, Any], run_config: dict[str, Any], artifact_manifest: dict[str, Any]
) -> dict[str, Any]:
    selected = run_config["selected_layers"]
    activation_values = list(run_config["activation_counts"].values())
    reference = artifact_manifest["reference"]
    strategies = artifact_manifest["strategies"]
    runtime = run_config["runtime"]
    data = run_config["data"]
    return {
        "payload_scope": PAYLOAD_SCOPE,
        "scope_claim": stage["tensor_scope"]["claim_scope"],
        "selected_tensor_count": len(selected),
        "selected_parameter_count": int(run_config["selected_parameter_count"]),
        "model_parameter_count": int(run_config["model_parameter_count"]),
        "actual_eval_tokens": int(run_config["actual_eval_tokens"]),
        "actual_calibration_activation_tokens_min": min(activation_values),
        "actual_calibration_activation_tokens_max": max(activation_values),
        "calib_text_count": int(data.get("calib_text_count", 0)),
        "eval_text_count": int(data.get("eval_text_count", 0)),
        "content_disjoint": data.get("content_disjoint"),
        "calib_digest": data.get("calib_digest"),
        "eval_digest": data.get("eval_digest"),
        "reference_artifact": {
            "path": reference["path"],
            "file_bytes": reference["file_bytes"],
            "sha256": reference["sha256"],
        },
        "strategy_artifacts": {
            item["strategy"]: {
                "path": item["artifact_path"],
                "file_bytes": item["artifact_file_bytes"],
                "sha256": item["artifact_sha256"],
            }
            for item in strategies
        },
        "runtime": {
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "torch": runtime.get("torch"),
            "transformers": runtime.get("transformers"),
            "datasets": runtime.get("datasets"),
            "numpy": runtime.get("numpy"),
            "cuda_available": runtime.get("cuda_available"),
            "cuda_device": runtime.get("cuda_device"),
        },
        "model_identity": run_config.get("model_identity"),
    }


def _validate_job(
    config: dict[str, Any],
    suite_root: Path,
    stage: dict[str, Any],
    entry: dict[str, Any],
    *,
    config_sha: str,
    source_snapshot: dict[str, dict[str, Any]],
    source_sha: str,
) -> ValidatedJob:
    stage_id = str(stage["id"])
    job_id = _job_id(stage_id)
    if entry.get("job_id") != job_id or entry.get("stage_id") != stage_id:
        raise ReportError(f"{stage_id}: suite manifest job identity mismatch")
    expected_entry = {
        "evidence_role": EVIDENCE_ROLE,
        "protocol_manifest_consumed": False,
        "seed_aggregation_allowed": False,
        "data_window_independence": "shared_sequential_windows_not_independent_across_seeds",
        "model_declared": stage["model"],
        "model_scale": stage["model_scale"],
        "seed": 17,
        "target_rate": 0.258,
        "suite_config_sha256": config_sha,
        "numerical_source_sha256": source_sha,
        "status": "completed_valid",
        "exit_code": 0,
    }
    for key, expected in expected_entry.items():
        if not _match(entry.get(key), expected):
            raise ReportError(f"{job_id}: suite manifest {key} mismatch")
    if entry.get("tensor_scope") != stage["tensor_scope"]:
        raise ReportError(f"{job_id}: suite manifest tensor scope mismatch")
    if not isinstance(entry.get("model_argument"), str) or not entry["model_argument"]:
        raise ReportError(f"{job_id}: missing resolved model argument")
    expected_job_sha = _expected_job_hash(config, stage, entry)
    if entry.get("job_config_sha256") != expected_job_sha:
        raise ReportError(f"{job_id}: job config SHA does not match the suite contract")
    expected_fingerprint = _object_sha256(
        {
            "suite_config_sha256": config_sha,
            "job_config_sha256": expected_job_sha,
            "numerical_source_sha256": source_sha,
        }
    )
    if entry.get("execution_fingerprint_sha256") != expected_fingerprint:
        raise ReportError(f"{job_id}: execution fingerprint mismatch")

    job_dir = suite_root / "jobs" / job_id
    if not job_dir.is_dir():
        raise ReportError(f"missing job directory: {job_dir}")
    if (job_dir / "RUNNING").exists() or (job_dir / "FAILED").exists():
        raise ReportError(f"{job_id}: terminal failure/running marker is present")
    _safe_file(job_dir, "COMPLETED")
    for relative in config["expected_outputs"]:
        _safe_file(job_dir, relative)

    record_path = job_dir / "_suite_job_record.json"
    state_path = suite_root / "_state" / f"{job_id}.json"
    record = _read_json(record_path)
    state = _read_json(state_path)
    critical = {
        "schema_version": JOB_RECORD_SCHEMA,
        "job_id": job_id,
        "suite_id": SUITE_ID,
        "evidence_role": EVIDENCE_ROLE,
        "protocol_manifest_consumed": False,
        "seed_aggregation_allowed": False,
        "suite_config_sha256": config_sha,
        "job_config_sha256": expected_job_sha,
        "numerical_source_sha256": source_sha,
        "execution_fingerprint_sha256": expected_fingerprint,
        "status": "COMPLETED",
        "exit_code": 0,
    }
    for name, raw in (("job record", record), ("state record", state)):
        for key, expected in critical.items():
            if raw.get(key) != expected:
                raise ReportError(f"{job_id}: {name} {key} mismatch")
    for key in ("evidence_sha256", "stdout", "stderr"):
        if record.get(key) != state.get(key):
            raise ReportError(f"{job_id}: state/job record {key} mismatch")
    _validate_log(suite_root, record, "stdout", job_id)
    _validate_log(suite_root, record, "stderr", job_id)

    run_path = job_dir / "run_config.json"
    run_config = _read_json(run_path)
    if run_config.get("model") != entry["model_argument"]:
        raise ReportError(f"{job_id}: run model differs from resolved suite model")
    if run_config.get("payload_scope") != PAYLOAD_SCOPE:
        raise ReportError(f"{job_id}: run payload scope changed")
    if run_config.get("source_snapshot") != source_snapshot:
        raise ReportError(f"{job_id}: run source snapshot differs from suite source")
    arguments = run_config.get("arguments")
    if not isinstance(arguments, dict):
        raise ReportError(f"{job_id}: run arguments are absent")
    effective = dict(config["common"])
    effective.update(
        {
            "model": entry["model_argument"],
            "revision": str(stage.get("revision", "")),
            "module_types": list(stage["tensor_scope"]["module_types"]),
            "layers": list(stage["tensor_scope"]["layers"]),
            "max_modules": int(stage["tensor_scope"]["max_modules"]),
            "target_ratios": [0.258],
            "endpoint_target": 0.258,
            "seed": 17,
        }
    )
    for key, expected in effective.items():
        if key in {"device", "svd_device"} or (key == "revision" and expected == ""):
            continue
        if not _match(arguments.get(key), expected):
            raise ReportError(f"{job_id}: run argument {key} mismatch")
    selected = run_config.get("selected_layers")
    expected_count = int(stage["tensor_scope"]["expected_selected_tensors"])
    if not isinstance(selected, list) or len(selected) != expected_count or len(set(selected)) != expected_count:
        raise ReportError(f"{job_id}: selected tensor count/uniqueness mismatch")
    allowed_layers = set(stage["tensor_scope"]["layers"])
    allowed_modules = set(stage["tensor_scope"]["module_types"])
    observed_scope_pairs: list[tuple[int, str]] = []
    for name in selected:
        if not isinstance(name, str) or _layer_index(name) not in allowed_layers or name.rsplit(".", 1)[-1] not in allowed_modules:
            raise ReportError(f"{job_id}: tensor outside declared scope: {name!r}")
        observed_scope_pairs.append((_layer_index(name), name.rsplit(".", 1)[-1]))
    expected_scope_pairs = {
        (int(layer), str(module))
        for layer in stage["tensor_scope"]["layers"]
        for module in stage["tensor_scope"]["module_types"]
    }
    if (
        len(observed_scope_pairs) != len(set(observed_scope_pairs))
        or set(observed_scope_pairs) != expected_scope_pairs
    ):
        raise ReportError(f"{job_id}: selected tensors do not exactly cover layer x module scope")
    activations = run_config.get("activation_counts")
    if not isinstance(activations, dict) or set(activations) != set(selected):
        raise ReportError(f"{job_id}: activation counts do not exactly cover selected tensors")
    if any(_integer(value, "activation count", positive=True) <= 0 for value in activations.values()):
        raise ReportError(f"{job_id}: invalid activation count")
    selected_parameters = _integer(run_config.get("selected_parameter_count"), "selected parameters", positive=True)
    model_parameters = _integer(run_config.get("model_parameter_count"), "model parameters", positive=True)
    if selected_parameters >= model_parameters:
        raise ReportError(f"{job_id}: selected parameter scope is not a strict model subset")
    actual_tokens = _integer(run_config.get("actual_eval_tokens"), "actual eval tokens", positive=True)
    planned_tokens = int(config["common"]["eval_limit"]) * (int(config["common"]["sequence_length"]) - 1) * int(config["common"]["batch_size"])
    if actual_tokens != planned_tokens:
        raise ReportError(f"{job_id}: actual evaluation tokens differ from the declared plan")
    baseline = run_config.get("baseline_metrics")
    if not isinstance(baseline, dict) or _integer(baseline.get("tokens"), "baseline tokens", positive=True) != actual_tokens:
        raise ReportError(f"{job_id}: baseline token count mismatch")
    data = run_config.get("data")
    if not isinstance(data, dict) or data.get("content_disjoint") is not True:
        raise ReportError(f"{job_id}: calibration/evaluation content is not declared disjoint")
    common = config["common"]
    requested_data = {
        "dataset": common["dataset"],
        "subset": common["subset"],
        "split": common["split"],
        "backup_name": "",
        "sequence_length": int(common["sequence_length"]),
        "batch_size": int(common["batch_size"]),
        "allow_fallback": False,
    }
    if data.get("requested") != requested_data or data.get("source_used") != "dataset:wikitext":
        raise ReportError(f"{job_id}: WikiText-2 raw validation source contract changed")
    texts_per_window = int(common["texts_per_batch_window"])
    calibration_texts = int(common["calib_limit"]) * texts_per_window
    evaluation_texts = int(common["eval_limit"]) * texts_per_window
    recovery_texts = texts_per_window
    required_unique_texts = calibration_texts + evaluation_texts + recovery_texts
    requested_pool_texts = required_unique_texts + max(64, texts_per_window * 4)
    expected_source_metadata = [
        {
            "source": "dataset",
            "dataset": common["dataset"],
            "subset": common["subset"],
            "split": common["split"],
            "backup_name": "",
            "rows_requested": requested_pool_texts,
        }
    ]
    if data.get("source_metadata") != expected_source_metadata:
        raise ReportError(f"{job_id}: dataset source metadata changed")
    if data.get("fallback_allowed") is not False:
        raise ReportError(f"{job_id}: dataset fallback was enabled")
    for key, expected_value in (
        ("text_pool_count", requested_pool_texts),
        ("unique_text_pool_count", required_unique_texts),
        ("calib_text_count", calibration_texts),
        ("eval_text_count", evaluation_texts),
        ("eval_window_count", int(common["eval_limit"])),
        ("identical_text_overlap_count", 0),
    ):
        if _integer(data.get(key), f"{job_id} data {key}") != expected_value:
            raise ReportError(f"{job_id}: data {key} differs from the pilot contract")
    calibration_digest = _require_sha(data.get("calib_digest"), f"{job_id} calibration digest")
    evaluation_digest = _require_sha(data.get("eval_digest"), f"{job_id} evaluation digest")
    if calibration_digest == evaluation_digest:
        raise ReportError(f"{job_id}: calibration and evaluation digests are identical")
    if data.get("split_policy") != "content_disjoint_sequential_text_windows":
        raise ReportError(f"{job_id}: data split policy changed")
    if data.get("window_interval_semantics") != "paired fixed-window mean +/- 1.96 standard errors; descriptive, not an independence-based population CI":
        raise ReportError(f"{job_id}: fixed-window interval disclosure changed")
    if not isinstance(run_config.get("runtime"), dict):
        raise ReportError(f"{job_id}: runtime provenance is absent")
    if not isinstance(run_config.get("model_identity"), dict):
        raise ReportError(f"{job_id}: resolved model identity is absent")
    if (
        run_config["model_identity"].get("resolved_model_commit_hash")
        != EXPECTED_STAGE_MATERIAL[stage_id]["resolved_commit"]
    ):
        raise ReportError(f"{job_id}: resolved model commit differs from the pinned pilot checkpoint")

    endpoint_rows, endpoint_map, artifacts, artifact_hashes = _validate_endpoint_rows(
        job_dir, run_config, EXPECTED_STRATEGIES
    )
    psd_audit_sha = _validate_covariance_psd_audit(job_dir, run_config, selected)
    windows, pairs, window_sha = _validate_windows(job_dir, run_config, endpoint_map)
    comfort_rows, comfort_summary, comfort_hashes = _validate_comfort(
        job_dir, config, run_config, endpoint_map
    )
    evidence = _runner_evidence(stage, run_config, artifacts)
    if record.get("evidence_sha256") != _object_sha256(evidence):
        raise ReportError(f"{job_id}: suite job evidence hash mismatch")
    if entry.get("actual") != evidence:
        raise ReportError(f"{job_id}: suite manifest actual evidence differs from job files")

    hashes = {
        run_path.as_posix(): _file_sha256(run_path),
        record_path.as_posix(): _file_sha256(record_path),
        state_path.as_posix(): _file_sha256(state_path),
        (job_dir / "candidate_ablation.csv").as_posix(): _file_sha256(
            job_dir / "candidate_ablation.csv"
        ),
        (job_dir / "endpoint_window_nll.csv").as_posix(): window_sha,
        (job_dir / "covariance_psd_audit.csv").as_posix(): psd_audit_sha,
        **artifact_hashes,
        **comfort_hashes,
    }
    return ValidatedJob(
        stage=stage,
        manifest_entry=entry,
        run_config=run_config,
        endpoint_rows=endpoint_rows,
        endpoint_by_strategy=endpoint_map,
        artifact_manifest=artifacts,
        windows=windows,
        comfort_rows=comfort_rows,
        comfort_summary=comfort_summary,
        pairs=pairs,
        input_hashes=hashes,
    )


def validate_inputs(
    config_path: Path,
    suite_root: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[ValidatedJob], dict[str, str], str]:
    config_path = config_path.resolve()
    suite_root = suite_root.resolve()
    repo_root = repo_root.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    # Suite config identity is line-ending invariant so a Windows CRLF checkout
    # can audit the immutable Linux-produced manifest.
    config_sha = hashlib.sha256(
        config_path.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    suite_manifest_path = suite_root / "suite_manifest.json"
    suite_manifest = _read_json(suite_manifest_path)
    if suite_manifest.get("schema_version") != SUITE_MANIFEST_SCHEMA or suite_manifest.get("suite_id") != SUITE_ID:
        raise ReportError("suite manifest identity/schema mismatch")
    if suite_manifest.get("suite_config_sha256") != config_sha:
        raise ReportError("suite manifest config SHA differs from the supplied config")
    method_contract = suite_manifest.get("method_contract")
    if not isinstance(method_contract, dict) or tuple(method_contract.get("expected_strategies", ())) != EXPECTED_STRATEGIES:
        raise ReportError("suite manifest method contract changed")
    expected_counts = {"completed_valid": 3, "failed": 0, "invalid": 0, "planned": 0, "running": 0}
    if suite_manifest.get("status_counts") != expected_counts:
        raise ReportError("suite manifest does not contain exactly three valid completed jobs")
    source_snapshot = _validate_source_snapshot(suite_manifest.get("numerical_source_snapshot"), repo_root)
    source_sha = _object_sha256(source_snapshot)
    entries = suite_manifest.get("jobs")
    if not isinstance(entries, list) or [entry.get("job_id") for entry in entries if isinstance(entry, dict)] != [_job_id(stage) for stage in EXPECTED_STAGES]:
        raise ReportError("suite manifest job order/set differs from the pilot contract")
    jobs = [
        _validate_job(
            config,
            suite_root,
            stage,
            entry,
            config_sha=config_sha,
            source_snapshot=source_snapshot,
            source_sha=source_sha,
        )
        for stage, entry in zip(config["stages"], entries)
    ]
    raw_hashes = {
        config_path.as_posix(): config_sha,
        suite_manifest_path.as_posix(): _file_sha256(suite_manifest_path),
    }
    for source in source_snapshot.values():
        source_path = (repo_root / str(source["path"])).resolve()
        raw_hashes[source_path.as_posix()] = str(source["sha256"])
    for job in jobs:
        raw_hashes.update(job.input_hashes)
    hashes: dict[str, str] = {}
    for raw_path, sha in raw_hashes.items():
        path = Path(raw_path).resolve()
        try:
            key = path.relative_to(repo_root).as_posix()
        except ValueError:
            try:
                key = "suite/" + path.relative_to(suite_root).as_posix()
            except ValueError as exc:
                raise ReportError(f"evidence path is outside both repository and suite roots: {path}") from exc
        if key in hashes and hashes[key] != sha:
            raise ReportError(f"conflicting evidence hashes for {key}")
        hashes[key] = sha
    hashes["paper/results/build_scaling_pilot.py"] = _file_sha256(Path(__file__).resolve())
    return config, jobs, hashes, source_sha


ENDPOINT_FIELDS = (
    "model_id",
    "model",
    "model_scale",
    "scope_id",
    "scope_claim",
    "job_id",
    "seed",
    "target_selected_weight_rate",
    "selected_tensors",
    "selected_parameters",
    "model_parameters",
    "eval_tokens",
    "baseline_nll",
    "baseline_perplexity",
    "strategy",
    "artifact_file_bytes",
    "artifact_natural_file_bytes",
    "artifact_tail_padding_bytes",
    "reference_artifact_file_bytes",
    "selected_weight_physical_rate",
    "selected_weight_physical_compression_ratio",
    "physical_bits_per_selected_parameter",
    "added_physical_bits_per_selected_parameter_vs_q",
    "heldout_nll",
    "nll_delta",
    "heldout_perplexity",
    "perplexity_delta",
    "nll_recovery_vs_q",
    "nll_recovery_per_added_physical_bit_per_parameter",
    "physical_efficiency_kind",
    "normalized_hessian_cost",
    "rho_sl",
    "rho_qs",
    "rho_ql",
    "rho_sl_kind",
    "rho_qs_kind",
    "rho_ql_kind",
    "sl_near_orthogonal",
    "sparse_nnz",
    "lowrank_rank_sum",
    "folded_repair_dof",
    "strict_equal_bytes_to_ql",
    "artifact_sha256",
)


PAIR_FIELDS = (
    "model_id",
    "model",
    "job_id",
    "comparison_id",
    "left_strategy",
    "right_strategy",
    "left_artifact_file_bytes",
    "right_artifact_file_bytes",
    "artifact_file_byte_difference",
    "same_final_file_bytes",
    "endpoint_nll_difference_left_minus_right",
    "endpoint_perplexity_difference_left_minus_right",
    "fixed_window_count",
    "fixed_window_mean_nll_difference_left_minus_right",
    "fixed_window_sample_standard_error",
    "fixed_window_descriptive_interval_low",
    "fixed_window_descriptive_interval_high",
    "left_wins_lower_nll",
    "interval_semantics",
)


MODEL_FIELDS = (
    "model_id",
    "model",
    "model_declared",
    "model_scale",
    "scope_id",
    "scope_claim",
    "selected_tensors",
    "selected_parameters",
    "model_parameters",
    "eval_tokens",
    "baseline_nll",
    "baseline_perplexity",
    "q_artifact_file_bytes",
    "ql_and_strict_artifact_file_bytes",
    "ql_selected_weight_physical_rate",
    "ql_nll_delta",
    "strict_nll_delta",
    "strict_minus_ql_nll",
    "strict_minus_ql_perplexity",
    "strict_pair_interval_low",
    "strict_pair_interval_high",
    "strict_pair_left_wins",
    "strict_rho_sl",
    "strict_rho_qs",
    "strict_rho_ql",
    "strict_sl_near_orthogonal",
    "proxy_nll_correlation_min",
    "proxy_nll_correlation_max",
    "evidence_role",
    "seed_aggregation_allowed",
)


PATH_FIELDS = (
    "model_id",
    "model",
    "job_id",
    "strategy",
    "epsilon",
    "nll_delta",
    "normalized_hessian_cost",
    "taylor_fit_nll_delta",
    "taylor_fit_absolute_error",
    "small_epsilon_fit_max",
    "inside_taylor_fit_interval",
    "fit_is_extrapolation",
    "hessian_proxy_nll_correlation",
    "path_kind",
    "deployable",
    "eval_tokens",
    "target_selected_weight_rate",
    "evidence_role",
)


def _derived_rows(
    jobs: Sequence[ValidatedJob],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    endpoint_output: list[dict[str, object]] = []
    pair_output: list[dict[str, object]] = []
    model_output: list[dict[str, object]] = []
    path_output: list[dict[str, object]] = []
    for job in jobs:
        run = job.run_config
        selected_parameters = int(run["selected_parameter_count"])
        baseline = run["baseline_metrics"]
        q = job.endpoint_by_strategy["Q"]
        q_bytes = _integer(q["artifact_file_bytes"], "Q bytes", positive=True)
        q_nll_delta = _finite(q["nll_delta"], "Q NLL delta")
        reference_bytes = _integer(q["reference_artifact_file_bytes"], "reference bytes", positive=True)
        threshold = float(run["arguments"]["rho_threshold"])
        for row in job.endpoint_rows:
            strategy = row["strategy"]
            file_bytes = _integer(row["artifact_file_bytes"], f"{strategy} bytes", positive=True)
            added_bpp = 8.0 * (file_bytes - q_bytes) / selected_parameters
            recovery = q_nll_delta - _finite(row["nll_delta"], f"{strategy} NLL delta")
            if file_bytes == q_bytes:
                efficiency = float("nan")
                if strategy == "Q":
                    efficiency_kind = "reference_endpoint"
                elif strategy == "Q_global_scale":
                    efficiency_kind = "zero_byte_folded_recovery"
                else:
                    efficiency_kind = "same_byte_direct_recovery"
            elif file_bytes > q_bytes:
                efficiency = recovery / added_bpp
                efficiency_kind = "positive_exact_file_increment"
            else:
                efficiency = float("nan")
                efficiency_kind = "smaller_than_q_not_ranked"
            rho_sl = _optional_float(row["rho_sl"], f"{strategy} rho_sl")
            endpoint_output.append(
                {
                    "model_id": job.stage_id,
                    "model": job.label,
                    "model_scale": job.stage["model_scale"],
                    "scope_id": job.stage["tensor_scope"]["id"],
                    "scope_claim": job.stage["tensor_scope"]["claim_scope"],
                    "job_id": job.job_id,
                    "seed": 17,
                    "target_selected_weight_rate": _format(0.258),
                    "selected_tensors": len(run["selected_layers"]),
                    "selected_parameters": selected_parameters,
                    "model_parameters": int(run["model_parameter_count"]),
                    "eval_tokens": int(run["actual_eval_tokens"]),
                    "baseline_nll": _format(float(baseline["nll"])),
                    "baseline_perplexity": _format(float(baseline["perplexity"])),
                    "strategy": strategy,
                    "artifact_file_bytes": file_bytes,
                    "artifact_natural_file_bytes": _integer(row["artifact_natural_file_bytes"], "natural bytes"),
                    "artifact_tail_padding_bytes": _integer(row["artifact_tail_padding_bytes"], "tail padding"),
                    "reference_artifact_file_bytes": reference_bytes,
                    "selected_weight_physical_rate": _format(file_bytes / reference_bytes),
                    "selected_weight_physical_compression_ratio": _format(reference_bytes / file_bytes),
                    "physical_bits_per_selected_parameter": _format(8.0 * file_bytes / selected_parameters),
                    "added_physical_bits_per_selected_parameter_vs_q": _format(added_bpp),
                    "heldout_nll": _format(float(row["heldout_nll"])),
                    "nll_delta": _format(float(row["nll_delta"])),
                    "heldout_perplexity": _format(float(row["heldout_perplexity"])),
                    "perplexity_delta": _format(float(row["perplexity_delta"])),
                    "nll_recovery_vs_q": _format(recovery),
                    "nll_recovery_per_added_physical_bit_per_parameter": _format(efficiency),
                    "physical_efficiency_kind": efficiency_kind,
                    "normalized_hessian_cost": _format(float(row["normalized_hessian_cost"])),
                    "rho_sl": _format(rho_sl),
                    "rho_qs": _format(_optional_float(row["rho_qs"], f"{strategy} rho_qs")),
                    "rho_ql": _format(_optional_float(row["rho_ql"], f"{strategy} rho_ql")),
                    "rho_sl_kind": row["rho_sl_kind"],
                    "rho_qs_kind": row["rho_qs_kind"],
                    "rho_ql_kind": row["rho_ql_kind"],
                    "sl_near_orthogonal": "" if math.isnan(rho_sl) else str(abs(rho_sl) <= threshold).lower(),
                    "sparse_nnz": _integer(row["sparse_nnz"], "sparse nnz"),
                    "lowrank_rank_sum": _integer(row["lowrank_rank_sum"], "rank sum"),
                    "folded_repair_dof": _integer(row["folded_repair_dof"], "folded repair DOF"),
                    "strict_equal_bytes_to_ql": str(strategy == STRICT and file_bytes == _integer(job.endpoint_by_strategy[QL]["artifact_file_bytes"], "QL bytes")).lower(),
                    "artifact_sha256": row["artifact_sha256"],
                }
            )
        for pair in job.pairs:
            left = job.endpoint_by_strategy[pair.left]
            right = job.endpoint_by_strategy[pair.right]
            left_bytes = _integer(left["artifact_file_bytes"], "left bytes", positive=True)
            right_bytes = _integer(right["artifact_file_bytes"], "right bytes", positive=True)
            pair_output.append(
                {
                    "model_id": job.stage_id,
                    "model": job.label,
                    "job_id": job.job_id,
                    "comparison_id": pair.comparison_id,
                    "left_strategy": pair.left,
                    "right_strategy": pair.right,
                    "left_artifact_file_bytes": left_bytes,
                    "right_artifact_file_bytes": right_bytes,
                    "artifact_file_byte_difference": left_bytes - right_bytes,
                    "same_final_file_bytes": str(left_bytes == right_bytes).lower(),
                    "endpoint_nll_difference_left_minus_right": _format(float(left["nll_delta"]) - float(right["nll_delta"])),
                    "endpoint_perplexity_difference_left_minus_right": _format(float(left["perplexity_delta"]) - float(right["perplexity_delta"])),
                    "fixed_window_count": pair.count,
                    "fixed_window_mean_nll_difference_left_minus_right": _format(pair.mean),
                    "fixed_window_sample_standard_error": _format(pair.standard_error),
                    "fixed_window_descriptive_interval_low": _format(pair.interval_low),
                    "fixed_window_descriptive_interval_high": _format(pair.interval_high),
                    "left_wins_lower_nll": pair.left_wins,
                    "interval_semantics": INTERVAL_SEMANTICS,
                }
            )
        strict = job.endpoint_by_strategy[STRICT]
        ql = job.endpoint_by_strategy[QL]
        strict_pair = next(pair for pair in job.pairs if pair.comparison_id == "strict_qsl_vs_ql")
        correlations = [float(row["hessian_proxy_nll_correlation"]) for row in job.comfort_summary.values()]
        rho_sl = float(strict["rho_sl"])
        model_output.append(
            {
                "model_id": job.stage_id,
                "model": job.label,
                "model_declared": job.stage["model"],
                "model_scale": job.stage["model_scale"],
                "scope_id": job.stage["tensor_scope"]["id"],
                "scope_claim": job.stage["tensor_scope"]["claim_scope"],
                "selected_tensors": len(run["selected_layers"]),
                "selected_parameters": selected_parameters,
                "model_parameters": int(run["model_parameter_count"]),
                "eval_tokens": int(run["actual_eval_tokens"]),
                "baseline_nll": _format(float(baseline["nll"])),
                "baseline_perplexity": _format(float(baseline["perplexity"])),
                "q_artifact_file_bytes": q_bytes,
                "ql_and_strict_artifact_file_bytes": _integer(ql["artifact_file_bytes"], "QL bytes", positive=True),
                "ql_selected_weight_physical_rate": _format(float(ql["artifact_file_bytes"]) / reference_bytes),
                "ql_nll_delta": _format(float(ql["nll_delta"])),
                "strict_nll_delta": _format(float(strict["nll_delta"])),
                "strict_minus_ql_nll": _format(float(strict["nll_delta"]) - float(ql["nll_delta"])),
                "strict_minus_ql_perplexity": _format(float(strict["perplexity_delta"]) - float(ql["perplexity_delta"])),
                "strict_pair_interval_low": _format(strict_pair.interval_low),
                "strict_pair_interval_high": _format(strict_pair.interval_high),
                "strict_pair_left_wins": strict_pair.left_wins,
                "strict_rho_sl": _format(rho_sl),
                "strict_rho_qs": _format(float(strict["rho_qs"])),
                "strict_rho_ql": _format(float(strict["rho_ql"])),
                "strict_sl_near_orthogonal": str(abs(rho_sl) <= threshold).lower(),
                "proxy_nll_correlation_min": _format(min(correlations)),
                "proxy_nll_correlation_max": _format(max(correlations)),
                "evidence_role": EVIDENCE_ROLE,
                "seed_aggregation_allowed": "false",
            }
        )
        for row in job.comfort_rows:
            strategy = str(row["strategy"])
            epsilon = _finite(row["epsilon"], f"{strategy} path epsilon")
            summary = job.comfort_summary[strategy]
            fit_max = _finite(
                summary["small_epsilon_fit_max"], f"{strategy} path fit boundary"
            )
            path_output.append(
                {
                    "model_id": job.stage_id,
                    "model": job.label,
                    "job_id": job.job_id,
                    "strategy": strategy,
                    "epsilon": _format(epsilon),
                    "nll_delta": _format(_finite(row["nll_delta"], f"{strategy} path NLL")),
                    "normalized_hessian_cost": _format(
                        _finite(
                            row["normalized_hessian_cost"],
                            f"{strategy} path normalized Hessian",
                        )
                    ),
                    "taylor_fit_nll_delta": _format(
                        _finite(
                            row["taylor_fit_nll_delta"], f"{strategy} Taylor path"
                        )
                    ),
                    "taylor_fit_absolute_error": _format(
                        abs(
                            _finite(row["nll_delta"], f"{strategy} path NLL")
                            - _finite(
                                row["taylor_fit_nll_delta"],
                                f"{strategy} Taylor path",
                            )
                        )
                    ),
                    "small_epsilon_fit_max": _format(fit_max),
                    "inside_taylor_fit_interval": str(epsilon <= fit_max).lower(),
                    "fit_is_extrapolation": str(epsilon > fit_max).lower(),
                    "hessian_proxy_nll_correlation": _format(
                        _finite(
                            summary["hessian_proxy_nll_correlation"],
                            f"{strategy} path proxy correlation",
                        )
                    ),
                    "path_kind": row["path_kind"],
                    "deployable": str(
                        _boolean(row["deployable"], f"{strategy} path deployable")
                    ).lower(),
                    "eval_tokens": int(run["actual_eval_tokens"]),
                    "target_selected_weight_rate": _format(0.258),
                    "evidence_role": EVIDENCE_ROLE,
                }
            )
    return endpoint_output, pair_output, model_output, path_output


def _main_table(models: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Single-seed scalability smoke at target selected-weight artifact ratio 0.258. Pythia covers all MLP projections, while OPT and Qwen use ten depth-stratified MLP projections. Bytes are complete aligned research-codec files for selected tensors, not whole-model or production payloads. ``Strict$-$QL'' is the signed held-out NLL difference after tail padding to identical final bytes; the strict natural files leave 31,360/640/1,024 bytes unused for Pythia/OPT/Qwen. Negative favors the composite. Rows are not pooled or ranked across models.}",
        r"\label{tab:scaling-pilot}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr@{}}",
        r"\toprule",
        r"Model / scope & \shortstack{Tensors /\\M params} & \shortstack{QL=strict\\bytes (rate)} & $\Delta$NLL QL & $\Delta$NLL strict & Strict$-$QL & $\rho_{SL}$ & $\rho_{QS}/\rho_{QL}$ \\",
        r"\midrule",
    ]
    for row in models:
        scope = SCOPE_LABELS[str(row["scope_id"])]
        lines.append(
            "{} / {} & {} / {:.2f} & {:,} ({:.4f}) & {:+.4f} & {:+.4f} & {:+.4f} & {:+.3f} & {:+.3f}/{:+.3f} \\\\".format(
                _latex_escape(str(row["model"])),
                _latex_escape(scope),
                int(row["selected_tensors"]),
                int(row["selected_parameters"]) / 1e6,
                int(row["ql_and_strict_artifact_file_bytes"]),
                float(row["ql_selected_weight_physical_rate"]),
                float(row["ql_nll_delta"]),
                float(row["strict_nll_delta"]),
                float(row["strict_minus_ql_nll"]),
                float(row["strict_rho_sl"]),
                float(row["strict_rho_qs"]),
                float(row["strict_rho_ql"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _endpoint_table(endpoints: Sequence[Mapping[str, object]]) -> str:
    display = {
        "Q": "Q",
        "Q_global_scale": "Q + global scale",
        "Q_block_scale": "Q + block scale",
        "Q+S": "Q+S",
        "Q+S_OBS": "Q+S (OBS)",
        "Q+L": "Q+L",
        "Q+S+L_QL_budget": "Q+S+L strict",
        STRICT: "Q+S+L strict + scale",
        "Q+S+L": "Q+S+L relaxed",
        "Q+S_OBS+L": "Q+S (OBS)+L",
        "Q+S+L_component_scale": "Q+S+L relaxed + scale",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{All native endpoints for the three scalability-smoke jobs. Rate and bits/parameter charge complete selected-tensor artifact files. The scopes and tokenizers differ, so rows must only be compared within each model block.}",
        r"\label{tab:scaling-pilot-all-endpoints}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Model & Endpoint & Bytes & Rate & bits/sel. param & $\Delta$NLL & $\Delta$PPL & norm. $H$ & Repair (nnz/rank/dof) \\",
        r"\midrule",
    ]
    previous = None
    for row in endpoints:
        model = str(row["model"])
        if previous is not None and model != previous:
            lines.append(r"\addlinespace")
        label = model if model != previous else ""
        lines.append(
            "{} & {} & {:,} & {:.4f} & {:.3f} & {:+.4f} & {:+.3f} & {:.5f} & {}/{}/{} \\\\".format(
                _latex_escape(label),
                _latex_escape(display[str(row["strategy"])]),
                int(row["artifact_file_bytes"]),
                float(row["selected_weight_physical_rate"]),
                float(row["physical_bits_per_selected_parameter"]),
                float(row["nll_delta"]),
                float(row["perplexity_delta"]),
                float(row["normalized_hessian_cost"]),
                int(row["sparse_nnz"]),
                int(row["lowrank_rank_sum"]),
                int(row["folded_repair_dof"]),
            )
        )
        previous = model
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    )
    return "\n".join(lines) + "\n"


def _numbers(models: Sequence[Mapping[str, object]], source_sha: str) -> str:
    commands: list[tuple[str, str]] = [
        ("ScalePilotJobCount", str(len(models))),
        ("ScalePilotSeed", "17"),
        ("ScalePilotWindowCount", "8"),
        ("ScalePilotEvalTokens", f"{int(models[0]['eval_tokens']):,}"),
        ("ScalePilotNumericalSourceSHA", r"\texttt{" + source_sha[:12] + "}"),
    ]
    for row in models:
        prefix = MODEL_MACROS[str(row["model_id"])]
        commands.extend(
            [
                (f"ScalePilot{prefix}SelectedTensors", str(row["selected_tensors"])),
                (f"ScalePilot{prefix}SelectedParams", f"{int(row['selected_parameters']):,}"),
                (f"ScalePilot{prefix}StrictMinusQLNLL", f"{float(row['strict_minus_ql_nll']):+.4f}"),
                (f"ScalePilot{prefix}StrictMinusQLPPL", f"{float(row['strict_minus_ql_perplexity']):+.3f}"),
                (f"ScalePilot{prefix}StrictSLRho", f"{float(row['strict_rho_sl']):+.3f}"),
                (f"ScalePilot{prefix}StrictQSRho", f"{float(row['strict_rho_qs']):+.3f}"),
                (f"ScalePilot{prefix}StrictQLRho", f"{float(row['strict_rho_ql']):+.3f}"),
                (f"ScalePilot{prefix}StrictWins", f"{row['strict_pair_left_wins']}/8"),
            ]
        )
    return "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in commands) + "\n"


def _summary_markdown(
    models: Sequence[Mapping[str, object]], pairs: Sequence[Mapping[str, object]]
) -> str:
    lines = [
        "# Verified three-job scalability smoke",
        "",
        "This report is generated from the fail-closed scaling aggregate.  It contains three ",
        "separate seed-17 observations at selected-weight artifact target 0.258.  It is not a ",
        "multi-seed result, a cross-model leaderboard, a model-size trend, or a whole-model ",
        "compression claim.",
        "",
        "## Final-byte-equal conservative combination and Hessian geometry",
        "",
        "| Model / scope | Tensors | QL=strict bytes | Strict-QL NLL | 8-window wins | rho_SL | rho_QS | rho_QL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models:
        scope = SCOPE_LABELS[str(row["scope_id"])]
        lines.append(
            "| {} / {} | {} | {:,} | {:+.6f} | {}/8 | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                row["model"],
                scope,
                int(row["selected_tensors"]),
                int(row["ql_and_strict_artifact_file_bytes"]),
                float(row["strict_minus_ql_nll"]),
                int(row["strict_pair_left_wins"]),
                float(row["strict_rho_sl"]),
                float(row["strict_rho_qs"]),
                float(row["strict_rho_ql"]),
            )
        )
    lines.extend(
        [
            "",
            "`Strict-QL NLL` is the signed held-out endpoint difference; negative favors the ",
            "strict component-scaled Q+S+L endpoint.  Final byte equality is produced by tail ",
            "padding: the strict natural files leave 31,360/640/1,024 bytes unused for ",
            "Pythia/OPT/Qwen, so these are conservative candidates rather than ",
            "budget-exhausted frontiers.  The window count is descriptive over the ",
            "same eight fixed windows and is not a significance test.  Values with ",
            "`abs(rho_SL) <= 0.1` satisfy the declared near-orthogonality diagnostic; negative ",
            "Q-S or Q-L values are cancellation, not orthogonality.",
            "",
            "## Within-model parameter-utilization controls",
            "",
            "Every numeric difference below is `left - right`; negative NLL favors the left ",
            "endpoint.  Byte differences charge the complete aligned selected-tensor research ",
            "artifact.",
            "",
            "| Model | Comparison | Left / right | Byte difference | NLL difference | Same bytes? |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    pair_labels = {
        "global_scale_vs_q": "folded global scale",
        "block_scale_vs_q": "block scale",
        "obs_vs_qs": "fixed-support OBS values",
        "strict_qsl_vs_ql": "strict scaled composition",
    }
    for row in pairs:
        lines.append(
            "| {} | {} | `{}` / `{}` | {:+,} | {:+.6f} | {} |".format(
                row["model"],
                pair_labels[str(row["comparison_id"])],
                row["left_strategy"],
                row["right_strategy"],
                int(row["artifact_file_byte_difference"]),
                float(row["endpoint_nll_difference_left_minus_right"]),
                row["same_final_file_bytes"],
            )
        )
    lines.extend(
        [
            "",
            "Zero-byte scale or OBS improvements are reported as direct recovery rather than ",
            "infinite recovery-per-byte.  Positive-cost controls can be compared within a model ",
            "using `scaling_pilot_endpoints.csv`, which reports added exact bits per selected ",
            "parameter and held-out NLL recovery from Q.",
            "",
            "## Theory-to-experiment boundary",
            "",
            "- The signed rho values test local additivity in the declared PSD activation-Gram ",
            "  geometry; they do not certify held-out accuracy.",
            "- The strict equal-file-byte endpoint difference is the realized exchange test.  ",
            "  The experiment does not separately identify every term in the theoretical ",
            "  `P_A + Gamma_A` decomposition.",
            "- Six 13-point paths per model connect the local Taylor diagnostic to epsilon=1.  ",
            "  Fits use only epsilon <= 0.125; the remainder is labelled extrapolation.",
            "- External methods remain in A0/A1/B/C training-dependence lanes.  Literature ",
            "  numbers are not placed on these axes without the same checkpoint, tensor scope, ",
            "  data, serializer, and exported-state accounting.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_report(
    config_path: Path,
    suite_root: Path,
    output_dir: Path,
    *,
    repo_root: Path,
) -> list[Path]:
    config, jobs, input_hashes, source_sha = validate_inputs(
        config_path, suite_root, repo_root=repo_root
    )
    endpoints, pairs, models, paths = _derived_rows(jobs)
    contents = {
        "scaling_pilot_endpoints.csv": _csv_text(ENDPOINT_FIELDS, endpoints),
        "scaling_pilot_pairs.csv": _csv_text(PAIR_FIELDS, pairs),
        "scaling_pilot_models.csv": _csv_text(MODEL_FIELDS, models),
        "scaling_pilot_paths.csv": _csv_text(PATH_FIELDS, paths),
        "scaling_pilot_table.tex": _main_table(models),
        "scaling_pilot_endpoints_table.tex": _endpoint_table(endpoints),
        "scaling_pilot_numbers.tex": _numbers(models, source_sha),
        "scaling_pilot_summary.md": _summary_markdown(models, pairs),
    }
    output_hashes = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in contents.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "evidence_status": "verified_single_seed_scalability_smoke",
        "observation_count": 3,
        "seed": 17,
        "target_selected_weight_artifact_ratio": 0.258,
        "numerical_source_sha256": source_sha,
        "strategy_order": list(EXPECTED_STRATEGIES),
        "comparison_order": [item[0] for item in COMPARISONS],
        "formulas": {
            "selected_weight_physical_rate": "artifact_file_bytes / reference_artifact_file_bytes",
            "physical_bits_per_selected_parameter": "8 * artifact_file_bytes / selected_parameter_count",
            "added_physical_bits_per_selected_parameter_vs_q": "8 * (artifact_file_bytes - q_artifact_file_bytes) / selected_parameter_count",
            "nll_recovery_vs_q": "q_nll_delta - method_nll_delta",
            "positive_increment_efficiency": "nll_recovery_vs_q / added_physical_bits_per_selected_parameter_vs_q",
            "strict_same_byte_difference": "strict_qsl_nll_delta - ql_nll_delta; negative favors composite",
            "path_fit_region": "Taylor coefficients are fitted only on 0 < epsilon <= 0.125; epsilon > 0.125 is labelled extrapolation",
            "float32_hessian_decomposition_bound": {
                "formula": "abs(total - sum(terms)) <= factor * float32_epsilon * (abs(total) + sum(abs(term)))",
                "factor": HESSIAN_DECOMPOSITION_ULP_FACTOR,
                "float32_epsilon": FLOAT32_EPSILON,
                "hidden_absolute_tolerance": False,
            },
            "covariance_psd_repair_closure": {
                "rejection_rtol": PSD_REJECTION_RTOL,
                "storage_floor_rtol": FLOAT32_PSD_FLOOR_RTOL,
                "float32_closure_rtol": PSD_FLOAT32_CLOSURE_RTOL,
            },
        },
        "interval_semantics": INTERVAL_SEMANTICS,
        "claim_limitations": [
            "three separate model/scope observations are not pooled into a cross-model mean",
            "one seed and eight fixed windows do not support an independence-based significance claim",
            "Pythia full-MLP and OPT/Qwen depth-stratified scopes do not form a model-scale trend",
            "rates and bytes apply only to selected linear-weight research artifacts, not whole models or production payloads",
            "raw NLL, perplexity, delta perplexity, and file bytes are not ranked across tokenizers/scopes",
            "literature-reported methods are excluded from these numeric axes unless rerun under the identical checkpoint, scope, data, and codec",
            "path probes are one-dimensional interpolation slices; Taylor fits use epsilon <= 0.125 and are extrapolations beyond that interval",
        ],
        "input_sha256": dict(sorted(input_hashes.items())),
        "outputs": output_hashes,
        "config_contract": {
            "comfort_epsilons": config["common"]["comfort_epsilons"],
            "comfort_fit_max_epsilon": config["common"]["comfort_fit_max_epsilon"],
            "payload_scope": PAYLOAD_SCOPE,
            "production_backend": False,
            "seed_aggregation_allowed": False,
        },
    }
    contents["scaling_pilot_manifest.json"] = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    paths: list[Path] = []
    output_dir = output_dir.resolve()
    for name, value in contents.items():
        path = output_dir / name
        _atomic_write(path, value)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "large_scale_hessian_pilot_20260714.json",
    )
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=repo_root / "results" / "large_scale_hessian_pilot_20260714",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    args = parser.parse_args(argv)
    try:
        paths = build_report(
            args.config,
            args.suite_root,
            args.output_dir,
            repo_root=args.repo_root,
        )
    except ReportError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
