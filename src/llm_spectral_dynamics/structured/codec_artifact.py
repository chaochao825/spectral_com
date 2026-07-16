"""Deterministic research artifacts for the Q/S/L value-stream codec.

The format in this module is deliberately small and auditable.  It is not a
claim about a production inference backend.  Its purpose is to turn the
declared packed-code, scale, sparse-CSR, and low-rank streams into real bytes
so container metadata and alignment can no longer be silently omitted from a
fixed-rate comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MAGIC = b"HRCODEC1"
FORMAT_VERSION = 1
PREFIX = struct.Struct("<8sIIQ")  # magic, version, alignment, JSON bytes
DEFAULT_ALIGNMENT = 64
PACK_CHUNK_ELEMENTS = 1 << 20


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (int(value) + alignment - 1) // alignment * alignment


def pack_signed_codes(values: np.ndarray, bits: int) -> bytes:
    """Pack signed two's-complement integers into an LSB-first bit stream."""

    width = int(bits)
    if width <= 0 or width > 8:
        raise ValueError("bits must be within [1, 8]")
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("quantized codes must use an integer dtype")
    flat = array.reshape(-1)
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    if flat.size and (int(flat.min()) < minimum or int(flat.max()) > maximum):
        raise ValueError(f"code outside signed {width}-bit range [{minimum}, {maximum}]")
    if not flat.size:
        return b""

    # Keep every non-final chunk byte aligned so concatenating independently
    # packed chunks is identical to packing the full bit stream.  The previous
    # implementation materialized an ``N x bits`` int64 array; for a 7B MLP
    # tensor that temporary alone could require several GiB.
    alignment_period = 8 // math.gcd(width, 8)
    chunk_elements = max(
        alignment_period,
        (int(PACK_CHUNK_ELEMENTS) // alignment_period) * alignment_period,
    )
    shifts = np.arange(width, dtype=np.int64)
    mask = (1 << width) - 1
    packed = bytearray()
    for start in range(0, int(flat.size), chunk_elements):
        stop = min(start + chunk_elements, int(flat.size))
        chunk = flat[start:stop].astype(np.int64, copy=False)
        unsigned = chunk & mask
        bit_columns = ((unsigned[:, None] >> shifts) & 1).astype(np.uint8)
        packed.extend(np.packbits(bit_columns.reshape(-1), bitorder="little").tobytes())
    return bytes(packed)


def unpack_signed_codes(payload: bytes, *, count: int, bits: int) -> np.ndarray:
    width = int(bits)
    elements = int(count)
    if width <= 0 or width > 8:
        raise ValueError("bits must be within [1, 8]")
    if elements < 0:
        raise ValueError("count must be non-negative")
    required = (elements * width + 7) // 8
    if len(payload) != required:
        raise ValueError(f"packed code length mismatch: expected {required}, got {len(payload)}")
    if elements == 0:
        return np.zeros(0, dtype=np.int8)
    bits_array = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    columns = bits_array[: elements * width].reshape(elements, width).astype(np.int64)
    unsigned = columns @ (1 << np.arange(width, dtype=np.int64))
    sign = 1 << (width - 1)
    signed = np.where(unsigned >= sign, unsigned - (1 << width), unsigned)
    return signed.astype(np.int8)


def _index_dtype(columns: int) -> np.dtype:
    if columns <= 0:
        raise ValueError("columns must be positive")
    if columns <= 2**8:
        return np.dtype("u1")
    if columns <= 2**16:
        return np.dtype("<u2")
    return np.dtype("<u4")


@dataclass(frozen=True)
class LayerCodecPayload:
    name: str
    q_codes: np.ndarray
    q_scales: np.ndarray
    q_bits: int
    q_col_block_size: int | None = None
    sparse_values: np.ndarray | None = None
    sparse_mask: np.ndarray | None = None
    lowrank_left: np.ndarray | None = None
    lowrank_right: np.ndarray | None = None
    lowrank_factor_bits: int = 16
    lowrank_left_scales: np.ndarray | None = None
    lowrank_right_scales: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        codes = np.asarray(self.q_codes)
        if codes.ndim != 2:
            raise ValueError("q_codes must be a matrix")
        return int(codes.shape[0]), int(codes.shape[1])


@dataclass(frozen=True)
class LayerCodecAllocation:
    """Value-independent fields that determine one layer's artifact layout."""

    name: str
    shape: tuple[int, int]
    q_bits: int
    q_scale_shape: tuple[int, ...]
    q_col_block_size: int | None = None
    sparse_nnz: int = 0
    lowrank_rank: int = 0
    lowrank_factor_bits: int = 16
    lowrank_left_scale_shape: tuple[int, ...] = ()
    lowrank_right_scale_shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class CodecArtifactAllocationLayout:
    """Exact value-independent statistics for a canonical codec layout."""

    natural_file_bytes: int
    header_bytes: int
    payload_base_bytes: int
    payload_end_bytes: int
    stream_bytes: int
    internal_padding_bytes: int
    logical_payload_bits: int


@dataclass(frozen=True)
class ArtifactWriteResult:
    path: Path
    file_bytes: int
    logical_payload_bits: int
    stream_bytes: int
    container_bytes: int
    alignment_padding_bytes: int
    tail_padding_bytes: int
    sha256: str
    manifest: dict[str, object]


@dataclass(frozen=True)
class DecodedArtifact:
    manifest: dict[str, object]
    layers: dict[str, np.ndarray]
    file_bytes: int
    sha256: str


@dataclass(frozen=True)
class _Stream:
    name: str
    layer: str
    component: str
    encoding: str
    dtype: str
    shape: tuple[int, ...]
    logical_bits: int
    payload: bytes


@dataclass(frozen=True)
class _SizedStream:
    """A value-free stream descriptor for the shared canonical layout path."""

    name: str
    layer: str
    component: str
    encoding: str
    dtype: str
    shape: tuple[int, ...]
    logical_bits: int
    nbytes: int
    sha256: str = "0" * 64


def _stream_nbytes(stream: _Stream | _SizedStream) -> int:
    return len(stream.payload) if isinstance(stream, _Stream) else int(stream.nbytes)


def _stream_sha256(stream: _Stream | _SizedStream) -> str:
    return (
        hashlib.sha256(stream.payload).hexdigest()
        if isinstance(stream, _Stream)
        else str(stream.sha256)
    )


def _little_endian_bytes(array: np.ndarray, dtype: str) -> bytes:
    return np.asarray(array, dtype=np.dtype(dtype)).tobytes(order="C")


def _validate_layer(layer: LayerCodecPayload) -> None:
    rows, cols = layer.shape
    if not layer.name:
        raise ValueError("layer name must be non-empty")
    if rows <= 0 or cols <= 0:
        raise ValueError("layer dimensions must be positive")
    codes = np.asarray(layer.q_codes)
    bits = int(layer.q_bits)
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be within [1, 8]")
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError("quantized codes must use an integer dtype")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if codes.size and (int(codes.min()) < minimum or int(codes.max()) > maximum):
        raise ValueError(f"code outside signed {bits}-bit range [{minimum}, {maximum}]")
    scales = np.asarray(layer.q_scales)
    if layer.q_col_block_size is None:
        expected_scale_shape = (rows,)
    else:
        block = int(layer.q_col_block_size)
        if block <= 0:
            raise ValueError("q_col_block_size must be positive")
        expected_scale_shape = (rows, (cols + block - 1) // block)
    if scales.shape != expected_scale_shape:
        raise ValueError(f"q_scales shape mismatch: expected {expected_scale_shape}, got {scales.shape}")
    if (layer.sparse_values is None) != (layer.sparse_mask is None):
        raise ValueError("sparse_values and sparse_mask must be provided together")
    if layer.sparse_mask is not None:
        if np.asarray(layer.sparse_mask).shape != (rows, cols):
            raise ValueError("sparse_mask shape mismatch")
        if np.asarray(layer.sparse_values).shape != (rows, cols):
            raise ValueError("sparse_values shape mismatch")
        if int(np.count_nonzero(layer.sparse_mask)) == 0:
            raise ValueError("empty sparse components must be represented as None")
    if (layer.lowrank_left is None) != (layer.lowrank_right is None):
        raise ValueError("both low-rank factors must be provided together")
    if (layer.lowrank_left_scales is None) != (layer.lowrank_right_scales is None):
        raise ValueError("both low-rank factor scales must be provided together")
    if layer.lowrank_left is not None:
        left = np.asarray(layer.lowrank_left)
        right = np.asarray(layer.lowrank_right)
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != rows or right.shape[1] != cols:
            raise ValueError("low-rank factor shape mismatch")
        if left.shape[1] != right.shape[0]:
            raise ValueError("low-rank inner dimensions differ")
        if left.shape[1] == 0:
            raise ValueError("rank-0 low-rank components must be represented as None")
        if left.shape[1] > min(rows, cols):
            raise ValueError("low-rank rank exceeds the matrix dimensions")
        factor_bits = int(layer.lowrank_factor_bits)
        if factor_bits == 16:
            if layer.lowrank_left_scales is not None:
                raise ValueError("FP16 low-rank factors must not carry quantization scales")
        else:
            if factor_bits < 2 or factor_bits > 8:
                raise ValueError("quantized low-rank factor bits must be within [2, 8]")
            if layer.lowrank_left_scales is None:
                raise ValueError("quantized low-rank factors require row scales")
            if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(
                right.dtype, np.integer
            ):
                raise TypeError("quantized low-rank factors must use integer codes")
            minimum = -(1 << (factor_bits - 1))
            maximum = (1 << (factor_bits - 1)) - 1
            if left.size and (
                int(left.min()) < minimum or int(left.max()) > maximum
            ):
                raise ValueError("left low-rank factor code exceeds its signed bit width")
            if right.size and (
                int(right.min()) < minimum or int(right.max()) > maximum
            ):
                raise ValueError("right low-rank factor code exceeds its signed bit width")
            if np.asarray(layer.lowrank_left_scales).shape != (left.shape[0],):
                raise ValueError("left low-rank factor scale shape mismatch")
            if np.asarray(layer.lowrank_right_scales).shape != (right.shape[0],):
                raise ValueError("right low-rank factor scale shape mismatch")
    elif layer.lowrank_left_scales is not None:
        raise ValueError("rank-0 low-rank components must not carry factor scales")


def _layer_streams(layer: LayerCodecPayload) -> tuple[list[_Stream], dict[str, object]]:
    _validate_layer(layer)
    rows, cols = layer.shape
    codes = np.asarray(layer.q_codes)
    packed = pack_signed_codes(codes, int(layer.q_bits))
    streams = [
        _Stream(
            name=f"{layer.name}/q_codes",
            layer=layer.name,
            component="q_codes",
            encoding=f"signed_twos_complement_lsb_bitpack_{int(layer.q_bits)}",
            dtype="bitpack",
            shape=(rows, cols),
            logical_bits=rows * cols * int(layer.q_bits),
            payload=packed,
        ),
        _Stream(
            name=f"{layer.name}/q_scales",
            layer=layer.name,
            component="q_scales",
            encoding="raw_little_endian",
            dtype="float16",
            shape=tuple(map(int, np.asarray(layer.q_scales).shape)),
            logical_bits=int(np.asarray(layer.q_scales).size) * 16,
            payload=_little_endian_bytes(layer.q_scales, "<f2"),
        ),
    ]
    components: dict[str, str | None] = {
        "q_codes": streams[0].name,
        "q_scales": streams[1].name,
        "sparse_values": None,
        "sparse_row_ptr": None,
        "sparse_col_idx": None,
        "lowrank_left": None,
        "lowrank_right": None,
    }
    sparse_nnz = 0
    if layer.sparse_mask is not None:
        mask = np.asarray(layer.sparse_mask, dtype=bool)
        values = np.asarray(layer.sparse_values, dtype=np.float16)[mask]
        row_counts = np.count_nonzero(mask, axis=1).astype(np.uint32)
        row_ptr = np.concatenate([np.zeros(1, dtype=np.uint32), np.cumsum(row_counts, dtype=np.uint32)])
        col_dtype = _index_dtype(cols)
        col_idx = np.nonzero(mask)[1].astype(col_dtype, copy=False)
        sparse_nnz = int(values.size)
        additions = [
            _Stream(
                f"{layer.name}/sparse_values",
                layer.name,
                "sparse_values",
                "csr_row_major_values",
                "float16",
                (sparse_nnz,),
                sparse_nnz * 16,
                _little_endian_bytes(values, "<f2"),
            ),
            _Stream(
                f"{layer.name}/sparse_row_ptr",
                layer.name,
                "sparse_row_ptr",
                "csr_fixed",
                "uint32",
                (rows + 1,),
                (rows + 1) * 32,
                _little_endian_bytes(row_ptr, "<u4"),
            ),
            _Stream(
                f"{layer.name}/sparse_col_idx",
                layer.name,
                "sparse_col_idx",
                "csr_fixed",
                str(col_dtype),
                (sparse_nnz,),
                sparse_nnz * col_dtype.itemsize * 8,
                _little_endian_bytes(col_idx, col_dtype.str),
            ),
        ]
        streams.extend(additions)
        for stream in additions:
            components[stream.component] = stream.name
    rank = 0
    if layer.lowrank_left is not None:
        factor_bits = int(layer.lowrank_factor_bits)
        left = np.asarray(layer.lowrank_left)
        right = np.asarray(layer.lowrank_right)
        rank = int(left.shape[1])
        if factor_bits == 16:
            left = np.asarray(left, dtype=np.float16)
            right = np.asarray(right, dtype=np.float16)
            additions = [
                _Stream(
                    f"{layer.name}/lowrank_left",
                    layer.name,
                    "lowrank_left",
                    "raw_little_endian",
                    "float16",
                    tuple(map(int, left.shape)),
                    int(left.size) * 16,
                    _little_endian_bytes(left, "<f2"),
                ),
                _Stream(
                    f"{layer.name}/lowrank_right",
                    layer.name,
                    "lowrank_right",
                    "raw_little_endian",
                    "float16",
                    tuple(map(int, right.shape)),
                    int(right.size) * 16,
                    _little_endian_bytes(right, "<f2"),
                ),
            ]
        else:
            assert (
                layer.lowrank_left_scales is not None
                and layer.lowrank_right_scales is not None
            )
            left_scales = np.asarray(layer.lowrank_left_scales, dtype=np.float16)
            right_scales = np.asarray(layer.lowrank_right_scales, dtype=np.float16)
            additions = [
                _Stream(
                    f"{layer.name}/lowrank_left",
                    layer.name,
                    "lowrank_left",
                    f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                    "bitpack",
                    tuple(map(int, left.shape)),
                    int(left.size) * factor_bits,
                    pack_signed_codes(left, factor_bits),
                ),
                _Stream(
                    f"{layer.name}/lowrank_left_scales",
                    layer.name,
                    "lowrank_left_scales",
                    "raw_little_endian",
                    "float16",
                    tuple(map(int, left_scales.shape)),
                    int(left_scales.size) * 16,
                    _little_endian_bytes(left_scales, "<f2"),
                ),
                _Stream(
                    f"{layer.name}/lowrank_right",
                    layer.name,
                    "lowrank_right",
                    f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                    "bitpack",
                    tuple(map(int, right.shape)),
                    int(right.size) * factor_bits,
                    pack_signed_codes(right, factor_bits),
                ),
                _Stream(
                    f"{layer.name}/lowrank_right_scales",
                    layer.name,
                    "lowrank_right_scales",
                    "raw_little_endian",
                    "float16",
                    tuple(map(int, right_scales.shape)),
                    int(right_scales.size) * 16,
                    _little_endian_bytes(right_scales, "<f2"),
                ),
            ]
            components["lowrank_left_scales"] = additions[1].name
            components["lowrank_right_scales"] = additions[3].name
        streams.extend(additions)
        for stream in additions:
            components[stream.component] = stream.name
    record: dict[str, object] = {
        "name": layer.name,
        "shape": [rows, cols],
        "q_bits": int(layer.q_bits),
        "q_col_block_size": None if layer.q_col_block_size is None else int(layer.q_col_block_size),
        "sparse_nnz": sparse_nnz,
        "lowrank_rank": rank,
        "components": components,
    }
    if rank and int(layer.lowrank_factor_bits) != 16:
        record["lowrank_factor_bits"] = int(layer.lowrank_factor_bits)
    return streams, record


def _artifact_layout(
    *,
    kind: str,
    layer_records: list[dict[str, object]],
    streams: Sequence[_Stream | _SizedStream],
    alignment: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    bytes,
    int,
    int,
    int,
    int,
    int,
]:
    """Build the canonical layout shared by measurement and serialization."""

    alignment = int(alignment)
    _align_up(0, alignment)
    offset = 0
    stream_records: list[dict[str, object]] = []
    for stream in streams:
        nbytes = _stream_nbytes(stream)
        aligned = _align_up(offset, alignment)
        stream_records.append(
            {
                "name": stream.name,
                "layer": stream.layer,
                "component": stream.component,
                "encoding": stream.encoding,
                "dtype": stream.dtype,
                "shape": list(stream.shape),
                "offset": aligned,
                "nbytes": nbytes,
                "logical_bits": int(stream.logical_bits),
                "sha256": _stream_sha256(stream),
            }
        )
        offset = aligned + nbytes
    manifest: dict[str, object] = {
        "format": "llm_spectral_dynamics_research_codec",
        "version": FORMAT_VERSION,
        "kind": kind,
        "offset_semantics": "relative_to_aligned_payload_base",
        "alignment_bytes": alignment,
        "layers": layer_records,
        "streams": stream_records,
        "transparent_compression": "none",
    }
    header = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload_base = _align_up(PREFIX.size + len(header), alignment)
    natural_size = payload_base + offset
    stream_bytes = sum(_stream_nbytes(stream) for stream in streams)
    internal_padding = offset - stream_bytes
    logical_bits = sum(int(stream.logical_bits) for stream in streams)
    return (
        stream_records,
        manifest,
        header,
        payload_base,
        natural_size,
        stream_bytes,
        internal_padding,
        logical_bits,
    )


def _collect_codec_layers(
    layers: Sequence[LayerCodecPayload],
) -> tuple[list[_Stream], list[dict[str, object]]]:
    if not layers:
        raise ValueError("at least one layer is required")
    ordered = sorted(layers, key=lambda layer: layer.name)
    if len({layer.name for layer in ordered}) != len(ordered):
        raise ValueError("layer names must be unique")
    streams: list[_Stream] = []
    records: list[dict[str, object]] = []
    for layer in ordered:
        layer_streams, record = _layer_streams(layer)
        streams.extend(layer_streams)
        records.append(record)
    return streams, records


def codec_artifact_natural_file_bytes(
    layers: Sequence[LayerCodecPayload],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> int:
    """Return exact serialized bytes without writing an artifact.

    This executes the same packing, canonical-manifest and alignment path used
    by :func:`write_codec_artifact`; it is therefore suitable for discrete
    allocation searches that must charge physical rather than nominal bits.
    """

    streams, records = _collect_codec_layers(layers)
    layout = _artifact_layout(
        kind="qsl_selected_linear_weights",
        layer_records=records,
        streams=streams,
        alignment=alignment,
    )
    return int(layout[4])


def codec_artifact_allocation_natural_file_bytes(
    *,
    name: str,
    shape: tuple[int, int],
    q_bits: int,
    q_scale_shape: tuple[int, ...],
    q_col_block_size: int | None,
    sparse_nnz: int,
    lowrank_rank: int,
    lowrank_factor_bits: int = 16,
    lowrank_left_scale_shape: tuple[int, ...] = (),
    lowrank_right_scale_shape: tuple[int, ...] = (),
    alignment: int = DEFAULT_ALIGNMENT,
) -> int:
    """Measure a one-layer allocation from metadata alone.

    SHA256 values have a fixed 64-character representation, so numerical
    payload values cannot change the canonical manifest length.  This fast
    path is byte-identical to full serialization while avoiding repeated
    bit-packing during a discrete nnz/rank search.
    """

    rows, cols = map(int, shape)
    bits = int(q_bits)
    nonzero = int(sparse_nnz)
    rank = int(lowrank_rank)
    factor_bits = int(lowrank_factor_bits)
    scale_shape = tuple(map(int, q_scale_shape))
    left_scale_shape = tuple(map(int, lowrank_left_scale_shape))
    right_scale_shape = tuple(map(int, lowrank_right_scale_shape))
    if not name:
        raise ValueError("name must be non-empty")
    if rows <= 0 or cols <= 0 or bits <= 0 or bits > 8:
        raise ValueError("invalid shape or q_bits")
    if nonzero < 0 or nonzero > rows * cols:
        raise ValueError("sparse_nnz is outside the tensor extent")
    if rank < 0 or rank > min(rows, cols):
        raise ValueError("lowrank_rank is outside the matrix dimensions")
    if factor_bits != 16 and (factor_bits < 2 or factor_bits > 8):
        raise ValueError("lowrank_factor_bits must be 16 or within [2, 8]")
    expected_left_scale_shape = () if rank == 0 or factor_bits == 16 else (rows,)
    expected_right_scale_shape = () if rank == 0 or factor_bits == 16 else (rank,)
    if left_scale_shape != expected_left_scale_shape:
        raise ValueError("lowrank_left_scale_shape differs from the factor codec")
    if right_scale_shape != expected_right_scale_shape:
        raise ValueError("lowrank_right_scale_shape differs from the factor codec")
    if q_col_block_size is not None and int(q_col_block_size) <= 0:
        raise ValueError("q_col_block_size must be positive")
    expected_scale_shape = (
        (rows,)
        if q_col_block_size is None
        else (rows, (cols + int(q_col_block_size) - 1) // int(q_col_block_size))
    )
    if scale_shape != expected_scale_shape:
        raise ValueError(
            f"q_scale_shape mismatch: expected {expected_scale_shape}, got {scale_shape}"
        )

    components: dict[str, str | None] = {
        "q_codes": f"{name}/q_codes",
        "q_scales": f"{name}/q_scales",
        "sparse_values": None,
        "sparse_row_ptr": None,
        "sparse_col_idx": None,
        "lowrank_left": None,
        "lowrank_right": None,
    }
    specs: list[tuple[str, str, str, str, tuple[int, ...], int, int]] = [
        (
            components["q_codes"],
            "q_codes",
            f"signed_twos_complement_lsb_bitpack_{bits}",
            "bitpack",
            (rows, cols),
            (rows * cols * bits + 7) // 8,
            rows * cols * bits,
        ),
        (
            components["q_scales"],
            "q_scales",
            "raw_little_endian",
            "float16",
            scale_shape,
            math.prod(scale_shape) * 2,
            math.prod(scale_shape) * 16,
        ),
    ]
    if nonzero:
        col_dtype = _index_dtype(cols)
        sparse_specs = [
            (f"{name}/sparse_values", "sparse_values", "csr_row_major_values", "float16", (nonzero,), nonzero * 2, nonzero * 16),
            (f"{name}/sparse_row_ptr", "sparse_row_ptr", "csr_fixed", "uint32", (rows + 1,), (rows + 1) * 4, (rows + 1) * 32),
            (
                f"{name}/sparse_col_idx",
                "sparse_col_idx",
                "csr_fixed",
                str(col_dtype),
                (nonzero,),
                nonzero * col_dtype.itemsize,
                nonzero * col_dtype.itemsize * 8,
            ),
        ]
        specs.extend(sparse_specs)
        for stream_name, component, *_ in sparse_specs:
            components[component] = stream_name
    if rank:
        if factor_bits == 16:
            lowrank_specs = [
                (f"{name}/lowrank_left", "lowrank_left", "raw_little_endian", "float16", (rows, rank), rows * rank * 2, rows * rank * 16),
                (f"{name}/lowrank_right", "lowrank_right", "raw_little_endian", "float16", (rank, cols), rank * cols * 2, rank * cols * 16),
            ]
        else:
            lowrank_specs = [
                (
                    f"{name}/lowrank_left",
                    "lowrank_left",
                    f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                    "bitpack",
                    (rows, rank),
                    (rows * rank * factor_bits + 7) // 8,
                    rows * rank * factor_bits,
                ),
                (
                    f"{name}/lowrank_left_scales",
                    "lowrank_left_scales",
                    "raw_little_endian",
                    "float16",
                    left_scale_shape,
                    math.prod(left_scale_shape) * 2,
                    math.prod(left_scale_shape) * 16,
                ),
                (
                    f"{name}/lowrank_right",
                    "lowrank_right",
                    f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                    "bitpack",
                    (rank, cols),
                    (rank * cols * factor_bits + 7) // 8,
                    rank * cols * factor_bits,
                ),
                (
                    f"{name}/lowrank_right_scales",
                    "lowrank_right_scales",
                    "raw_little_endian",
                    "float16",
                    right_scale_shape,
                    math.prod(right_scale_shape) * 2,
                    math.prod(right_scale_shape) * 16,
                ),
            ]
            components["lowrank_left_scales"] = f"{name}/lowrank_left_scales"
            components["lowrank_right_scales"] = f"{name}/lowrank_right_scales"
        specs.extend(lowrank_specs)
        for stream_name, component, *_ in lowrank_specs:
            components[component] = stream_name

    offset = 0
    stream_records: list[dict[str, object]] = []
    for stream_name, component, encoding, dtype, stream_shape, nbytes, logical_bits in specs:
        aligned = _align_up(offset, int(alignment))
        stream_records.append(
            {
                "name": stream_name,
                "layer": name,
                "component": component,
                "encoding": encoding,
                "dtype": dtype,
                "shape": list(stream_shape),
                "offset": aligned,
                "nbytes": int(nbytes),
                "logical_bits": int(logical_bits),
                "sha256": "0" * 64,
            }
        )
        offset = aligned + int(nbytes)
    layer_record = {
        "name": name,
        "shape": [rows, cols],
        "q_bits": bits,
        "q_col_block_size": None if q_col_block_size is None else int(q_col_block_size),
        "sparse_nnz": nonzero,
        "lowrank_rank": rank,
        "components": components,
    }
    if rank and factor_bits != 16:
        layer_record["lowrank_factor_bits"] = factor_bits
    manifest = {
        "format": "llm_spectral_dynamics_research_codec",
        "version": FORMAT_VERSION,
        "kind": "qsl_selected_linear_weights",
        "offset_semantics": "relative_to_aligned_payload_base",
        "alignment_bytes": int(alignment),
        "layers": [layer_record],
        "streams": stream_records,
        "transparent_compression": "none",
    }
    header = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _align_up(PREFIX.size + len(header), int(alignment)) + offset


def codec_artifact_allocations_layout(
    layers: Sequence[LayerCodecAllocation],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> CodecArtifactAllocationLayout:
    """Return exact canonical layout statistics from allocation metadata.

    Payload values only affect the 64-character SHA256 strings in the
    canonical manifest, never its byte length.  Consequently the layout below
    is byte-identical to :func:`codec_artifact_natural_file_bytes` while it
    avoids touching or packing full weight arrays during allocation search.
    Final selected endpoints are still cross-checked with the real serializer.
    """

    if not layers:
        raise ValueError("at least one layer is required")
    alignment = int(alignment)
    _align_up(0, alignment)
    ordered = sorted(layers, key=lambda layer: layer.name)
    if len({layer.name for layer in ordered}) != len(ordered):
        raise ValueError("layer names must be unique")

    layer_records: list[dict[str, object]] = []
    specs: list[tuple[str, str, str, str, tuple[int, ...], int, int]] = []
    stream_layers: dict[str, str] = {}
    for layer in ordered:
        if not layer.name:
            raise ValueError("layer name must be non-empty")
        rows, cols = map(int, layer.shape)
        bits = int(layer.q_bits)
        nonzero = int(layer.sparse_nnz)
        rank = int(layer.lowrank_rank)
        factor_bits = int(layer.lowrank_factor_bits)
        scale_shape = tuple(map(int, layer.q_scale_shape))
        left_scale_shape = tuple(map(int, layer.lowrank_left_scale_shape))
        right_scale_shape = tuple(map(int, layer.lowrank_right_scale_shape))
        if rows <= 0 or cols <= 0:
            raise ValueError("layer dimensions must be positive")
        if bits <= 0 or bits > 8:
            raise ValueError("bits must be within [1, 8]")
        if nonzero < 0 or nonzero > rows * cols:
            raise ValueError("sparse_nnz is outside the tensor extent")
        if rank < 0 or rank > min(rows, cols):
            raise ValueError("lowrank_rank is outside the matrix dimensions")
        if factor_bits != 16 and (factor_bits < 2 or factor_bits > 8):
            raise ValueError("lowrank_factor_bits must be 16 or within [2, 8]")
        expected_left_scale_shape = (
            () if rank == 0 or factor_bits == 16 else (rows,)
        )
        expected_right_scale_shape = (
            () if rank == 0 or factor_bits == 16 else (rank,)
        )
        if left_scale_shape != expected_left_scale_shape:
            raise ValueError("lowrank_left_scale_shape differs from the factor codec")
        if right_scale_shape != expected_right_scale_shape:
            raise ValueError("lowrank_right_scale_shape differs from the factor codec")
        if layer.q_col_block_size is not None and int(layer.q_col_block_size) <= 0:
            raise ValueError("q_col_block_size must be positive")
        expected_scale_shape = (
            (rows,)
            if layer.q_col_block_size is None
            else (rows, (cols + int(layer.q_col_block_size) - 1) // int(layer.q_col_block_size))
        )
        if scale_shape != expected_scale_shape:
            raise ValueError(
                f"q_scale_shape mismatch: expected {expected_scale_shape}, got {scale_shape}"
            )

        components: dict[str, str | None] = {
            "q_codes": f"{layer.name}/q_codes",
            "q_scales": f"{layer.name}/q_scales",
            "sparse_values": None,
            "sparse_row_ptr": None,
            "sparse_col_idx": None,
            "lowrank_left": None,
            "lowrank_right": None,
        }
        layer_specs: list[tuple[str, str, str, str, tuple[int, ...], int, int]] = [
            (
                components["q_codes"],
                "q_codes",
                f"signed_twos_complement_lsb_bitpack_{bits}",
                "bitpack",
                (rows, cols),
                (rows * cols * bits + 7) // 8,
                rows * cols * bits,
            ),
            (
                components["q_scales"],
                "q_scales",
                "raw_little_endian",
                "float16",
                scale_shape,
                math.prod(scale_shape) * 2,
                math.prod(scale_shape) * 16,
            ),
        ]
        if nonzero:
            col_dtype = _index_dtype(cols)
            sparse_specs = [
                (
                    f"{layer.name}/sparse_values",
                    "sparse_values",
                    "csr_row_major_values",
                    "float16",
                    (nonzero,),
                    nonzero * 2,
                    nonzero * 16,
                ),
                (
                    f"{layer.name}/sparse_row_ptr",
                    "sparse_row_ptr",
                    "csr_fixed",
                    "uint32",
                    (rows + 1,),
                    (rows + 1) * 4,
                    (rows + 1) * 32,
                ),
                (
                    f"{layer.name}/sparse_col_idx",
                    "sparse_col_idx",
                    "csr_fixed",
                    str(col_dtype),
                    (nonzero,),
                    nonzero * col_dtype.itemsize,
                    nonzero * col_dtype.itemsize * 8,
                ),
            ]
            layer_specs.extend(sparse_specs)
            for stream_name, component, *_ in sparse_specs:
                components[component] = stream_name
        if rank:
            if factor_bits == 16:
                lowrank_specs = [
                    (
                        f"{layer.name}/lowrank_left",
                        "lowrank_left",
                        "raw_little_endian",
                        "float16",
                        (rows, rank),
                        rows * rank * 2,
                        rows * rank * 16,
                    ),
                    (
                        f"{layer.name}/lowrank_right",
                        "lowrank_right",
                        "raw_little_endian",
                        "float16",
                        (rank, cols),
                        rank * cols * 2,
                        rank * cols * 16,
                    ),
                ]
            else:
                lowrank_specs = [
                    (
                        f"{layer.name}/lowrank_left",
                        "lowrank_left",
                        f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                        "bitpack",
                        (rows, rank),
                        (rows * rank * factor_bits + 7) // 8,
                        rows * rank * factor_bits,
                    ),
                    (
                        f"{layer.name}/lowrank_left_scales",
                        "lowrank_left_scales",
                        "raw_little_endian",
                        "float16",
                        left_scale_shape,
                        math.prod(left_scale_shape) * 2,
                        math.prod(left_scale_shape) * 16,
                    ),
                    (
                        f"{layer.name}/lowrank_right",
                        "lowrank_right",
                        f"signed_twos_complement_lsb_bitpack_{factor_bits}",
                        "bitpack",
                        (rank, cols),
                        (rank * cols * factor_bits + 7) // 8,
                        rank * cols * factor_bits,
                    ),
                    (
                        f"{layer.name}/lowrank_right_scales",
                        "lowrank_right_scales",
                        "raw_little_endian",
                        "float16",
                        right_scale_shape,
                        math.prod(right_scale_shape) * 2,
                        math.prod(right_scale_shape) * 16,
                    ),
                ]
                components["lowrank_left_scales"] = (
                    f"{layer.name}/lowrank_left_scales"
                )
                components["lowrank_right_scales"] = (
                    f"{layer.name}/lowrank_right_scales"
                )
            layer_specs.extend(lowrank_specs)
            for stream_name, component, *_ in lowrank_specs:
                components[component] = stream_name
        specs.extend(layer_specs)
        for stream_name, *_ in layer_specs:
            stream_layers[stream_name] = layer.name
        layer_record: dict[str, object] = {
            "name": layer.name,
            "shape": [rows, cols],
            "q_bits": bits,
            "q_col_block_size": (
                None if layer.q_col_block_size is None else int(layer.q_col_block_size)
            ),
            "sparse_nnz": nonzero,
            "lowrank_rank": rank,
            "components": components,
        }
        if rank and factor_bits != 16:
            layer_record["lowrank_factor_bits"] = factor_bits
        layer_records.append(layer_record)

    sized_streams = [
        _SizedStream(
            name=stream_name,
            layer=stream_layers[stream_name],
            component=component,
            encoding=encoding,
            dtype=dtype,
            shape=shape,
            logical_bits=int(logical_bits),
            nbytes=int(nbytes),
        )
        for stream_name, component, encoding, dtype, shape, nbytes, logical_bits in specs
    ]
    layout = _artifact_layout(
        kind="qsl_selected_linear_weights",
        layer_records=layer_records,
        streams=sized_streams,
        alignment=alignment,
    )
    return CodecArtifactAllocationLayout(
        natural_file_bytes=int(layout[4]),
        header_bytes=len(layout[2]),
        payload_base_bytes=int(layout[3]),
        payload_end_bytes=int(layout[4]) - int(layout[3]),
        stream_bytes=int(layout[5]),
        internal_padding_bytes=int(layout[6]),
        logical_payload_bits=int(layout[7]),
    )


def codec_artifact_allocations_natural_file_bytes(
    layers: Sequence[LayerCodecAllocation],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> int:
    """Measure a complete multi-layer allocation from metadata alone."""

    return codec_artifact_allocations_layout(
        layers, alignment=alignment
    ).natural_file_bytes


def _write_artifact(
    path: str | os.PathLike[str],
    *,
    kind: str,
    layer_records: list[dict[str, object]],
    streams: Sequence[_Stream],
    alignment: int,
    target_file_bytes: int | None,
) -> ArtifactWriteResult:
    alignment = int(alignment)
    (
        stream_records,
        manifest,
        header,
        payload_base,
        natural_size,
        stream_bytes,
        internal_padding,
        logical_bits,
    ) = _artifact_layout(
        kind=kind,
        layer_records=layer_records,
        streams=streams,
        alignment=alignment,
    )
    requested = natural_size if target_file_bytes is None else int(target_file_bytes)
    if requested < natural_size:
        raise ValueError(f"target_file_bytes {requested} is smaller than natural artifact {natural_size}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(PREFIX.pack(MAGIC, FORMAT_VERSION, alignment, len(header)))
        handle.write(header)
        handle.write(b"\0" * (payload_base - handle.tell()))
        for stream, record in zip(streams, stream_records, strict=True):
            absolute = payload_base + int(record["offset"])
            handle.write(b"\0" * (absolute - handle.tell()))
            handle.write(stream.payload)
        handle.write(b"\0" * (requested - handle.tell()))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    raw = target.read_bytes()
    tail_padding = requested - natural_size
    return ArtifactWriteResult(
        path=target,
        file_bytes=len(raw),
        logical_payload_bits=logical_bits,
        stream_bytes=stream_bytes,
        container_bytes=payload_base,
        alignment_padding_bytes=(payload_base - PREFIX.size - len(header)) + internal_padding,
        tail_padding_bytes=tail_padding,
        sha256=hashlib.sha256(raw).hexdigest(),
        manifest=manifest,
    )


def write_codec_artifact(
    path: str | os.PathLike[str],
    layers: Sequence[LayerCodecPayload],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    target_file_bytes: int | None = None,
) -> ArtifactWriteResult:
    streams, records = _collect_codec_layers(layers)
    return _write_artifact(
        path,
        kind="qsl_selected_linear_weights",
        layer_records=records,
        streams=streams,
        alignment=alignment,
        target_file_bytes=target_file_bytes,
    )


def write_fp16_reference_artifact(
    path: str | os.PathLike[str],
    weights: Mapping[str, np.ndarray],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> ArtifactWriteResult:
    if not weights:
        raise ValueError("at least one reference tensor is required")
    records: list[dict[str, object]] = []
    streams: list[_Stream] = []
    for name in sorted(weights):
        value = np.asarray(weights[name])
        if value.ndim != 2:
            raise ValueError("reference tensors must be matrices")
        payload = _little_endian_bytes(value, "<f2")
        stream_name = f"{name}/dense_fp16"
        streams.append(
            _Stream(
                stream_name,
                name,
                "dense_fp16",
                "raw_little_endian",
                "float16",
                tuple(map(int, value.shape)),
                int(value.size) * 16,
                payload,
            )
        )
        records.append(
            {
                "name": name,
                "shape": list(map(int, value.shape)),
                "components": {"dense_fp16": stream_name},
            }
        )
    return _write_artifact(
        path,
        kind="fp16_selected_linear_reference",
        layer_records=records,
        streams=streams,
        alignment=alignment,
        target_file_bytes=None,
    )


def _read_container(path: str | os.PathLike[str]) -> tuple[bytes, dict[str, object], int, int]:
    raw = Path(path).read_bytes()
    if len(raw) < PREFIX.size:
        raise ValueError("artifact is shorter than its prefix")
    magic, version, alignment, header_size = PREFIX.unpack_from(raw)
    if magic != MAGIC or int(version) != FORMAT_VERSION:
        raise ValueError("unsupported artifact magic or version")
    header_end = PREFIX.size + int(header_size)
    if header_end > len(raw):
        raise ValueError("truncated artifact manifest")
    manifest = json.loads(raw[PREFIX.size:header_end].decode("utf-8"))
    payload_base = _align_up(header_end, int(alignment))
    return raw, manifest, payload_base, int(alignment)


def read_codec_artifact(path: str | os.PathLike[str]) -> DecodedArtifact:
    raw, manifest, payload_base, alignment = _read_container(path)
    if manifest.get("kind") != "qsl_selected_linear_weights":
        raise ValueError("artifact is not a Q/S/L codec")
    if int(manifest.get("alignment_bytes", -1)) != alignment:
        raise ValueError("prefix/manifest alignment mismatch")
    stream_payloads: dict[str, bytes] = {}
    stream_records = {str(record["name"]): record for record in manifest["streams"]}
    for name, record in stream_records.items():
        start = payload_base + int(record["offset"])
        stop = start + int(record["nbytes"])
        if start % alignment or stop > len(raw):
            raise ValueError(f"invalid stream bounds for {name}")
        payload = raw[start:stop]
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"stream checksum mismatch for {name}")
        stream_payloads[name] = payload
    decoded: dict[str, np.ndarray] = {}
    for layer in manifest["layers"]:
        name = str(layer["name"])
        rows, cols = map(int, layer["shape"])
        components = layer["components"]
        code_name = str(components["q_codes"])
        scale_name = str(components["q_scales"])
        codes = unpack_signed_codes(
            stream_payloads[code_name], count=rows * cols, bits=int(layer["q_bits"])
        ).reshape(rows, cols)
        scale_record = stream_records[scale_name]
        scales = np.frombuffer(stream_payloads[scale_name], dtype="<f2").reshape(scale_record["shape"])
        block = layer["q_col_block_size"]
        if block is None:
            quantized = codes.astype(np.float32) * scales.astype(np.float32).reshape(rows, 1)
        else:
            quantized = np.empty((rows, cols), dtype=np.float32)
            for group in range(scales.shape[1]):
                start = group * int(block)
                stop = min((group + 1) * int(block), cols)
                quantized[:, start:stop] = (
                    codes[:, start:stop].astype(np.float32) * scales[:, group : group + 1].astype(np.float32)
                )
        quantized = quantized.astype(np.float16).astype(np.float32)
        sparse = np.zeros((rows, cols), dtype=np.float32)
        if components["sparse_values"] is not None:
            value_name = str(components["sparse_values"])
            row_name = str(components["sparse_row_ptr"])
            col_name = str(components["sparse_col_idx"])
            values = np.frombuffer(stream_payloads[value_name], dtype="<f2").astype(np.float32)
            row_ptr = np.frombuffer(stream_payloads[row_name], dtype="<u4")
            col_record = stream_records[col_name]
            col_dtype = np.dtype(str(col_record["dtype"]))
            col_idx = np.frombuffer(stream_payloads[col_name], dtype=col_dtype).astype(np.int64)
            if row_ptr.size != rows + 1 or int(row_ptr[-1]) != values.size or col_idx.size != values.size:
                raise ValueError(f"invalid CSR structure for {name}")
            if np.any(row_ptr[1:] < row_ptr[:-1]) or (col_idx.size and int(col_idx.max()) >= cols):
                raise ValueError(f"invalid CSR indices for {name}")
            for row in range(rows):
                start, stop = int(row_ptr[row]), int(row_ptr[row + 1])
                sparse[row, col_idx[start:stop]] = values[start:stop]
        lowrank = np.zeros((rows, cols), dtype=np.float32)
        if components["lowrank_left"] is not None:
            left_name = str(components["lowrank_left"])
            right_name = str(components["lowrank_right"])
            left_record = stream_records[left_name]
            right_record = stream_records[right_name]
            factor_bits = int(layer.get("lowrank_factor_bits", 16))
            if factor_bits == 16:
                left = np.frombuffer(
                    stream_payloads[left_name], dtype="<f2"
                ).reshape(left_record["shape"])
                right = np.frombuffer(
                    stream_payloads[right_name], dtype="<f2"
                ).reshape(right_record["shape"])
            else:
                left = unpack_signed_codes(
                    stream_payloads[left_name],
                    count=math.prod(left_record["shape"]),
                    bits=factor_bits,
                ).reshape(left_record["shape"])
                right = unpack_signed_codes(
                    stream_payloads[right_name],
                    count=math.prod(right_record["shape"]),
                    bits=factor_bits,
                ).reshape(right_record["shape"])
                left_scale_name = str(components["lowrank_left_scales"])
                right_scale_name = str(components["lowrank_right_scales"])
                left_scales = np.frombuffer(
                    stream_payloads[left_scale_name], dtype="<f2"
                )
                right_scales = np.frombuffer(
                    stream_payloads[right_scale_name], dtype="<f2"
                )
                left = left.astype(np.float32) * left_scales.astype(
                    np.float32
                ).reshape(-1, 1)
                right = right.astype(np.float32) * right_scales.astype(
                    np.float32
                ).reshape(-1, 1)
            lowrank = (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16).astype(np.float32)
        decoded[name] = (quantized + sparse + lowrank).astype(np.float16).astype(np.float32)
    return DecodedArtifact(
        manifest=manifest,
        layers=decoded,
        file_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def read_fp16_reference_artifact(path: str | os.PathLike[str]) -> DecodedArtifact:
    raw, manifest, payload_base, alignment = _read_container(path)
    if manifest.get("kind") != "fp16_selected_linear_reference":
        raise ValueError("artifact is not an FP16 reference")
    records = {str(record["name"]): record for record in manifest["streams"]}
    layers: dict[str, np.ndarray] = {}
    for layer in manifest["layers"]:
        name = str(layer["name"])
        stream_name = str(layer["components"]["dense_fp16"])
        record = records[stream_name]
        start = payload_base + int(record["offset"])
        stop = start + int(record["nbytes"])
        payload = raw[start:stop]
        if start % alignment or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"invalid reference stream for {name}")
        layers[name] = np.frombuffer(payload, dtype="<f2").reshape(record["shape"]).astype(np.float32)
    return DecodedArtifact(
        manifest=manifest,
        layers=layers,
        file_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
