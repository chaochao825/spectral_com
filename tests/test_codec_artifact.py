from __future__ import annotations

import numpy as np
import pytest

import llm_spectral_dynamics.structured.codec_artifact as codec
from llm_spectral_dynamics.structured.codec_artifact import (
    LayerCodecAllocation,
    LayerCodecPayload,
    codec_artifact_allocation_natural_file_bytes,
    codec_artifact_allocations_layout,
    codec_artifact_allocations_natural_file_bytes,
    codec_artifact_natural_file_bytes,
    pack_signed_codes,
    read_codec_artifact,
    read_fp16_reference_artifact,
    unpack_signed_codes,
    write_codec_artifact,
    write_fp16_reference_artifact,
)


@pytest.mark.parametrize(
    ("bits", "values"),
    [
        (1, [-1, 0, -1]),
        (2, [-2, -1, 0, 1, -2]),
        (4, [-8, -7, -1, 0, 1, 7, -3]),
        (8, [-128, -1, 0, 1, 127]),
    ],
)
def test_signed_bitpack_roundtrip(bits: int, values: list[int]) -> None:
    source = np.asarray(values, dtype=np.int8)
    packed = pack_signed_codes(source, bits)
    assert len(packed) == (source.size * bits + 7) // 8
    np.testing.assert_array_equal(unpack_signed_codes(packed, count=source.size, bits=bits), source)


@pytest.mark.parametrize("bits", range(1, 9))
def test_chunked_signed_bitpack_matches_reference(
    bits: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    minimum = -(1 << (bits - 1))
    span = 1 << bits
    source = (minimum + (np.arange(73, dtype=np.int64) * 7) % span).astype(np.int16)
    unsigned = source.astype(np.int64) & (span - 1)
    columns = ((unsigned[:, None] >> np.arange(bits, dtype=np.int64)) & 1).astype(np.uint8)
    reference = np.packbits(columns.reshape(-1), bitorder="little").tobytes()
    monkeypatch.setattr(codec, "PACK_CHUNK_ELEMENTS", 5)
    assert pack_signed_codes(source, bits) == reference


def test_multilayer_container_bytes_are_not_additive() -> None:
    allocation_x = [
        LayerCodecAllocation("a", (3, 7), 2, (3,), lowrank_rank=0),
        LayerCodecAllocation("b", (4, 17), 2, (4,), lowrank_rank=1),
    ]
    allocation_y = [
        LayerCodecAllocation("a", (3, 7), 2, (3,), lowrank_rank=3),
        LayerCodecAllocation("b", (4, 17), 2, (4,), lowrank_rank=0),
    ]
    additive_x = sum(
        codec_artifact_allocations_natural_file_bytes([layer], alignment=64)
        for layer in allocation_x
    )
    additive_y = sum(
        codec_artifact_allocations_natural_file_bytes([layer], alignment=64)
        for layer in allocation_y
    )
    exact_x = codec_artifact_allocations_layout(allocation_x, alignment=64)
    exact_y = codec_artifact_allocations_layout(allocation_y, alignment=64)
    assert additive_x == 2856 and additive_y == 2866
    assert exact_x.natural_file_bytes == 2658
    assert exact_y.natural_file_bytes == 2632
    assert additive_x < additive_y
    assert exact_x.natural_file_bytes > exact_y.natural_file_bytes


def test_codec_artifact_roundtrip_and_real_byte_accounting(tmp_path) -> None:
    codes_a = np.array([[-7, -1, 0, 2, 7], [1, -2, 3, -4, 5]], dtype=np.int8)
    scales_a = np.array([0.25, 0.5], dtype=np.float16)
    mask = np.array([[True, False, False, True, False], [False, True, False, False, False]])
    sparse_values = np.zeros((2, 5), dtype=np.float16)
    sparse_values[mask] = np.array([0.125, -0.25, 0.375], dtype=np.float16)
    left = np.array([[0.5], [-0.25]], dtype=np.float16)
    right = np.array([[0.25, -0.5, 0.75, 0.0, 0.125]], dtype=np.float16)

    codes_b = np.array([[-2, 0, 1, 3, -1], [2, -3, 0, 1, -2]], dtype=np.int8)
    scales_b = np.array([[0.25, 0.5, 0.75], [0.125, 0.25, 0.5]], dtype=np.float16)
    layers = [
        LayerCodecPayload(
            "layer.a",
            codes_a,
            scales_a,
            4,
            sparse_values=sparse_values,
            sparse_mask=mask,
            lowrank_left=left,
            lowrank_right=right,
        ),
        LayerCodecPayload("layer.b", codes_b, scales_b, 4, q_col_block_size=2),
    ]
    target = tmp_path / "candidate.hrc"
    result = write_codec_artifact(target, layers, alignment=64)
    decoded = read_codec_artifact(target)

    q_a = (codes_a.astype(np.float32) * scales_a.astype(np.float32)[:, None]).astype(np.float16).astype(np.float32)
    s_a = np.where(mask, sparse_values.astype(np.float32), 0.0)
    l_a = (left.astype(np.float32) @ right.astype(np.float32)).astype(np.float16).astype(np.float32)
    expected_a = (q_a + s_a + l_a).astype(np.float16).astype(np.float32)
    expected_b = np.empty((2, 5), dtype=np.float32)
    for group in range(3):
        start, stop = group * 2, min((group + 1) * 2, 5)
        expected_b[:, start:stop] = codes_b[:, start:stop] * scales_b[:, group : group + 1]
    expected_b = expected_b.astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(decoded.layers["layer.a"], expected_a)
    np.testing.assert_array_equal(decoded.layers["layer.b"], expected_b)
    assert decoded.sha256 == result.sha256
    assert result.file_bytes == target.stat().st_size
    assert codec_artifact_natural_file_bytes(layers, alignment=64) == result.file_bytes
    allocations = [
        LayerCodecAllocation(
            name=layer.name,
            shape=layer.shape,
            q_bits=layer.q_bits,
            q_scale_shape=tuple(layer.q_scales.shape),
            q_col_block_size=layer.q_col_block_size,
            sparse_nnz=(
                0 if layer.sparse_mask is None else int(np.count_nonzero(layer.sparse_mask))
            ),
            lowrank_rank=(
                0 if layer.lowrank_left is None else int(layer.lowrank_left.shape[1])
            ),
        )
        for layer in layers
    ]
    assert (
        codec_artifact_allocations_natural_file_bytes(allocations, alignment=64)
        == result.file_bytes
    )
    for layer in layers:
        assert codec_artifact_allocation_natural_file_bytes(
            name=layer.name,
            shape=layer.shape,
            q_bits=layer.q_bits,
            q_scale_shape=tuple(layer.q_scales.shape),
            q_col_block_size=layer.q_col_block_size,
            sparse_nnz=0 if layer.sparse_mask is None else int(np.count_nonzero(layer.sparse_mask)),
            lowrank_rank=0 if layer.lowrank_left is None else int(layer.lowrank_left.shape[1]),
            alignment=64,
        ) == codec_artifact_natural_file_bytes([layer], alignment=64)
    assert result.file_bytes * 8 > result.logical_payload_bits
    assert all(int(stream["offset"]) % 64 == 0 for stream in result.manifest["streams"])

    padded = tmp_path / "candidate_padded.hrc"
    padded_result = write_codec_artifact(
        padded, layers, alignment=64, target_file_bytes=result.file_bytes + 128
    )
    assert padded_result.file_bytes == result.file_bytes + 128
    assert padded_result.tail_padding_bytes == 128
    np.testing.assert_array_equal(read_codec_artifact(padded).layers["layer.a"], expected_a)


def test_quantized_lowrank_factor_roundtrip_and_metadata_oracle(tmp_path) -> None:
    q_codes = np.zeros((2, 3), dtype=np.int8)
    q_scales = np.ones(2, dtype=np.float16)
    left_codes = np.array([[3], [-2]], dtype=np.int8)
    left_scales = np.array([0.25, 0.5], dtype=np.float16)
    right_codes = np.array([[2, -3, 1]], dtype=np.int8)
    right_scales = np.array([0.125], dtype=np.float16)
    layer = LayerCodecPayload(
        "quantized.lowrank",
        q_codes,
        q_scales,
        4,
        lowrank_left=left_codes,
        lowrank_right=right_codes,
        lowrank_factor_bits=4,
        lowrank_left_scales=left_scales,
        lowrank_right_scales=right_scales,
    )
    path = tmp_path / "quantized_lowrank.hrc"
    result = write_codec_artifact(path, [layer], alignment=64)
    decoded = read_codec_artifact(path)

    left = left_codes.astype(np.float32) * left_scales.astype(np.float32)[:, None]
    right = right_codes.astype(np.float32) * right_scales.astype(np.float32)[:, None]
    expected = (left @ right).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(decoded.layers[layer.name], expected)
    allocation = LayerCodecAllocation(
        name=layer.name,
        shape=layer.shape,
        q_bits=layer.q_bits,
        q_scale_shape=tuple(layer.q_scales.shape),
        lowrank_rank=1,
        lowrank_factor_bits=4,
        lowrank_left_scale_shape=(2,),
        lowrank_right_scale_shape=(1,),
    )
    assert (
        codec_artifact_allocations_natural_file_bytes([allocation], alignment=64)
        == result.file_bytes
    )
    assert (
        codec_artifact_allocation_natural_file_bytes(
            name=layer.name,
            shape=layer.shape,
            q_bits=layer.q_bits,
            q_scale_shape=tuple(layer.q_scales.shape),
            q_col_block_size=None,
            sparse_nnz=0,
            lowrank_rank=1,
            lowrank_factor_bits=4,
            lowrank_left_scale_shape=(2,),
            lowrank_right_scale_shape=(1,),
            alignment=64,
        )
        == result.file_bytes
    )
    assert result.manifest["layers"][0]["lowrank_factor_bits"] == 4


def test_codec_artifact_detects_stream_corruption(tmp_path) -> None:
    layer = LayerCodecPayload(
        "layer",
        np.array([[1, -1]], dtype=np.int8),
        np.array([0.5], dtype=np.float16),
        4,
    )
    path = tmp_path / "corrupt.hrc"
    result = write_codec_artifact(path, [layer])
    stream = result.manifest["streams"][0]
    raw = bytearray(path.read_bytes())
    # Offsets are relative to the aligned payload base.  Locate the stream by
    # searching for its short unique bit-packed payload to avoid duplicating
    # container-prefix logic in the test.
    packed = pack_signed_codes(layer.q_codes, 4)
    location = bytes(raw).find(packed)
    assert location >= 0 and int(stream["nbytes"]) == len(packed)
    raw[location] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_codec_artifact(path)


def test_fp16_reference_artifact_roundtrip(tmp_path) -> None:
    weights = {
        "b": np.arange(12, dtype=np.float32).reshape(3, 4) / 7,
        "a": np.array([[1.0, -2.0]], dtype=np.float32),
    }
    path = tmp_path / "reference.hrc"
    result = write_fp16_reference_artifact(path, weights)
    decoded = read_fp16_reference_artifact(path)
    assert result.logical_payload_bits == sum(value.size for value in weights.values()) * 16
    for name, value in weights.items():
        np.testing.assert_array_equal(decoded.layers[name], value.astype(np.float16).astype(np.float32))


def test_explicit_empty_optional_components_fail_closed() -> None:
    codes = np.zeros((2, 3), dtype=np.int8)
    scales = np.ones(2, dtype=np.float16)
    with pytest.raises(ValueError, match="empty sparse components must be represented as None"):
        codec_artifact_natural_file_bytes(
            [
                LayerCodecPayload(
                    "empty.sparse",
                    codes,
                    scales,
                    4,
                    sparse_values=np.zeros((2, 3), dtype=np.float16),
                    sparse_mask=np.zeros((2, 3), dtype=bool),
                )
            ]
        )
    with pytest.raises(ValueError, match="rank-0 low-rank components must be represented as None"):
        codec_artifact_natural_file_bytes(
            [
                LayerCodecPayload(
                    "empty.lowrank",
                    codes,
                    scales,
                    4,
                    lowrank_left=np.zeros((2, 0), dtype=np.float16),
                    lowrank_right=np.zeros((0, 3), dtype=np.float16),
                )
            ]
        )
    with pytest.raises(ValueError, match="layer name must be non-empty"):
        codec_artifact_natural_file_bytes([LayerCodecPayload("", codes, scales, 4)])
    with pytest.raises(ValueError, match="layer dimensions must be positive"):
        codec_artifact_natural_file_bytes(
            [
                LayerCodecPayload(
                    "empty.rows",
                    np.zeros((0, 3), dtype=np.int8),
                    np.zeros((0,), dtype=np.float16),
                    4,
                )
            ]
        )
    with pytest.raises(ValueError, match="low-rank rank exceeds the matrix dimensions"):
        codec_artifact_natural_file_bytes(
            [
                LayerCodecPayload(
                    "rank.too.large",
                    np.zeros((1, 3), dtype=np.int8),
                    np.ones((1,), dtype=np.float16),
                    4,
                    lowrank_left=np.zeros((1, 2), dtype=np.float16),
                    lowrank_right=np.zeros((2, 3), dtype=np.float16),
                )
            ]
        )


def test_codec_artifact_is_deterministic_under_layer_reordering(tmp_path) -> None:
    layers = [
        LayerCodecPayload(
            name,
            np.array([[1, -1, 0]], dtype=np.int8),
            np.array([0.5], dtype=np.float16),
            4,
        )
        for name in ("layer.b", "layer.a")
    ]
    forward = write_codec_artifact(tmp_path / "forward.hrc", layers)
    reverse = write_codec_artifact(tmp_path / "reverse.hrc", list(reversed(layers)))
    assert forward.sha256 == reverse.sha256
    assert (tmp_path / "forward.hrc").read_bytes() == (tmp_path / "reverse.hrc").read_bytes()


def test_metadata_estimator_rejects_invalid_block_before_division() -> None:
    with pytest.raises(ValueError, match="q_col_block_size must be positive"):
        codec_artifact_allocation_natural_file_bytes(
            name="layer",
            shape=(2, 3),
            q_bits=4,
            q_scale_shape=(2, 1),
            q_col_block_size=0,
            sparse_nnz=0,
            lowrank_rank=0,
        )
