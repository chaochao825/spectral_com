import math

import numpy as np

from llm_spectral_dynamics.structured.oasr import (
    activation_clustered_permutation,
    block_circulant_param_count,
    block_circulant_project,
    compute_conditional_overlap,
    estimate_memory,
    monarch_like_two_block_project,
    norm_sorted_permutations,
    permuted_block_circulant_project,
    lowrank_project,
    rtn_quantize,
)


def circulant_from_coeff(coeff):
    coeff = np.asarray(coeff, dtype=np.float32)
    size = coeff.size
    return coeff[(np.arange(size)[None, :] - np.arange(size)[:, None]) % size]


def test_block_circulant_projection_shape_and_exact_circulant():
    block = circulant_from_coeff([1.0, 2.0, -1.0, 0.5] * 8)
    projected = block_circulant_project(block, block_size=32)
    assert projected.matrix.shape == block.shape
    assert projected.params == 32
    assert np.allclose(projected.matrix, block, atol=1e-6)


def test_block_circulant_projection_non_divisible_dimensions():
    matrix = np.arange(35 * 49, dtype=np.float32).reshape(35, 49)
    projected = block_circulant_project(matrix, block_size=16)
    assert projected.matrix.shape == matrix.shape
    assert projected.params == block_circulant_param_count(matrix.shape, 16)
    assert np.isfinite(projected.matrix).all()


def test_block_circulant_projection_partial_block_uses_observed_entries_only():
    matrix = np.eye(2, dtype=np.float32)
    projected = block_circulant_project(matrix, block_size=16)
    assert projected.matrix.shape == matrix.shape
    assert np.allclose(projected.matrix, matrix, atol=1e-6)


def test_permuted_block_circulant_projection_matches_manual_unpermute():
    matrix = np.random.default_rng(2).normal(size=(16, 16)).astype(np.float32)
    row_perm = np.array([1, 0] + list(range(2, 16)), dtype=np.int64)
    col_perm = np.array(list(range(15, -1, -1)), dtype=np.int64)
    projected = permuted_block_circulant_project(matrix, block_size=16, row_perm=row_perm, col_perm=col_perm)
    manual = block_circulant_project(matrix[np.ix_(row_perm, col_perm)], block_size=16).matrix
    restored = manual[np.ix_(np.argsort(row_perm), np.argsort(col_perm))]
    assert projected.matrix.shape == matrix.shape
    assert np.allclose(projected.matrix, restored, atol=1e-6)


def test_norm_and_activation_cluster_permutations_are_valid():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(9, 11)).astype(np.float32)
    x = rng.normal(size=(7, 11)).astype(np.float32)
    row_perm, col_perm = norm_sorted_permutations(matrix)
    act_perm = activation_clustered_permutation(x, block_size=4)
    assert np.array_equal(np.sort(row_perm), np.arange(matrix.shape[0]))
    assert np.array_equal(np.sort(col_perm), np.arange(matrix.shape[1]))
    assert np.array_equal(np.sort(act_perm), np.arange(matrix.shape[1]))


def test_monarch_like_two_block_param_count_and_shape():
    matrix = np.random.default_rng(4).normal(size=(35, 49)).astype(np.float32)
    projected = monarch_like_two_block_project(matrix, block_size=16)
    assert projected.matrix.shape == matrix.shape
    assert projected.params == 2 * block_circulant_param_count(matrix.shape, 16)
    assert np.isfinite(projected.matrix).all()


def test_memory_accounting_for_q_l_c_and_qcl():
    shape = (64, 128)
    q = estimate_memory(shape, q_bits=4)
    ql = estimate_memory(shape, q_bits=4, l_rank=4)
    qc = estimate_memory(shape, q_bits=4, c_params=block_circulant_param_count(shape, 32))
    qcl = estimate_memory(shape, q_bits=4, c_params=block_circulant_param_count(shape, 32), l_rank=4)
    assert math.isclose(q, 0.25)
    assert math.isclose(ql, 0.25 + 4 * (64 + 128) / (64 * 128))
    assert math.isclose(qc, 0.25 + block_circulant_param_count(shape, 32) / (64 * 128))
    assert math.isclose(qcl, qc + 4 * (64 + 128) / (64 * 128))


def test_conditional_overlap_known_cases():
    x = np.eye(2, dtype=np.float32)
    left = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    right = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    opposite = -right
    orthogonal = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    assert math.isclose(compute_conditional_overlap(x, left, right), 1.0)
    assert math.isclose(compute_conditional_overlap(x, left, opposite), -1.0)
    assert math.isclose(compute_conditional_overlap(x, left, orthogonal), 0.0)


def test_lowrank_rank_zero_residual():
    matrix = np.random.default_rng(0).normal(size=(8, 6)).astype(np.float32)
    projected = lowrank_project(matrix, rank=0)
    assert projected.rank == 0
    assert projected.params == 0
    assert np.allclose(projected.matrix, np.zeros_like(matrix))


def test_rtn_quantize_group_size_preserves_shape_and_finiteness():
    matrix = np.random.default_rng(1).normal(size=(7, 19)).astype(np.float32)
    quantized = rtn_quantize(matrix, bits=4, group_size=8)
    assert quantized.shape == matrix.shape
    assert np.isfinite(quantized).all()


if __name__ == "__main__":
    for _name, _func in sorted(globals().items()):
        if _name.startswith("test_"):
            _func()
            print(_name, "ok")
