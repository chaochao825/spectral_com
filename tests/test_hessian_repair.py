from __future__ import annotations

import math

import numpy as np
import pytest

from llm_spectral_dynamics.structured.hessian_repair import (
    PreparedInputCovariance,
    block_group_ids,
    exact_payload_accounting,
    hessian_basis_repair,
    hessian_constrained_basis_repair,
    hessian_group_scale_repair,
    hessian_row_block_scale_repair,
    input_hessian_cosine,
    input_hessian_inner,
    input_hessian_quadratic,
    obs_retained_support_correction,
    prepare_input_covariance,
    quadratic_comfort_path,
    repair_cancellation_gain,
    support_encoding_bits,
    validated_input_covariance,
)


def test_input_covariance_hessian_primitives() -> None:
    a = np.array([[1.0, 2.0], [0.0, -1.0]])
    b = np.array([[2.0, 0.0], [1.0, 3.0]])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    expected = float(np.trace(a @ cov @ b.T))
    assert input_hessian_inner(a, b, cov) == pytest.approx(expected)
    assert input_hessian_quadratic(a, cov) == pytest.approx(0.5 * np.trace(a @ cov @ a.T))
    assert input_hessian_cosine(a, a, cov) == pytest.approx(1.0)
    assert repair_cancellation_gain(a, -0.25 * a, cov) == pytest.approx(0.5)
    assert repair_cancellation_gain(np.zeros_like(a), b, cov) == pytest.approx(0.0)


def test_prepared_covariance_is_immutable_and_factory_only() -> None:
    raw = np.array([[2.0, 0.25], [0.25, 1.0]])
    prepared = prepare_input_covariance(raw, 2)
    snapshot = prepared.matrix.copy()

    raw[0, 0] = 99.0
    np.testing.assert_array_equal(prepared.matrix, snapshot)
    assert not prepared.matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        prepared.matrix[0, 0] = 3.0
    with pytest.raises(TypeError, match="validation factory"):
        PreparedInputCovariance(np.eye(2))


def test_prepared_covariance_reuses_a_single_full_psd_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = np.linalg.eigvalsh

    def counted_eigvalsh(matrix: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", counted_eigvalsh)
    covariance = np.array([[2.0, 0.25], [0.25, 1.0]])
    prepared = prepare_input_covariance(covariance, 2)
    assert calls == 1

    delta = np.array([[1.0, -0.5]])
    basis = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    groups = np.array([[0, 1]])
    input_hessian_inner(delta, delta, prepared)
    input_hessian_quadratic(delta, prepared)
    input_hessian_cosine(delta, delta, prepared)
    repair_cancellation_gain(delta, -delta, prepared)
    hessian_basis_repair(delta, basis, prepared)
    hessian_group_scale_repair(
        np.zeros_like(delta), delta, delta, groups, prepared, storage_dtype=None
    )
    hessian_row_block_scale_repair(
        np.zeros_like(delta), delta, prepared, col_block_size=1, storage_dtype=None
    )
    obs_retained_support_correction(
        delta, np.array([[True, False]]), prepared
    )
    assert calls == 1

    with pytest.raises(ValueError, match="applied again"):
        obs_retained_support_correction(
            delta, np.array([[True, False]]), prepared, damping=1e-6
        )

    # A raw covariance passed to a nested high-level repair is audited once,
    # then handed to its internal basis solve as the prepared object.
    hessian_group_scale_repair(
        np.zeros_like(delta), delta, delta, groups, covariance, storage_dtype=None
    )
    assert calls == 2


def test_tiny_numerical_negative_covariance_is_shifted_but_indefinite_is_rejected() -> None:
    delta = np.array([[0.0, 1.0]])
    for scale in (1e-12, 1.0, 1e12):
        nearly_psd = scale * np.diag([1.0, -1e-8])
        repaired = validated_input_covariance(nearly_psd, 2)
        assert np.linalg.eigvalsh(repaired).min() >= 0.0
        assert input_hessian_quadratic(delta, nearly_psd) >= 0.0
        obs_retained_support_correction(
            np.ones((1, 2)), np.array([[True, False]]), nearly_psd
        )

        with pytest.raises(ValueError, match="positive semidefinite"):
            input_hessian_quadratic(delta, scale * np.diag([1.0, -1e-3]))

    zero = validated_input_covariance(np.zeros((2, 2)), 2, psd_floor_rtol=1e-3)
    np.testing.assert_array_equal(zero, np.zeros((2, 2)))

    positive = np.diag([2.0, 3.0])
    repaired_positive = validated_input_covariance(positive, 2, psd_floor_rtol=0.1)
    np.testing.assert_array_equal(repaired_positive, positive)


def test_exact_payload_counts_row_scales_and_alignment() -> None:
    payload = exact_payload_accounting(
        (2, 4),
        base_code_bits=4,
        base_scale_count=2,
        base_scale_bits=16,
    )
    assert payload.reference_bits == 128
    assert payload.total_bits == 64
    assert payload.ratio == pytest.approx(0.5)
    aligned = exact_payload_accounting(
        (2, 4),
        base_code_bits=3,
        base_scale_count=1,
        base_scale_bits=5,
        alignment_bits=8,
    )
    assert aligned.total_bits == 32
    assert sum(item.padding_bits for item in aligned.items) == 3


def test_support_encodings_distinguish_lower_bound_and_realizable_csr() -> None:
    mask = np.array([[True, True, False, False], [True, False, True, False]])
    bits, selected = support_encoding_bits((2, 4), mask=mask, encoding="auto")
    assert bits == 8
    assert selected in {"bitmap", "fixed_row"}
    entropy, entropy_name = support_encoding_bits((2, 4), mask=mask, encoding="entropy")
    assert entropy == 7
    assert entropy_name == "entropy"
    csr, csr_name = support_encoding_bits(
        (2, 4), mask=mask, encoding="csr_fixed", index_bits=8, row_pointer_bits=32
    )
    assert csr == 4 * 8 + 3 * 32
    assert csr_name == "csr_fixed"
    assert support_encoding_bits((2, 4), nonzero=0, encoding="auto") == (0, "empty")
    assert support_encoding_bits((2, 4), nonzero=8, encoding="auto") == (0, "dense")


def test_payload_counts_sparse_lowrank_and_folded_repair() -> None:
    payload = exact_payload_accounting(
        (4, 8),
        base_code_bits=4,
        base_scale_count=4,
        sparse_nonzero=3,
        support_encoding="csr_fixed",
        sparse_index_bits=8,
        lowrank_rank=2,
        repair_param_count=5,
        repair_folded=True,
    )
    fields = payload.as_dict()
    assert fields["base_codes_raw_bits"] == 32 * 4
    assert fields["base_scales_raw_bits"] == 4 * 16
    assert fields["sparse_values_raw_bits"] == 3 * 16
    assert fields["sparse_support_raw_bits"] == 3 * 8 + 5 * 32
    assert fields["lowrank_factors_raw_bits"] == 2 * (4 + 8) * 16
    assert fields["repair_raw_bits"] == 0
    explicit = exact_payload_accounting((4, 8), repair_param_count=5, repair_param_bits=16)
    assert explicit.as_dict()["repair_raw_bits"] == 80


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (0, 4)},
        {"shape": (2, 4), "base_code_bits": -1},
        {"shape": (2, 4), "lowrank_rank": 3},
        {"shape": (2, 4), "sparse_nonzero": 9},
        {"shape": (2, 4), "alignment_bits": 0},
    ],
)
def test_payload_rejects_invalid_specs(kwargs: dict[str, object]) -> None:
    shape = kwargs.pop("shape")
    with pytest.raises(ValueError):
        exact_payload_accounting(shape, **kwargs)  # type: ignore[arg-type]


def test_obs_schur_correction_is_retained_support_orthogonal() -> None:
    weight = np.array([[1.0, 2.0]])
    mask = np.array([[False, True]])
    cov = np.array([[2.0, 1.0], [1.0, 3.0]])
    result = obs_retained_support_correction(weight, mask, cov)
    np.testing.assert_allclose(result.delta, np.array([[-1.0, 1.0 / 3.0]]), atol=1e-12)
    assert result.corrected_cost == pytest.approx(5.0 / 6.0)
    assert result.schur_cost == pytest.approx(result.corrected_cost)
    assert result.corrected_cost < result.naive_cost
    assert result.max_retained_stationarity < 1e-12
    retained_only = np.array([[0.0, 7.0]])
    assert input_hessian_inner(result.delta, retained_only, cov) == pytest.approx(0.0, abs=1e-12)


def test_obs_handles_singular_and_degenerate_supports() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = obs_retained_support_correction(
        np.array([[1.0, 2.0]]), np.array([[False, True]]), singular
    )
    np.testing.assert_allclose(result.delta, np.array([[-1.0, 1.0]]), atol=1e-10)
    assert result.corrected_cost == pytest.approx(0.0, abs=1e-10)
    all_kept = obs_retained_support_correction(
        np.array([[1.0, 2.0]]), np.array([[True, True]]), np.eye(2)
    )
    np.testing.assert_allclose(all_kept.corrected_weight, np.array([[1.0, 2.0]]))
    none_kept = obs_retained_support_correction(
        np.array([[1.0, 2.0]]), np.array([[False, False]]), np.eye(2)
    )
    np.testing.assert_allclose(none_kept.corrected_weight, np.zeros((1, 2)))


def test_obs_rejects_indefinite_hessian_and_support_explosion() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        obs_retained_support_correction(
            np.ones((1, 2)), np.array([[True, False]]), np.diag([1.0, -1.0])
        )
    with pytest.raises(ValueError, match="unique support"):
        obs_retained_support_correction(
            np.ones((2, 2)), np.array([[True, False], [False, True]]), np.eye(2), max_unique_supports=1
        )


def test_basis_and_group_scale_repair_are_monotone() -> None:
    target = np.array([[2.0, 4.0]])
    decoded = np.array([[1.0, 2.0]])
    basis = decoded[None, :, :]
    direct = hessian_basis_repair(decoded - target, basis, np.eye(2))
    assert direct.coefficients[0] == pytest.approx(1.0)
    assert direct.cost_after == pytest.approx(0.0)
    groups = np.zeros_like(decoded, dtype=np.int64)
    grouped = hessian_group_scale_repair(target, decoded, decoded, groups, np.eye(2))
    np.testing.assert_allclose(grouped.repaired, target)
    assert grouped.scales[0] == pytest.approx(2.0)
    assert grouped.stored_scale_count == 1


def test_row_block_scale_repair_uses_deployable_group_count() -> None:
    target = np.array([[2.0, 4.0, 1.5, 3.0], [1.0, 2.0, 3.0, 6.0]])
    decoded = np.array([[1.0, 2.0, 1.0, 2.0], [0.5, 1.0, 2.0, 4.0]])
    result = hessian_row_block_scale_repair(
        target,
        decoded,
        np.eye(4),
        col_block_size=2,
        storage_dtype=np.float16,
    )
    assert result.stored_scale_count == 4
    assert result.scales.shape == (2, 2)
    assert result.cost_after <= result.cost_before
    np.testing.assert_allclose(result.repaired, target, atol=1e-3)


def test_constrained_scale_repair_reports_rank_and_feasibility() -> None:
    delta = np.array([[1.0, 1.0]])
    basis = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    constraints = np.array([[[1.0, -1.0]]])
    result = hessian_constrained_basis_repair(delta, basis, constraints, np.eye(2))
    assert result.constraint_rank == 1
    assert result.feasible
    assert result.repair.max_basis_stationarity < 1e-10
    assert input_hessian_inner(constraints[0], result.repair.repaired_delta, np.eye(2)) == pytest.approx(
        0.0, abs=1e-10
    )
    one_basis = basis[:1]
    two_constraints = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    impossible = hessian_constrained_basis_repair(delta, one_basis, two_constraints, np.eye(2))
    assert not impossible.feasible

    # Feasibility must be scale-relative: the former unit denominator would
    # silently accept this non-zero but small unsatisfied constraint.
    tiny_delta = np.array([[1e-10, 0.0]])
    zero_basis = np.zeros((1, 1, 2))
    tiny_impossible = hessian_constrained_basis_repair(
        tiny_delta,
        zero_basis,
        np.array([[[1.0, 0.0]]]),
        np.eye(2),
    )
    assert tiny_impossible.relative_constraint_residual == pytest.approx(1.0)
    assert not tiny_impossible.feasible


def test_block_ids_and_quadratic_comfort_path() -> None:
    groups = block_group_ids((3, 5), row_block_size=2, col_block_size=3)
    assert groups.shape == (3, 5)
    assert len(np.unique(groups)) == 4
    path = quadratic_comfort_path(np.array([[1.0, 2.0]]), np.eye(2), [0.0, 0.5, 1.0])
    assert path[1]["hessian_cost"] == pytest.approx(0.25 * path[2]["hessian_cost"])
    assert path[0]["path_kind"] == "noncodec_interpolation"
    assert path[2]["path_kind"] == "codec_endpoint"
