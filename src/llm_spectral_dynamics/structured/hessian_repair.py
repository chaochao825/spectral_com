from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12
_PREPARED_COVARIANCE_TOKEN = object()


class PreparedInputCovariance:
    """An immutable covariance whose full PSD audit has already succeeded.

    Instances are created only by this module or by the numerical runner after
    it performs the equivalent one-spectrum audit.  Repair kernels can reuse
    the object without silently repeating an ``O(d^3)`` eigendecomposition.
    """

    __slots__ = ("_matrix",)

    def __init__(self, matrix: np.ndarray, *, _token: object | None = None) -> None:
        if _token is not _PREPARED_COVARIANCE_TOKEN:
            raise TypeError("use a covariance validation factory")
        self._matrix = matrix

    @classmethod
    def _from_validated_array(cls, covariance: np.ndarray) -> "PreparedInputCovariance":
        matrix = np.array(covariance, dtype=np.float64, order="C", copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("prepared covariance must be square")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("prepared covariance contains non-finite values")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("prepared covariance must be exactly symmetric")
        matrix.setflags(write=False)
        return cls(matrix, _token=_PREPARED_COVARIANCE_TOKEN)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


def _as_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def validated_input_covariance(
    covariance: np.ndarray,
    input_dim: int,
    *,
    damping: float = 0.0,
    psd_rtol: float = 1e-7,
    psd_floor_rtol: float = 0.0,
) -> np.ndarray:
    """Return a symmetric PSD covariance after bounded numerical repair.

    ``psd_rtol`` is a scale-relative rejection threshold: matrices whose
    negative eigenvalue exceeds it are materially indefinite and fail closed.
    ``psd_floor_rtol`` optionally adds a declared relative spectral floor, for
    example before storing the matrix in float32.  The zero matrix remains
    zero; no absolute unit-scale tolerance is introduced.
    """

    cov = _as_matrix(covariance, name="covariance")
    if cov.shape != (input_dim, input_dim):
        raise ValueError(f"covariance shape mismatch: expected {(input_dim, input_dim)}, got {cov.shape}")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if psd_rtol < 0.0:
        raise ValueError("psd_rtol must be non-negative")
    if psd_floor_rtol < 0.0:
        raise ValueError("psd_floor_rtol must be non-negative")
    cov = 0.5 * (cov + cov.T)
    if damping:
        cov = cov + float(damping) * np.eye(input_dim, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(cov)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if minimum < -float(psd_rtol) * scale:
        raise ValueError("covariance must be positive semidefinite")
    target_floor = float(psd_floor_rtol) * scale
    if minimum < target_floor:
        # Empirical activation Grams are accumulated/stored with finite
        # precision.  Accept only a tiny relative negative eigenvalue and
        # remove it with a declared diagonal shift before using the matrix as
        # a PSD geometry.  Materially indefinite inputs still fail above.  A
        # strictly zero matrix has scale=target_floor=minimum=0 and is kept.
        numerical_margin = np.finfo(np.float64).eps * scale * 16.0
        cov = cov + (target_floor - minimum + numerical_margin) * np.eye(
            input_dim, dtype=np.float64
        )
    return cov


def prepare_input_covariance(
    covariance: np.ndarray,
    input_dim: int,
    *,
    damping: float = 0.0,
    psd_rtol: float = 1e-7,
    psd_floor_rtol: float = 0.0,
) -> PreparedInputCovariance:
    """Validate once and bind an immutable covariance for repeated repairs."""

    validated = validated_input_covariance(
        covariance,
        input_dim,
        damping=damping,
        psd_rtol=psd_rtol,
        psd_floor_rtol=psd_floor_rtol,
    )
    return PreparedInputCovariance._from_validated_array(validated)


def _prepare_covariance_argument(
    covariance: np.ndarray | PreparedInputCovariance,
    input_dim: int,
    *,
    damping: float = 0.0,
    psd_rtol: float = 1e-7,
) -> PreparedInputCovariance:
    if isinstance(covariance, PreparedInputCovariance):
        if damping:
            raise ValueError("damping cannot be applied again to a prepared covariance")
        if covariance.matrix.shape != (input_dim, input_dim):
            raise ValueError(
                f"covariance shape mismatch: expected {(input_dim, input_dim)}, "
                f"got {covariance.matrix.shape}"
            )
        return covariance
    return prepare_input_covariance(
        covariance,
        input_dim,
        damping=damping,
        psd_rtol=psd_rtol,
    )


def _input_hessian_inner_prepared(
    delta_a: np.ndarray, delta_b: np.ndarray, covariance: np.ndarray
) -> float:
    return float(np.sum((delta_a @ covariance) * delta_b))


def _input_hessian_quadratic_prepared(delta: np.ndarray, covariance: np.ndarray) -> float:
    return 0.5 * _input_hessian_inner_prepared(delta, delta, covariance)


def input_hessian_inner(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
) -> float:
    """Return ``tr(delta_a @ covariance @ delta_b.T)``.

    This is the input-covariance Hessian proxy for a linear layer.  It is an
    activation-MSE / ``C \u2297 I_out`` metric, not a full task Hessian with output
    coupling.
    """

    a = _as_matrix(delta_a, name="delta_a")
    b = _as_matrix(delta_b, name="delta_b")
    if a.shape != b.shape:
        raise ValueError(f"delta shape mismatch: {a.shape} != {b.shape}")
    cov = _prepare_covariance_argument(covariance, a.shape[1]).matrix
    return _input_hessian_inner_prepared(a, b, cov)


def input_hessian_quadratic(
    delta: np.ndarray, covariance: np.ndarray | PreparedInputCovariance
) -> float:
    error = _as_matrix(delta, name="delta")
    cov = _prepare_covariance_argument(covariance, error.shape[1]).matrix
    return _input_hessian_quadratic_prepared(error, cov)


def input_hessian_cosine(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
) -> float:
    a = _as_matrix(delta_a, name="delta_a")
    b = _as_matrix(delta_b, name="delta_b")
    if a.shape != b.shape:
        raise ValueError(f"delta shape mismatch: {a.shape} != {b.shape}")
    cov = _prepare_covariance_argument(covariance, a.shape[1]).matrix
    numerator = _input_hessian_inner_prepared(a, b, cov)
    norm_a = math.sqrt(max(_input_hessian_inner_prepared(a, a, cov), 0.0))
    norm_b = math.sqrt(max(_input_hessian_inner_prepared(b, b, cov), 0.0))
    denominator = norm_a * norm_b
    if denominator <= EPS:
        return 0.0
    value = float(numerator / denominator)
    if not math.isfinite(value):
        return value
    return max(-1.0, min(1.0, value))


def repair_cancellation_gain(
    compression_delta: np.ndarray,
    repair_delta: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
) -> float:
    """Return the compression self-loss fraction removed by the cross term.

    With ``L(c + r) = 1/2 <c,c>_H + 1/2 <r,r>_H + <c,r>_H``, this
    diagnostic is ``-2 <c,r>_H / <c,c>_H``.  A positive value therefore
    measures cancellation of the compression perturbation by the repair; it
    is deliberately distinct from ``|rho_H| ~= 0`` orthogonality and does not
    include the repair's own self-loss.
    """

    compression = _as_matrix(compression_delta, name="compression_delta")
    repair = _as_matrix(repair_delta, name="repair_delta")
    if compression.shape != repair.shape:
        raise ValueError(f"delta shape mismatch: {compression.shape} != {repair.shape}")
    cov = _prepare_covariance_argument(covariance, compression.shape[1]).matrix
    denominator = _input_hessian_inner_prepared(compression, compression, cov)
    if denominator <= EPS:
        return 0.0
    return float(-2.0 * _input_hessian_inner_prepared(compression, repair, cov) / denominator)


def _ceil_log2(value: int) -> int:
    if value < 0:
        raise ValueError("value must be non-negative")
    return 0 if value <= 1 else int(math.ceil(math.log2(value)))


def _align_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError("bit count must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment_bits must be positive")
    return int(((value + alignment - 1) // alignment) * alignment)


def _entropy_support_bits(total: int, nonzero: int) -> int:
    if nonzero < 0 or nonzero > total:
        raise ValueError("nonzero count must be within the tensor size")
    if nonzero in {0, total}:
        return 0
    # lgamma avoids constructing a potentially enormous binomial integer.
    log2_combinations = (
        math.lgamma(total + 1) - math.lgamma(nonzero + 1) - math.lgamma(total - nonzero + 1)
    ) / math.log(2.0)
    return int(math.ceil(max(log2_combinations, 0.0) - 1e-12))


def _storage_index_bits(cardinality: int) -> int:
    if cardinality <= 0:
        raise ValueError("cardinality must be positive")
    if cardinality <= 2**8:
        return 8
    if cardinality <= 2**16:
        return 16
    return 32


def support_encoding_bits(
    shape: tuple[int, int],
    *,
    nonzero: int | None = None,
    mask: np.ndarray | None = None,
    encoding: str = "auto",
    index_bits: int | None = None,
    row_pointer_bits: int = 32,
) -> tuple[int, str]:
    """Return support bits and the selected encoding.

    ``entropy`` is an information-theoretic lower bound. ``csr_fixed`` is the
    realizable codec used by the exact-rate experiments. ``auto`` chooses the
    smallest of bitmap, COO, CSR and fixed-row encodings without using the
    entropy lower bound.
    """

    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")
    total = rows * cols
    row_counts: np.ndarray | None = None
    if mask is not None:
        bool_mask = np.asarray(mask, dtype=bool)
        if bool_mask.shape != (rows, cols):
            raise ValueError(f"mask shape mismatch: expected {(rows, cols)}, got {bool_mask.shape}")
        inferred = int(np.count_nonzero(bool_mask))
        if nonzero is not None and int(nonzero) != inferred:
            raise ValueError("nonzero does not match mask")
        nonzero = inferred
        row_counts = np.count_nonzero(bool_mask, axis=1)
    if nonzero is None:
        raise ValueError("provide either nonzero or mask")
    nonzero = int(nonzero)
    if nonzero < 0 or nonzero > total:
        raise ValueError("nonzero count must be within the tensor size")
    if nonzero == 0:
        return 0, "empty"
    if nonzero == total:
        return 0, "dense"

    col_ideal = _ceil_log2(cols)
    row_ideal = _ceil_log2(rows)
    pointer_ideal = _ceil_log2(nonzero + 1)
    choices: dict[str, int] = {
        "bitmap": total,
        "coo": nonzero * (row_ideal + col_ideal),
        "csr": nonzero * col_ideal + (rows + 1) * pointer_ideal,
    }
    if row_counts is not None and np.all(row_counts == row_counts[0]):
        choices["fixed_row"] = nonzero * col_ideal

    if encoding == "entropy":
        return _entropy_support_bits(total, nonzero), "entropy"
    if encoding == "csr_fixed":
        col_bits = _storage_index_bits(cols) if index_bits is None else int(index_bits)
        if col_bits <= 0 or row_pointer_bits <= 0:
            raise ValueError("fixed CSR index widths must be positive")
        return nonzero * col_bits + (rows + 1) * int(row_pointer_bits), "csr_fixed"
    if encoding == "auto":
        selected = min(choices, key=lambda key: (choices[key], key))
        return int(choices[selected]), selected
    if encoding not in choices:
        raise ValueError(f"unsupported support encoding: {encoding}")
    return int(choices[encoding]), encoding


@dataclass(frozen=True)
class PayloadItem:
    name: str
    raw_bits: int
    padding_bits: int
    stored_bits: int


@dataclass(frozen=True)
class PayloadBreakdown:
    shape: tuple[int, int]
    reference_bits: int
    items: tuple[PayloadItem, ...]
    support_encoding: str

    @property
    def total_bits(self) -> int:
        return int(sum(item.stored_bits for item in self.items))

    @property
    def ratio(self) -> float:
        return float(self.total_bits) / float(self.reference_bits)

    @property
    def compression_ratio(self) -> float:
        return float(self.reference_bits) / float(max(self.total_bits, 1))

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rows": self.shape[0],
            "cols": self.shape[1],
            "reference_bits": self.reference_bits,
            "payload_bits": self.total_bits,
            "payload_ratio": self.ratio,
            "compression_ratio": self.compression_ratio,
            "support_encoding": self.support_encoding,
        }
        for item in self.items:
            payload[f"{item.name}_raw_bits"] = item.raw_bits
            payload[f"{item.name}_padding_bits"] = item.padding_bits
            payload[f"{item.name}_stored_bits"] = item.stored_bits
        return payload


def exact_payload_accounting(
    shape: tuple[int, int],
    *,
    reference_bits_per_weight: int = 16,
    base_code_bits: int = 0,
    base_code_count: int | None = None,
    base_scale_count: int = 0,
    base_scale_bits: int = 16,
    sparse_mask: np.ndarray | None = None,
    sparse_nonzero: int | None = None,
    sparse_value_bits: int = 16,
    support_encoding: str = "auto",
    sparse_index_bits: int | None = None,
    sparse_row_pointer_bits: int = 32,
    sparse_scale_count: int = 0,
    sparse_scale_bits: int = 16,
    lowrank_rank: int = 0,
    lowrank_factor_bits: int = 16,
    lowrank_scale_count: int = 0,
    lowrank_scale_bits: int = 16,
    repair_param_count: int = 0,
    repair_param_bits: int = 16,
    repair_folded: bool = False,
    metadata_bits: int = 0,
    alignment_bits: int = 1,
) -> PayloadBreakdown:
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")
    counts = {
        "reference_bits_per_weight": reference_bits_per_weight,
        "base_code_bits": base_code_bits,
        "base_scale_count": base_scale_count,
        "base_scale_bits": base_scale_bits,
        "sparse_value_bits": sparse_value_bits,
        "sparse_scale_count": sparse_scale_count,
        "sparse_scale_bits": sparse_scale_bits,
        "lowrank_rank": lowrank_rank,
        "lowrank_factor_bits": lowrank_factor_bits,
        "lowrank_scale_count": lowrank_scale_count,
        "lowrank_scale_bits": lowrank_scale_bits,
        "repair_param_count": repair_param_count,
        "repair_param_bits": repair_param_bits,
        "metadata_bits": metadata_bits,
    }
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("bit widths and counts must be non-negative")
    if reference_bits_per_weight <= 0:
        raise ValueError("reference_bits_per_weight must be positive")
    if lowrank_rank > min(rows, cols):
        raise ValueError("lowrank_rank exceeds the matrix rank bound")
    total = rows * cols
    code_count = total if base_code_count is None else int(base_code_count)
    if code_count < 0:
        raise ValueError("base_code_count must be non-negative")
    if sparse_mask is not None:
        mask = np.asarray(sparse_mask, dtype=bool)
        if mask.shape != (rows, cols):
            raise ValueError(f"sparse_mask shape mismatch: expected {(rows, cols)}, got {mask.shape}")
        inferred_nonzero = int(np.count_nonzero(mask))
        if sparse_nonzero is not None and int(sparse_nonzero) != inferred_nonzero:
            raise ValueError("sparse_nonzero does not match sparse_mask")
        sparse_nonzero = inferred_nonzero
    if sparse_nonzero is None:
        sparse_nonzero = 0
    sparse_nonzero = int(sparse_nonzero)
    if sparse_nonzero < 0 or sparse_nonzero > total:
        raise ValueError("sparse_nonzero must be within the matrix size")
    if sparse_nonzero:
        support_bits, selected_support = support_encoding_bits(
            (rows, cols),
            nonzero=sparse_nonzero,
            mask=sparse_mask,
            encoding=support_encoding,
            index_bits=sparse_index_bits,
            row_pointer_bits=sparse_row_pointer_bits,
        )
    else:
        support_bits, selected_support = 0, "empty"

    raw_items = (
        ("base_codes", code_count * int(base_code_bits)),
        ("base_scales", int(base_scale_count) * int(base_scale_bits)),
        ("sparse_values", sparse_nonzero * int(sparse_value_bits)),
        ("sparse_support", support_bits),
        ("sparse_scales", int(sparse_scale_count) * int(sparse_scale_bits)),
        ("lowrank_factors", int(lowrank_rank) * (rows + cols) * int(lowrank_factor_bits)),
        ("lowrank_scales", int(lowrank_scale_count) * int(lowrank_scale_bits)),
        (
            "repair",
            0 if repair_folded else int(repair_param_count) * int(repair_param_bits),
        ),
        ("metadata", int(metadata_bits)),
    )
    items: list[PayloadItem] = []
    for name, raw in raw_items:
        stored = _align_up(int(raw), int(alignment_bits))
        items.append(PayloadItem(name=name, raw_bits=int(raw), padding_bits=stored - int(raw), stored_bits=stored))
    return PayloadBreakdown(
        shape=(rows, cols),
        reference_bits=total * int(reference_bits_per_weight),
        items=tuple(items),
        support_encoding=selected_support,
    )


@dataclass(frozen=True)
class BasisRepairResult:
    repaired_delta: np.ndarray
    coefficients: np.ndarray
    cost_before: float
    cost_after: float
    gram_rank: int
    gram_condition: float
    max_basis_stationarity: float
    active_basis_count: int


def _basis_gram(
    delta: np.ndarray,
    basis: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    components = np.asarray(basis, dtype=np.float64)
    if components.ndim != 3 or components.shape[1:] != delta.shape:
        raise ValueError(f"basis must have shape (k, {delta.shape[0]}, {delta.shape[1]})")
    if not np.all(np.isfinite(components)):
        raise ValueError("basis contains non-finite values")
    count = components.shape[0]
    gram = np.empty((count, count), dtype=np.float64)
    gradient = np.empty(count, dtype=np.float64)
    for left in range(count):
        gradient[left] = _input_hessian_inner_prepared(components[left], delta, covariance)
        for right in range(left, count):
            value = _input_hessian_inner_prepared(
                components[left], components[right], covariance
            )
            gram[left, right] = value
            gram[right, left] = value
    return gram, gradient


def _symmetric_pinv(matrix: np.ndarray, *, rcond: float) -> tuple[np.ndarray, int, float]:
    if rcond < 0.0:
        raise ValueError("rcond must be non-negative")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    maximum = max(float(np.max(np.abs(values), initial=0.0)), EPS)
    threshold = float(rcond) * maximum
    active = values > threshold
    rank = int(np.count_nonzero(active))
    if rank:
        inverse = (vectors[:, active] / values[active]) @ vectors[:, active].T
        condition = float(np.max(values[active]) / max(float(np.min(values[active])), EPS))
    else:
        inverse = np.zeros_like(matrix)
        condition = float("inf")
    return inverse, rank, condition


def hessian_basis_repair(
    delta: np.ndarray,
    basis: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
    *,
    rcond: float = 1e-10,
    ridge: float = 0.0,
) -> BasisRepairResult:
    error = _as_matrix(delta, name="delta")
    cov = _prepare_covariance_argument(covariance, error.shape[1]).matrix
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    components = np.asarray(basis, dtype=np.float64)
    if components.ndim != 3 or components.shape[1:] != error.shape:
        raise ValueError(f"basis must have shape (k, {error.shape[0]}, {error.shape[1]})")
    if components.shape[0] == 0:
        return BasisRepairResult(
            repaired_delta=error.copy(),
            coefficients=np.zeros(0, dtype=np.float64),
            cost_before=_input_hessian_quadratic_prepared(error, cov),
            cost_after=_input_hessian_quadratic_prepared(error, cov),
            gram_rank=0,
            gram_condition=float("inf"),
            max_basis_stationarity=0.0,
            active_basis_count=0,
        )
    gram, gradient = _basis_gram(error, components, cov)
    solve_gram = gram + float(ridge) * np.eye(gram.shape[0], dtype=np.float64)
    inverse, rank, condition = _symmetric_pinv(solve_gram, rcond=rcond)
    coefficients = -(inverse @ gradient)
    repaired = error + np.tensordot(coefficients, components, axes=(0, 0))
    stationarity = gram @ coefficients + gradient
    scale = max(float(np.linalg.norm(gradient)), float(np.linalg.norm(gram) * np.linalg.norm(coefficients)), EPS)
    return BasisRepairResult(
        repaired_delta=repaired,
        coefficients=coefficients,
        cost_before=_input_hessian_quadratic_prepared(error, cov),
        cost_after=_input_hessian_quadratic_prepared(repaired, cov),
        gram_rank=rank,
        gram_condition=condition,
        max_basis_stationarity=float(np.max(np.abs(stationarity), initial=0.0) / scale),
        active_basis_count=int(np.count_nonzero(np.linalg.norm(components.reshape(components.shape[0], -1), axis=1) > EPS)),
    )


@dataclass(frozen=True)
class ConstrainedBasisRepairResult:
    repair: BasisRepairResult
    constraint_rank: int
    relative_constraint_residual: float
    feasible: bool


def hessian_constrained_basis_repair(
    delta: np.ndarray,
    basis: np.ndarray,
    constraints: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
    *,
    rcond: float = 1e-10,
    constraint_rtol: float = 1e-8,
) -> ConstrainedBasisRepairResult:
    """Minimize Hessian cost while nulling selected Hessian cross terms.

    Feasibility requires the existing cross-term vector to lie in the column
    space of ``A[j, g] = <constraint_j, basis_g>_H``.  This makes explicit why
    one scalar cannot generally orthogonalize several independent directions.
    """

    error = _as_matrix(delta, name="delta")
    components = np.asarray(basis, dtype=np.float64)
    directions = np.asarray(constraints, dtype=np.float64)
    if components.ndim != 3 or components.shape[1:] != error.shape:
        raise ValueError("basis shape mismatch")
    if directions.ndim != 3 or directions.shape[1:] != error.shape:
        raise ValueError("constraints shape mismatch")
    cov = _prepare_covariance_argument(covariance, error.shape[1]).matrix
    gram, gradient = _basis_gram(error, components, cov)
    matrix = np.empty((directions.shape[0], components.shape[0]), dtype=np.float64)
    target = np.empty(directions.shape[0], dtype=np.float64)
    for row, direction in enumerate(directions):
        target[row] = -_input_hessian_inner_prepared(direction, error, cov)
        for col, component in enumerate(components):
            matrix[row, col] = _input_hessian_inner_prepared(direction, component, cov)
    if matrix.size == 0:
        coefficients = np.zeros(components.shape[0], dtype=np.float64)
        rank = 0
        null_basis = np.eye(components.shape[0], dtype=np.float64)
    else:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
        threshold = float(rcond) * max(float(np.max(singular, initial=0.0)), EPS)
        active = singular > threshold
        rank = int(np.count_nonzero(active))
        coefficients = np.linalg.pinv(matrix, rcond=rcond) @ target
        null_basis = vh[rank:].T
        if null_basis.size:
            reduced_gram = null_basis.T @ gram @ null_basis
            reduced_gradient = null_basis.T @ (gradient + gram @ coefficients)
            reduced_inverse, _, _ = _symmetric_pinv(reduced_gram, rcond=rcond)
            coefficients = coefficients - null_basis @ (reduced_inverse @ reduced_gradient)
    residual = matrix @ coefficients - target
    constraint_scale = max(
        float(np.linalg.norm(target)),
        float(np.linalg.norm(matrix) * np.linalg.norm(coefficients)),
        float(np.finfo(np.float64).tiny),
    )
    relative_residual = float(np.linalg.norm(residual) / constraint_scale)
    repaired_delta = error + np.tensordot(coefficients, components, axes=(0, 0))
    stationarity = gram @ coefficients + gradient
    projected_stationarity = null_basis.T @ stationarity
    stationarity_scale = max(
        float(np.linalg.norm(null_basis.T @ gradient)),
        float(np.linalg.norm(null_basis.T @ gram) * np.linalg.norm(coefficients)),
        EPS,
    )
    repair = BasisRepairResult(
        repaired_delta=repaired_delta,
        coefficients=coefficients,
        cost_before=_input_hessian_quadratic_prepared(error, cov),
        cost_after=_input_hessian_quadratic_prepared(repaired_delta, cov),
        gram_rank=int(np.linalg.matrix_rank(gram, tol=max(rcond * np.linalg.norm(gram), EPS))),
        gram_condition=float(np.linalg.cond(gram)) if gram.size else float("inf"),
        # A constrained optimum is stationary only along feasible tangent
        # directions (the null space of the constraint matrix).  Reporting
        # the raw gradient here would incorrectly include the KKT multiplier.
        max_basis_stationarity=float(
            np.max(np.abs(projected_stationarity), initial=0.0) / stationarity_scale
        ),
        active_basis_count=components.shape[0],
    )
    return ConstrainedBasisRepairResult(
        repair=repair,
        constraint_rank=rank,
        relative_constraint_residual=relative_residual,
        feasible=relative_residual <= float(constraint_rtol),
    )


def block_group_ids(
    shape: tuple[int, int],
    *,
    row_block_size: int,
    col_block_size: int,
) -> np.ndarray:
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0 or row_block_size <= 0 or col_block_size <= 0:
        raise ValueError("shape and block sizes must be positive")
    row_ids = np.arange(rows, dtype=np.int64) // int(row_block_size)
    col_ids = np.arange(cols, dtype=np.int64) // int(col_block_size)
    col_groups = int(math.ceil(cols / float(col_block_size)))
    return row_ids[:, None] * col_groups + col_ids[None, :]


@dataclass(frozen=True)
class GroupScaleRepairResult:
    repaired: np.ndarray
    scales: np.ndarray
    basis_repair: BasisRepairResult
    stored_scale_count: int
    quantized_storage: bool


def hessian_group_scale_repair(
    target: np.ndarray,
    decoded: np.ndarray,
    scale_component: np.ndarray,
    group_ids: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
    *,
    scale_bounds: tuple[float, float] | None = (0.0, 2.0),
    storage_dtype: np.dtype | type | None = np.float16,
    rcond: float = 1e-10,
    ridge: float = 0.0,
) -> GroupScaleRepairResult:
    wanted = _as_matrix(target, name="target")
    current = _as_matrix(decoded, name="decoded")
    component = _as_matrix(scale_component, name="scale_component")
    groups = np.asarray(group_ids)
    if wanted.shape != current.shape or wanted.shape != component.shape or groups.shape != wanted.shape:
        raise ValueError("target, decoded, scale_component and group_ids must share a shape")
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("group_ids must be integer-valued")
    unique = np.unique(groups)
    if unique.size and int(unique[0]) < 0:
        raise ValueError("group_ids must be non-negative")
    prepared_covariance = _prepare_covariance_argument(covariance, wanted.shape[1])
    cov = prepared_covariance.matrix
    basis = np.stack([np.where(groups == group, component, 0.0) for group in unique], axis=0)
    result = hessian_basis_repair(
        current - wanted, basis, prepared_covariance, rcond=rcond, ridge=ridge
    )
    scales = 1.0 + result.coefficients
    if scale_bounds is not None:
        lower, upper = map(float, scale_bounds)
        if not lower <= upper:
            raise ValueError("invalid scale bounds")
        scales = np.clip(scales, lower, upper)
    if storage_dtype is not None:
        scales = scales.astype(storage_dtype).astype(np.float64)
    repaired = current.copy()
    for index, group in enumerate(unique):
        repaired = repaired + (scales[index] - 1.0) * np.where(groups == group, component, 0.0)
    if _input_hessian_quadratic_prepared(
        repaired - wanted, cov
    ) > _input_hessian_quadratic_prepared(current - wanted, cov) + 1e-10:
        repaired = current.copy()
        scales = np.ones_like(scales)
    return GroupScaleRepairResult(
        repaired=repaired,
        scales=scales,
        basis_repair=result,
        stored_scale_count=int(unique.size),
        quantized_storage=storage_dtype is not None,
    )


@dataclass(frozen=True)
class RowBlockScaleRepairResult:
    repaired: np.ndarray
    scales: np.ndarray
    cost_before: float
    cost_after: float
    stored_scale_count: int
    max_relative_stationarity: float


def hessian_row_block_scale_repair(
    target: np.ndarray,
    decoded: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
    *,
    col_block_size: int,
    scale_bounds: tuple[float, float] | None = (0.0, 2.0),
    storage_dtype: np.dtype | type | None = np.float16,
    rcond: float = 1e-10,
    ridge: float = 0.0,
) -> RowBlockScaleRepairResult:
    """Optimize one multiplicative scale per output-row/input-column block.

    This specialized implementation avoids materializing a basis tensor with
    one full matrix per group.  The returned scale count replaces the original
    per-row quantizer scales in payload accounting.
    """

    wanted = _as_matrix(target, name="target")
    current = _as_matrix(decoded, name="decoded")
    if wanted.shape != current.shape:
        raise ValueError("target and decoded must share a shape")
    if col_block_size <= 0:
        raise ValueError("col_block_size must be positive")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    cov = _prepare_covariance_argument(covariance, wanted.shape[1]).matrix
    rows, cols = wanted.shape
    group_count = int(math.ceil(cols / float(col_block_size)))
    scales = np.ones((rows, group_count), dtype=np.float64)
    repaired = current.copy()
    max_stationarity = 0.0
    for row in range(rows):
        components = np.zeros((group_count, cols), dtype=np.float64)
        for group in range(group_count):
            start = group * col_block_size
            stop = min((group + 1) * col_block_size, cols)
            components[group, start:stop] = current[row, start:stop]
        error = current[row] - wanted[row]
        projected = components @ cov
        gram = projected @ components.T
        gradient = projected @ error
        inverse, _, _ = _symmetric_pinv(
            gram + float(ridge) * np.eye(group_count, dtype=np.float64), rcond=rcond
        )
        row_scales = 1.0 - inverse @ gradient
        if scale_bounds is not None:
            lower, upper = map(float, scale_bounds)
            if not lower <= upper:
                raise ValueError("invalid scale bounds")
            row_scales = np.clip(row_scales, lower, upper)
        if storage_dtype is not None:
            row_scales = row_scales.astype(storage_dtype).astype(np.float64)
        candidate = row_scales @ components
        old_cost = 0.5 * float(error @ cov @ error)
        new_error = candidate - wanted[row]
        new_cost = 0.5 * float(new_error @ cov @ new_error)
        if new_cost <= old_cost + 1e-10:
            repaired[row] = candidate
            scales[row] = row_scales
        else:
            new_error = error
        stationarity = components @ cov @ new_error
        stationarity_scale = max(
            float(np.linalg.norm(components @ cov) * np.linalg.norm(new_error)), EPS
        )
        max_stationarity = max(
            max_stationarity,
            float(np.max(np.abs(stationarity), initial=0.0) / stationarity_scale),
        )
    return RowBlockScaleRepairResult(
        repaired=repaired,
        scales=scales,
        cost_before=_input_hessian_quadratic_prepared(current - wanted, cov),
        cost_after=_input_hessian_quadratic_prepared(repaired - wanted, cov),
        stored_scale_count=rows * group_count,
        max_relative_stationarity=max_stationarity,
    )


@dataclass(frozen=True)
class OBSCorrectionResult:
    corrected_weight: np.ndarray
    delta: np.ndarray
    naive_cost: float
    corrected_cost: float
    schur_cost: float
    max_retained_stationarity: float
    relative_stationarity: float
    unique_support_count: int
    effective_ranks: np.ndarray
    rhs_null_residual_max: float


def obs_retained_support_correction(
    weight: np.ndarray,
    retained_mask: np.ndarray,
    input_hessian: np.ndarray | PreparedInputCovariance,
    *,
    damping: float = 0.0,
    rcond: float = 1e-10,
    psd_rtol: float = 1e-7,
    rhs_rtol: float = 1e-8,
    max_unique_supports: int | None = None,
) -> OBSCorrectionResult:
    """Refit retained values so pruning error is support-orthogonal in ``H``.

    The support is frozen.  The correction is folded into stored survivor
    values, so it adds no payload beyond the support and values already counted.
    """

    matrix = _as_matrix(weight, name="weight")
    mask = np.asarray(retained_mask, dtype=bool)
    if mask.shape != matrix.shape:
        raise ValueError(f"retained_mask shape mismatch: expected {matrix.shape}, got {mask.shape}")
    cov = _prepare_covariance_argument(
        input_hessian,
        matrix.shape[1],
        damping=damping,
        psd_rtol=psd_rtol,
    ).matrix
    keys = {row.tobytes() for row in mask}
    if max_unique_supports is not None and len(keys) > int(max_unique_supports):
        raise ValueError(f"unique support count {len(keys)} exceeds guard {max_unique_supports}")
    naive = np.where(mask, matrix, 0.0)
    naive_delta = naive - matrix
    corrected = naive.copy()
    effective_ranks = np.zeros(matrix.shape[0], dtype=np.int64)
    factor_cache: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rhs_null_residual_max = 0.0
    schur_cost = 0.0
    for row_index in range(matrix.shape[0]):
        row_mask = mask[row_index]
        retained = np.flatnonzero(row_mask)
        pruned = np.flatnonzero(~row_mask)
        if not pruned.size:
            effective_ranks[row_index] = int(retained.size)
            continue
        if not retained.size:
            delta_pruned = -matrix[row_index, pruned]
            schur_cost += 0.5 * float(delta_pruned @ cov[np.ix_(pruned, pruned)] @ delta_pruned)
            continue
        key = row_mask.tobytes()
        cached = factor_cache.get(key)
        if cached is None:
            c_rr = cov[np.ix_(retained, retained)]
            values, vectors = np.linalg.eigh(0.5 * (c_rr + c_rr.T))
            maximum = max(float(np.max(np.abs(values), initial=0.0)), EPS)
            active = values > float(rcond) * maximum
            factor_cache[key] = (retained, pruned, values, vectors)
        else:
            retained, pruned, values, vectors = cached
            maximum = max(float(np.max(np.abs(values), initial=0.0)), EPS)
            active = values > float(rcond) * maximum
        effective_ranks[row_index] = int(np.count_nonzero(active))
        delta_pruned = -matrix[row_index, pruned]
        rhs = -(cov[np.ix_(retained, pruned)] @ delta_pruned)
        if np.any(active):
            projection = vectors[:, active] @ (vectors[:, active].T @ rhs)
            solution = vectors[:, active] @ ((vectors[:, active].T @ rhs) / values[active])
        else:
            projection = np.zeros_like(rhs)
            solution = np.zeros_like(rhs)
        null_residual = float(np.linalg.norm(rhs - projection) / max(np.linalg.norm(rhs), EPS))
        rhs_null_residual_max = max(rhs_null_residual_max, null_residual)
        if null_residual > float(rhs_rtol):
            raise ValueError(
                f"OBS right-hand side leaves the retained Hessian range: residual={null_residual:.3e}"
            )
        corrected[row_index, retained] = matrix[row_index, retained] + solution
        row_delta = np.zeros(matrix.shape[1], dtype=np.float64)
        row_delta[pruned] = delta_pruned
        row_delta[retained] = solution
        schur_cost += 0.5 * float(row_delta @ cov @ row_delta)
    delta = corrected - matrix
    stationarity_matrix = delta @ cov
    retained_stationarity = np.abs(stationarity_matrix[mask])
    max_stationarity = float(np.max(retained_stationarity, initial=0.0))
    relative_stationarity = max_stationarity / max(float(np.linalg.norm(stationarity_matrix)), EPS)
    return OBSCorrectionResult(
        corrected_weight=corrected,
        delta=delta,
        naive_cost=_input_hessian_quadratic_prepared(naive_delta, cov),
        corrected_cost=_input_hessian_quadratic_prepared(delta, cov),
        schur_cost=float(schur_cost),
        max_retained_stationarity=max_stationarity,
        relative_stationarity=relative_stationarity,
        unique_support_count=len(keys),
        effective_ranks=effective_ranks,
        rhs_null_residual_max=rhs_null_residual_max,
    )


def quadratic_comfort_path(
    delta: np.ndarray,
    covariance: np.ndarray | PreparedInputCovariance,
    epsilons: Iterable[float],
) -> list[dict[str, float]]:
    full_cost = input_hessian_quadratic(delta, covariance)
    rows: list[dict[str, float]] = []
    for epsilon in epsilons:
        value = float(epsilon)
        rows.append(
            {
                "epsilon": value,
                "hessian_cost": value * value * full_cost,
                "full_hessian_cost": full_cost,
                "path_kind": "codec_endpoint" if value == 1.0 else "noncodec_interpolation",
            }
        )
    return rows
