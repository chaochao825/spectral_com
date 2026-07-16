from __future__ import annotations

"""Exact-rate Hessian repair probe on a small pretrained causal LM.

The script deliberately separates three questions which are easy to conflate:

* rate: every endpoint is charged for packed Q codes, FP16 scales, FP16 sparse
  values, realizable CSR support, and FP16 low-rank factors;
* local geometry: self terms, cross terms and signed Hessian cosines use the
  input-covariance (``C \\otimes I_out``) proxy;
* task behavior: held-out WikiText NLL/PPL and an epsilon interpolation probe
  test whether the local quadratic comfort zone survives at the codec endpoint.

Only the selected linear tensors are included in the reported payload ratio;
all other model tensors remain FP16 and unchanged.
"""

import argparse
import bisect
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import build_model_snapshot_manifest as model_snapshot_manifest
import run_pretrained_llm_orthogonality as base
from llm_spectral_dynamics.structured.codec_artifact import (
    LayerCodecAllocation,
    LayerCodecPayload,
    codec_artifact_allocation_natural_file_bytes,
    codec_artifact_allocations_layout,
    codec_artifact_allocations_natural_file_bytes,
    codec_artifact_natural_file_bytes,
    read_codec_artifact,
    read_fp16_reference_artifact,
    write_codec_artifact,
    write_fp16_reference_artifact,
)
from llm_spectral_dynamics.structured.hessian_repair import (
    PayloadBreakdown,
    PreparedInputCovariance,
    exact_payload_accounting,
    hessian_basis_repair,
    hessian_row_block_scale_repair,
    obs_retained_support_correction,
)


EPS = 1e-12
NUMERICAL_PSD_REJECTION_RTOL = 1e-7
FLOAT32_PSD_FLOOR_RTOL = float(8.0 * np.finfo(np.float32).eps)
GLOBAL_ALLOCATOR_FRONTIER_STATE_LIMIT = 100000
GLOBAL_ALLOCATOR_EXPANDED_STATE_LIMIT = 500000
STRATEGY_ORDER = (
    "Q",
    "Q_global_scale",
    "Q_block_scale",
    "Q+S",
    "Q+S_OBS",
    "Q+L",
    "Q+S_OBS_global",
    "Q+L_global",
    "Q+S_OBS_or_L_global",
    "Q+S+L_QL_budget",
    "Q+S+L_QL_budget_component_scale",
    "Q+S+L",
    "Q+S_OBS+L",
    "Q+S+L_component_scale",
)


class ExactNaturalMatchSearchLimitError(RuntimeError):
    """The optional exact-byte counterfactual search exceeded its hard limit."""

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


GLOBAL_SINGLE_COMPONENT_STRATEGIES = (
    "Q+S_OBS_global",
    "Q+L_global",
)

GLOBAL_NONJOINT_CONTROL_STRATEGY = "Q+S_OBS_or_L_global"
GLOBAL_CONTROL_STRATEGIES = (
    *GLOBAL_SINGLE_COMPONENT_STRATEGIES,
    GLOBAL_NONJOINT_CONTROL_STRATEGY,
)

PLOT_STRATEGY_LABELS = {
    "Q_global_scale": "Q global-scale",
    "Q_block_scale": "Q block-scale",
    "Q+S_OBS": "Q+S OBS",
    "Q+S_OBS_global": "Q+S OBS (global)",
    "Q+L_global": "Q+L (global)",
    "Q+S_OBS_or_L_global": "Q+S OBS or Q+L (global, no joint layer)",
    "Q+S+L_QL_budget": "QSL <= QL bits",
    "Q+S+L_QL_budget_component_scale": "QSL <= QL + scale",
    "Q+S_OBS+L": "Q+S OBS+L",
    "Q+S+L_component_scale": "QSL + scale",
}


def _plot_strategy_label(strategy: object) -> str:
    value = str(strategy)
    return PLOT_STRATEGY_LABELS.get(value, value)


def _runtime_fp16(value: np.ndarray) -> np.ndarray:
    """Round a decoded tensor to the runtime/storage endpoint used in this probe."""

    return np.asarray(value, dtype=np.float16).astype(np.float32)


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _format_float(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _text_digest(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_snapshot(
    model_snapshot_manifest_path: str = "",
) -> dict[str, dict[str, object]]:
    """Hash the source files that define future numerical and codec runs."""

    repo_root = Path(__file__).resolve().parents[1]
    paths = {
        "runner": Path(__file__).resolve(),
        "codec": repo_root / "src" / "llm_spectral_dynamics" / "structured" / "codec_artifact.py",
        "hessian_repair": repo_root / "src" / "llm_spectral_dynamics" / "structured" / "hessian_repair.py",
        "base_runner": repo_root / "scripts" / "run_pretrained_llm_orthogonality.py",
    }
    if model_snapshot_manifest_path:
        paths["model_snapshot_tool"] = (
            repo_root / "scripts" / "build_model_snapshot_manifest.py"
        )
    snapshot: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        snapshot[name] = {
            "path": path.relative_to(repo_root).as_posix(),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return snapshot


def prepare_fresh_output_dir(path: Path) -> Path:
    """Create an auditable output directory and refuse stale/partial reuse."""

    target = Path(path)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(
            f"output directory must be absent or empty; move the prior run aside first: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    (target / "RUNNING").write_text(
        json.dumps({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def mark_output_complete(path: Path) -> None:
    """Atomically publish a completion marker after every required output exists."""

    target = Path(path)
    running = target / "RUNNING"
    if not running.is_file():
        raise RuntimeError(f"missing RUNNING marker for {target}")
    temporary = target / ".COMPLETED.tmp"
    temporary.write_text(
        json.dumps({"completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target / "COMPLETED")
    running.unlink()


def split_content_disjoint_text_windows(args: argparse.Namespace, texts: list[str]) -> None:
    """Create calibration/evaluation/recovery splits with unique source text.

    The upstream splitter guarantees disjoint source *positions* but may place
    a duplicated dataset row in two splits.  This stricter variant preserves
    first-occurrence order, removes exact duplicate content, and refuses to
    repeat rows when the requested evidence cannot be formed.
    """

    calib_count = max(int(args.calib_limit) * int(args.texts_per_batch_window), 1)
    eval_count = max(int(args.eval_limit) * int(args.texts_per_batch_window), 1)
    recovery_batches = max(
        int(getattr(args, "spq_lora_train_limit", 0)),
        int(getattr(args, "spq_lora_steps", 0)),
        1,
    )
    recovery_count = max(recovery_batches * int(args.texts_per_batch_window), 1)
    needed = calib_count + eval_count + recovery_count
    unique_texts: list[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = str(text).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_texts.append(normalized)
        if len(unique_texts) >= needed:
            break
    if len(unique_texts) < needed:
        raise RuntimeError(
            "content-disjoint evidence requires "
            f"{needed} unique texts, but only {len(unique_texts)} were loaded"
        )
    args.calib_texts = unique_texts[:calib_count]
    args.eval_texts = unique_texts[calib_count : calib_count + eval_count]
    args.recovery_texts = unique_texts[calib_count + eval_count : needed]
    args.text_split_policy = "content_disjoint_sequential_text_windows"
    args.unique_text_pool_count = len(unique_texts)


def _load_unique_dataset_role_texts(
    args: argparse.Namespace,
    *,
    split: str,
    role: str,
    needed: int,
    excluded: set[str],
) -> tuple[list[str], str, list[dict[str, object]], int]:
    """Load one dataset split and reject duplicate content across evidence roles."""

    if needed <= 0:
        raise ValueError(f"{role} requires a positive text count")
    role_args = copy.copy(args)
    role_args.data_cfg = {**args.data_cfg, "split": split}
    margin = max(64, int(args.texts_per_batch_window) * 4)
    pool, source, metadata = base.load_eval_texts(
        role_args,
        limit=needed + margin + len(excluded),
    )
    if not str(source).startswith("dataset:wikitext"):
        raise RuntimeError(f"real WikiText was required for {role}, got {source!r}")
    selected: list[str] = []
    local_seen: set[str] = set()
    for text in pool:
        normalized = str(text).strip()
        if not normalized or normalized in local_seen or normalized in excluded:
            continue
        local_seen.add(normalized)
        selected.append(normalized)
        if len(selected) >= needed:
            break
    if len(selected) < needed:
        raise RuntimeError(
            f"{role} requires {needed} unique non-overlapping texts from split "
            f"{split!r}, but only {len(selected)} were available"
        )
    annotated_metadata = [
        {**row, "evidence_role": role, "requested_split": split}
        for row in metadata
    ]
    return selected, source, annotated_metadata, len(pool)


def load_two_stage_text_windows(args: argparse.Namespace) -> None:
    """Bind calibration, allocation selection, and final evaluation to separate splits."""

    role_splits = {
        "calibration": str(args.calibration_split),
        "selection": str(args.selection_split),
        "test": str(args.test_split),
    }
    if len(set(role_splits.values())) != len(role_splits):
        raise ValueError(
            "two-stage selection requires distinct calibration, selection, and test splits"
        )
    texts_per_window = int(args.texts_per_batch_window)
    calib_count = max(int(args.calib_limit) * texts_per_window, 1)
    selection_count = max(int(args.selection_limit) * texts_per_window, 1)
    comfort_limit = 1 if args.skip_comfort else int(args.eval_limit)
    recovery_count = max(comfort_limit * texts_per_window, 1)
    eval_count = max(int(args.eval_limit) * texts_per_window, 1)

    excluded: set[str] = set()
    calibration, calibration_source, calibration_metadata, calibration_pool_count = (
        _load_unique_dataset_role_texts(
            args,
            split=role_splits["calibration"],
            role="calibration_proxy_fit",
            needed=calib_count,
            excluded=excluded,
        )
    )
    excluded.update(calibration)
    validation, selection_source, selection_metadata, selection_pool_count = (
        _load_unique_dataset_role_texts(
            args,
            split=role_splits["selection"],
            role="allocation_validation_and_comfort",
            needed=selection_count + recovery_count,
            excluded=excluded,
        )
    )
    args.selection_texts = validation[:selection_count]
    args.recovery_texts = validation[selection_count:]
    excluded.update(validation)
    evaluation, test_source, test_metadata, test_pool_count = (
        _load_unique_dataset_role_texts(
            args,
            split=role_splits["test"],
            role="final_test_evaluation",
            needed=eval_count,
            excluded=excluded,
        )
    )

    args.calib_texts = calibration
    args.eval_texts = evaluation
    args.comfort_texts = args.recovery_texts
    args.comfort_eval_limit = comfort_limit
    args.comfort_evidence_role = "validation_comfort_disjoint_from_allocation_selection"
    args.text_split_policy = (
        "independent_train_validation_test_dataset_splits_with_cross_role_content_deduplication"
    )
    args.text_source_used = (
        f"{calibration_source}|{selection_source}|{test_source}"
    )
    args.text_source_metadata = [
        *calibration_metadata,
        *selection_metadata,
        *test_metadata,
    ]
    args.text_pool_count = (
        calibration_pool_count + selection_pool_count + test_pool_count
    )
    args.unique_text_pool_count = len(excluded) + len(evaluation)
    args.data_role_splits = role_splits


def _git_snapshot() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
    }


def _model_identity(model: torch.nn.Module, tokenizer: object, requested_revision: str) -> dict[str, object]:
    config = getattr(model, "config", None)
    config_payload = config.to_dict() if config is not None and hasattr(config, "to_dict") else {}
    serialized = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "requested_revision": requested_revision or None,
        "resolved_model_commit_hash": getattr(config, "_commit_hash", None),
        "resolved_tokenizer_commit_hash": tokenizer_kwargs.get("_commit_hash"),
        "model_config_sha256": hashlib.sha256(serialized).hexdigest(),
        "model_name_or_path": getattr(config, "_name_or_path", None),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }


@dataclass(frozen=True)
class QuantCodec:
    codes: np.ndarray
    scales: np.ndarray
    bits: int
    col_block_size: int | None = None
    quantizer: str = "symmetric_rtn"

    @property
    def scale_count(self) -> int:
        return int(self.scales.size)

    def decode(self) -> np.ndarray:
        codes = self.codes.astype(np.float32, copy=False)
        scales = self.scales.astype(np.float32, copy=False)
        if self.col_block_size is None:
            return _runtime_fp16(codes * scales.reshape(-1, 1))
        rows, cols = codes.shape
        decoded = np.empty((rows, cols), dtype=np.float32)
        for group in range(scales.shape[1]):
            start = group * int(self.col_block_size)
            stop = min((group + 1) * int(self.col_block_size), cols)
            decoded[:, start:stop] = codes[:, start:stop] * scales[:, group : group + 1]
        return _runtime_fp16(decoded)

    def scaled(self, multiplier: float) -> "QuantCodec":
        stored = np.asarray(self.scales.astype(np.float32) * float(multiplier), dtype=np.float16)
        return QuantCodec(
            self.codes,
            stored,
            self.bits,
            self.col_block_size,
            self.quantizer,
        )


@dataclass(frozen=True)
class SparseCodec:
    values: np.ndarray
    mask: np.ndarray

    @property
    def nonzero_count(self) -> int:
        # The support is frozen: a survivor rounded to zero still consumes a slot.
        return int(np.count_nonzero(self.mask))

    def decode(self) -> np.ndarray:
        return np.where(self.mask, self.values.astype(np.float32), 0.0).astype(np.float32)

    def scaled(self, multiplier: float) -> "SparseCodec":
        scaled = np.asarray(self.values.astype(np.float32) * float(multiplier), dtype=np.float16)
        return SparseCodec(np.where(self.mask, scaled, np.float16(0.0)), self.mask)


@dataclass(frozen=True)
class LowRankCodec:
    left: np.ndarray
    right: np.ndarray
    factor_bits: int = 16
    left_scales: np.ndarray | None = None
    right_scales: np.ndarray | None = None
    quantizer: str = "fp16"

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def scale_count(self) -> int:
        if self.factor_bits == 16:
            return 0
        assert self.left_scales is not None and self.right_scales is not None
        return int(self.left_scales.size + self.right_scales.size)

    def _decode_factor(
        self,
        values: np.ndarray,
        scales: np.ndarray | None,
    ) -> np.ndarray:
        if self.factor_bits == 16:
            return values.astype(np.float32)
        if scales is None:
            raise ValueError("quantized low-rank factors require stored scales")
        return values.astype(np.float32) * scales.astype(np.float32).reshape(-1, 1)

    def decode(self) -> np.ndarray:
        left = self._decode_factor(self.left, self.left_scales)
        right = self._decode_factor(self.right, self.right_scales)
        product = left @ right
        return _runtime_fp16(product)

    def scaled(self, multiplier: float) -> "LowRankCodec":
        if self.factor_bits == 16:
            left = np.asarray(
                self.left.astype(np.float32) * float(multiplier),
                dtype=np.float16,
            )
            return LowRankCodec(left, self.right)
        assert self.left_scales is not None
        stored_scales = np.asarray(
            self.left_scales.astype(np.float32) * float(multiplier),
            dtype=np.float16,
        )
        return LowRankCodec(
            self.left,
            self.right,
            factor_bits=self.factor_bits,
            left_scales=stored_scales,
            right_scales=self.right_scales,
            quantizer=self.quantizer,
        )


@dataclass
class Candidate:
    strategy: str
    target_ratio: float
    layer: str
    weight: np.ndarray
    q: QuantCodec
    sparse: SparseCodec | None = None
    lowrank: LowRankCodec | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    repair_dof: int = 0

    @property
    def q_decoded(self) -> np.ndarray:
        return self.q.decode()

    @property
    def sparse_decoded(self) -> np.ndarray:
        return np.zeros_like(self.weight, dtype=np.float32) if self.sparse is None else self.sparse.decode()

    @property
    def lowrank_decoded(self) -> np.ndarray:
        return np.zeros_like(self.weight, dtype=np.float32) if self.lowrank is None else self.lowrank.decode()

    @property
    def pre_runtime_sum(self) -> np.ndarray:
        return self.q_decoded + self.sparse_decoded + self.lowrank_decoded

    @property
    def final(self) -> np.ndarray:
        return _runtime_fp16(self.pre_runtime_sum)

    @property
    def sparse_nnz(self) -> int:
        return 0 if self.sparse is None else self.sparse.nonzero_count

    @property
    def rank(self) -> int:
        return 0 if self.lowrank is None else self.lowrank.rank

    def payload(self, *, support_encoding: str) -> PayloadBreakdown:
        return exact_payload_accounting(
            tuple(map(int, self.weight.shape)),
            base_code_bits=int(self.q.bits),
            base_scale_count=self.q.scale_count,
            base_scale_bits=16,
            sparse_mask=None if self.sparse is None else self.sparse.mask,
            sparse_value_bits=16,
            support_encoding=support_encoding,
            lowrank_rank=self.rank,
            lowrank_factor_bits=(
                16 if self.lowrank is None else int(self.lowrank.factor_bits)
            ),
            lowrank_scale_count=(
                0 if self.lowrank is None else int(self.lowrank.scale_count)
            ),
            repair_param_count=int(self.repair_dof),
            repair_param_bits=16,
            repair_folded=True,
        )


@dataclass(frozen=True)
class RankedGlobalAllocation:
    """One exact-file-feasible allocation retained by proxy screening."""

    candidates: dict[str, Candidate]
    natural_file_bytes: int
    hessian_cost: float
    choices: tuple[int, ...]
    allocation_digest: str


def _artifact_layer(candidate: Candidate) -> LayerCodecPayload:
    """Map one selected endpoint into the independently decodable codec API."""

    return LayerCodecPayload(
        name=candidate.layer,
        q_codes=candidate.q.codes,
        q_scales=candidate.q.scales,
        q_bits=candidate.q.bits,
        q_col_block_size=candidate.q.col_block_size,
        sparse_values=None if candidate.sparse is None else candidate.sparse.values,
        sparse_mask=None if candidate.sparse is None else candidate.sparse.mask,
        lowrank_left=None if candidate.lowrank is None else candidate.lowrank.left,
        lowrank_right=None if candidate.lowrank is None else candidate.lowrank.right,
        lowrank_factor_bits=(
            16 if candidate.lowrank is None else candidate.lowrank.factor_bits
        ),
        lowrank_left_scales=(
            None if candidate.lowrank is None else candidate.lowrank.left_scales
        ),
        lowrank_right_scales=(
            None if candidate.lowrank is None else candidate.lowrank.right_scales
        ),
    )


def _artifact_allocation(candidate: Candidate) -> LayerCodecAllocation:
    """Return the value-independent metadata that fixes serialized bytes."""

    shape = tuple(map(int, candidate.weight.shape))
    if tuple(map(int, candidate.q.codes.shape)) != shape:
        raise ValueError("candidate quantized-code shape differs from its weight")
    if candidate.sparse is not None and (
        tuple(map(int, candidate.sparse.values.shape)) != shape
        or tuple(map(int, candidate.sparse.mask.shape)) != shape
    ):
        raise ValueError("candidate sparse component shape differs from its weight")
    if candidate.lowrank is not None and (
        candidate.lowrank.left.shape[0] != shape[0]
        or candidate.lowrank.right.shape[1] != shape[1]
        or candidate.lowrank.left.shape[1] != candidate.lowrank.right.shape[0]
    ):
        raise ValueError("candidate low-rank component shape differs from its weight")
    return LayerCodecAllocation(
        name=candidate.layer,
        shape=shape,
        q_bits=int(candidate.q.bits),
        q_scale_shape=tuple(map(int, candidate.q.scales.shape)),
        q_col_block_size=candidate.q.col_block_size,
        sparse_nnz=candidate.sparse_nnz,
        lowrank_rank=candidate.rank,
        lowrank_factor_bits=(
            16 if candidate.lowrank is None else candidate.lowrank.factor_bits
        ),
        lowrank_left_scale_shape=(
            ()
            if candidate.lowrank is None
            or candidate.lowrank.left_scales is None
            else tuple(map(int, candidate.lowrank.left_scales.shape))
        ),
        lowrank_right_scale_shape=(
            ()
            if candidate.lowrank is None
            or candidate.lowrank.right_scales is None
            else tuple(map(int, candidate.lowrank.right_scales.shape))
        ),
    )


def _allocation_artifact_file_bytes(
    *,
    layer: str,
    q: QuantCodec,
    shape: tuple[int, int],
    sparse_nonzero: int,
    rank: int,
    lowrank_factor_bits: int = 16,
    alignment: int,
) -> int:
    """Measure an allocation exactly from shape/encoding metadata only."""

    rows, cols = map(int, shape)
    nonzero = int(sparse_nonzero)
    bounded_rank = int(rank)
    if nonzero < 0 or nonzero > rows * cols:
        raise ValueError("sparse_nonzero is outside the tensor extent")
    if bounded_rank < 0 or bounded_rank > min(rows, cols):
        raise ValueError("rank is outside the matrix dimensions")
    return codec_artifact_allocation_natural_file_bytes(
        name=layer,
        shape=(rows, cols),
        q_bits=q.bits,
        q_scale_shape=tuple(map(int, q.scales.shape)),
        q_col_block_size=q.col_block_size,
        sparse_nnz=nonzero,
        lowrank_rank=bounded_rank,
        lowrank_factor_bits=int(lowrank_factor_bits),
        lowrank_left_scale_shape=(
            (rows,)
            if bounded_rank > 0 and int(lowrank_factor_bits) != 16
            else ()
        ),
        lowrank_right_scale_shape=(
            (bounded_rank,)
            if bounded_rank > 0 and int(lowrank_factor_bits) != 16
            else ()
        ),
        alignment=alignment,
    )


def _max_sparse_under_serialized_budget(
    *,
    layer: str,
    q: QuantCodec,
    shape: tuple[int, int],
    rank: int,
    logical_max_nonzero: int,
    budget_file_bytes: int,
    lowrank_factor_bits: int = 16,
    alignment: int,
) -> int:
    """Binary-search the largest CSR support whose real artifact fits."""

    maximum = int(logical_max_nonzero)
    budget = int(budget_file_bytes)
    if maximum < 0:
        raise ValueError("logical_max_nonzero must be non-negative")

    def size(nonzero: int) -> int:
        return _allocation_artifact_file_bytes(
            layer=layer,
            q=q,
            shape=shape,
            sparse_nonzero=nonzero,
            rank=rank,
            lowrank_factor_bits=lowrank_factor_bits,
            alignment=alignment,
        )

    if size(0) > budget:
        return -1
    if size(maximum) <= budget:
        return maximum
    low, high = 0, maximum
    while low < high:
        middle = (low + high + 1) // 2
        if size(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


class LowRankFactorizer:
    """Whitened/SVD factorizer which caches the covariance eigendecomposition."""

    def __init__(
        self,
        covariance: torch.Tensor,
        *,
        method: str,
        device: str,
        whitening_floor_ratio: float = 1e-5,
        svd_solver: str = "auto",
        randomized_oversampling: int = 4,
        randomized_niter: int = 2,
        randomized_seed: int = 0,
        seed_namespace: str = "",
    ) -> None:
        self.method = str(method)
        self.device = str(device)
        self.whitening_floor_ratio = float(whitening_floor_ratio)
        self.svd_solver = str(svd_solver)
        self.randomized_oversampling = int(randomized_oversampling)
        self.randomized_niter = int(randomized_niter)
        self.randomized_seed = int(randomized_seed)
        self.seed_namespace = str(seed_namespace)
        self._solver_call_counts: dict[str, int] = {}
        if not math.isfinite(self.whitening_floor_ratio) or self.whitening_floor_ratio < 0.0:
            raise ValueError("whitening floor ratio must be finite and non-negative")
        if self.svd_solver not in {"auto", "full", "randomized"}:
            raise ValueError(f"unsupported SVD solver: {self.svd_solver}")
        if self.randomized_oversampling < 0 or self.randomized_niter < 0:
            raise ValueError("randomized SVD oversampling/niter must be non-negative")
        self.sqrt_h: torch.Tensor | None = None
        self.inv_sqrt_h: torch.Tensor | None = None
        self.diagnostics: dict[str, object] = {
            "method": self.method,
            "whitening_floor_ratio": self.whitening_floor_ratio,
            "svd_solver": self.svd_solver,
            "randomized_oversampling": self.randomized_oversampling,
            "randomized_niter": self.randomized_niter,
            "randomized_seed": self.randomized_seed,
            "randomized_seed_namespace": self.seed_namespace,
            "randomized_seed_scheme": "sha256(job_seed|layer|rank), isolated with torch.random.fork_rng",
            "resolved_svd_solvers": [],
            "svd_solver_call_counts": {},
            "factorizer_regularization_applied": False,
        }
        if self.method not in {"svd", "whitened_svd"}:
            raise ValueError(f"unsupported low-rank method: {self.method}")
        if self.method == "whitened_svd":
            h = covariance.float().to(self.device)
            evals, evecs = torch.linalg.eigh(0.5 * (h + h.transpose(0, 1)))
            # This is a factorizer-only numerical regularizer.  It does not
            # alter the declared metric used to score the decoded endpoint,
            # and is therefore recorded separately from covariance damping.
            floor = torch.clamp(
                evals.max() * self.whitening_floor_ratio,
                min=torch.finfo(evals.dtype).tiny,
            )
            clipped = evals < floor
            fitted_evals = torch.clamp(evals, min=floor)
            maximum = float(evals.max().detach().cpu()) if evals.numel() else 0.0
            minimum = float(evals.min().detach().cpu()) if evals.numel() else 0.0
            fitted_minimum = (
                float(fitted_evals.min().detach().cpu()) if fitted_evals.numel() else 0.0
            )
            self.diagnostics.update(
                {
                    "score_covariance_min_eigenvalue": minimum,
                    "score_covariance_max_eigenvalue": maximum,
                    "fit_covariance_floor": float(floor.detach().cpu()),
                    "fit_covariance_clipped_eigenvalues": int(clipped.sum().detach().cpu()),
                    "fit_covariance_dimension": int(evals.numel()),
                    "fit_covariance_condition_after_floor": maximum
                    / max(fitted_minimum, float(torch.finfo(evals.dtype).tiny)),
                    "factorizer_regularization_applied": bool(torch.any(clipped).detach().cpu()),
                }
            )
            self.sqrt_h = evecs @ torch.diag(torch.sqrt(fitted_evals)) @ evecs.transpose(0, 1)
            self.inv_sqrt_h = evecs @ torch.diag(torch.rsqrt(fitted_evals)) @ evecs.transpose(0, 1)

    def factorize(self, residual: np.ndarray, rank: int) -> LowRankCodec | None:
        rows, cols = residual.shape
        bounded = max(0, min(int(rank), min(rows, cols)))
        if bounded <= 0:
            return None
        work = torch.from_numpy(np.asarray(residual, dtype=np.float32)).to(self.device)
        if self.method == "whitened_svd":
            assert self.sqrt_h is not None and self.inv_sqrt_h is not None
            transformed = work @ self.sqrt_h
            decomposition_target = transformed
        else:
            decomposition_target = work

        minimum_dimension = min(decomposition_target.shape)
        use_randomized = self.svd_solver == "randomized" or (
            self.svd_solver == "auto" and decomposition_target.numel() >= 8_000_000
        )
        randomized_q = min(
            minimum_dimension,
            max(bounded, bounded + self.randomized_oversampling),
        )
        if use_randomized and randomized_q < minimum_dimension:
            seed_payload = (
                f"{self.randomized_seed}|{self.seed_namespace}|rank={bounded}"
            ).encode("utf-8")
            call_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big") % (
                2**31 - 1
            )
            fork_devices: list[int] = []
            if decomposition_target.is_cuda:
                device_index = decomposition_target.device.index
                fork_devices = [
                    torch.cuda.current_device() if device_index is None else int(device_index)
                ]
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                torch.manual_seed(call_seed)
                if decomposition_target.is_cuda:
                    torch.cuda.manual_seed(call_seed)
                u, singular, v = torch.svd_lowrank(
                    decomposition_target,
                    q=randomized_q,
                    niter=self.randomized_niter,
                )
            vh = v.transpose(0, 1)
            resolved_solver = "torch.svd_lowrank"
            self.diagnostics["last_randomized_seed"] = call_seed
        else:
            u, singular, vh = torch.linalg.svd(decomposition_target, full_matrices=False)
            resolved_solver = "torch.linalg.svd"
        self.diagnostics["last_resolved_svd_solver"] = resolved_solver
        self.diagnostics["last_randomized_q"] = (
            randomized_q if resolved_solver == "torch.svd_lowrank" else 0
        )
        self._solver_call_counts[resolved_solver] = self._solver_call_counts.get(resolved_solver, 0) + 1
        self.diagnostics["resolved_svd_solvers"] = sorted(self._solver_call_counts)
        self.diagnostics["svd_solver_call_counts"] = dict(sorted(self._solver_call_counts.items()))
        left = u[:, :bounded] * singular[:bounded]
        right = vh[:bounded, :]
        if self.method == "whitened_svd":
            assert self.inv_sqrt_h is not None
            right = right @ self.inv_sqrt_h

        # Balance each rank-one term before FP16 storage; the dense result is
        # unchanged before rounding, but neither factor unnecessarily underflows.
        left = left.float()
        right = right.float()
        for index in range(bounded):
            left_norm = torch.linalg.norm(left[:, index]).clamp_min(EPS)
            right_norm = torch.linalg.norm(right[index, :]).clamp_min(EPS)
            balance = torch.sqrt(right_norm / left_norm)
            left[:, index] *= balance
            right[index, :] /= balance
        stored_left = left.to(dtype=torch.float16, device="cpu").numpy()
        stored_right = right.to(dtype=torch.float16, device="cpu").numpy()
        return LowRankCodec(stored_left, stored_right)


def prepare_metric_covariance(
    covariance: torch.Tensor,
    *,
    mode: str = "full",
    damping_ratio: float = 0.0,
    psd_rtol: float = NUMERICAL_PSD_REJECTION_RTOL,
    storage_floor_rtol: float = FLOAT32_PSD_FLOOR_RTOL,
) -> tuple[
    torch.Tensor,
    PreparedInputCovariance,
    dict[str, float | bool | str],
]:
    """Validate once and return the float32 PSD endpoint-scoring geometry.

    Activation Grams are accumulated in float64 but stored by the upstream
    collector as float32.  A cast can introduce a tiny negative eigenvalue even
    after the declared mean-diagonal ridge.  We reject materially indefinite
    matrices, then add an eight-float32-epsilon spectral floor.  Endpoint
    scoring, OBS, sparse selection, candidate costs, Gram matrices, and rho use
    this geometry; whitened-SVD fitting may apply its separately recorded
    factorizer-only eigenvalue floor.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got shape {tuple(covariance.shape)}")
    if mode not in {"full", "diagonal", "identity"}:
        raise ValueError(f"unsupported covariance mode: {mode}")
    if not math.isfinite(float(damping_ratio)) or float(damping_ratio) < 0.0:
        raise ValueError("covariance damping ratio must be finite and non-negative")

    collected = covariance.detach().cpu().double().numpy()
    collected = 0.5 * (collected + collected.T)
    collected_eigenvalues = np.linalg.eigvalsh(collected)
    collected_scale = float(np.max(np.abs(collected_eigenvalues), initial=0.0))
    collected_minimum = float(np.min(collected_eigenvalues)) if collected_eigenvalues.size else 0.0
    collected_diagonal_mean = float(np.mean(np.diag(collected))) if collected.size else 0.0
    if collected_minimum < -float(psd_rtol) * collected_scale:
        raise ValueError(
            "collected covariance must be positive semidefinite; it is materially indefinite "
            "before the requested geometry ablation"
        )

    if mode == "diagonal":
        original_diagonal = np.diag(collected).copy()
        original = np.diag(original_diagonal)
        mode_eigenvalues = original_diagonal
    elif mode == "identity":
        # Preserve the average activation energy while removing both
        # anisotropy and off-diagonal correlations.  A genuinely zero Gram
        # remains zero instead of receiving an arbitrary unit scale.
        identity_scale = max(collected_diagonal_mean, 0.0)
        original = np.eye(collected.shape[0], dtype=np.float64) * identity_scale
        mode_eigenvalues = np.full(collected.shape[0], identity_scale, dtype=np.float64)
    else:
        original = collected.copy()
        mode_eigenvalues = collected_eigenvalues

    damping_scale = max(float(np.mean(np.diag(original))), 0.0) if original.size else 0.0
    configured_damping = float(damping_ratio) * damping_scale
    if configured_damping:
        original = original + configured_damping * np.eye(original.shape[0], dtype=np.float64)
    # Damping is an identity shift, so its spectrum is obtained algebraically;
    # diagonal/identity mode spectra are analytic and full mode reuses the
    # already-computed collected spectrum.  This avoids repeated O(d^3)
    # decompositions on large selected projections.
    original_eigenvalues = mode_eigenvalues + configured_damping
    original_scale = float(np.max(np.abs(original_eigenvalues), initial=0.0))
    original_minimum = float(np.min(original_eigenvalues)) if original_eigenvalues.size else 0.0
    if original_minimum < -float(psd_rtol) * original_scale:
        raise ValueError("requested covariance geometry is materially indefinite")
    target_floor = float(storage_floor_rtol) * original_scale
    floor_shift = 0.0
    if original_minimum < target_floor:
        numerical_margin = np.finfo(np.float64).eps * original_scale * 16.0
        floor_shift = target_floor - original_minimum + numerical_margin
    repaired = original
    if floor_shift:
        repaired = repaired + floor_shift * np.eye(original.shape[0], dtype=np.float64)
    prepared = torch.from_numpy(repaired).to(device=covariance.device, dtype=torch.float32)
    prepared_input = PreparedInputCovariance._from_validated_array(repaired)
    # All subsequent transformations are scalar identity shifts, so the final
    # spectrum follows exactly in real arithmetic from the one validated
    # eigenspectrum above.  The scale-relative floor is eight float32 epsilons,
    # leaving a declared storage margin without another cubic decomposition.
    final_eigenvalues = original_eigenvalues + floor_shift
    final_scale = float(np.max(np.abs(final_eigenvalues), initial=0.0))
    final_minimum = float(np.min(final_eigenvalues)) if final_eigenvalues.size else 0.0
    if final_minimum < 0.0:
        raise ValueError("prepared covariance is not positive semidefinite")
    diagonal_shift = float(floor_shift)
    scale_for_ratio = original_scale if original_scale > 0.0 else 1.0
    report: dict[str, float | bool | str] = {
        "covariance_mode": mode,
        "configured_damping_ratio": float(damping_ratio),
        "configured_damping": configured_damping,
        "collected_min_eigenvalue": collected_minimum,
        "collected_spectral_scale": collected_scale,
        "collected_diagonal_mean": collected_diagonal_mean,
        "original_min_eigenvalue": original_minimum,
        "original_spectral_scale": original_scale,
        "original_min_relative": original_minimum / scale_for_ratio,
        "final_min_eigenvalue": final_minimum,
        "final_spectral_scale": final_scale,
        "diagonal_shift": diagonal_shift,
        "diagonal_shift_relative": diagonal_shift / scale_for_ratio,
        "repair_applied": bool(diagonal_shift > 0.0),
        "spectrum_decomposition_count": 1,
        "final_spectrum_source": "algebraic_identity_shift_from_validated_spectrum",
        "downstream_covariance_binding": "immutable_prevalidated_input_covariance",
        "psd_rejection_rtol": float(psd_rtol),
        "float32_storage_floor_rtol": float(storage_floor_rtol),
    }
    return prepared, prepared_input, report


def fit_sparse_lowrank_components(
    *,
    residual_q: np.ndarray,
    covariance_tensor: torch.Tensor,
    factorizer: LowRankFactorizer,
    nonzero: int,
    rank: int,
    sparse_method: str,
    residual_order: str,
    torch_dtype: torch.dtype,
    sparse_refit: str = "naive",
    covariance: PreparedInputCovariance | None = None,
    metric: HessianMetric | None = None,
    obs_rcond: float = 1e-10,
    lowrank_factor_bits: int = 16,
    lowrank_factor_quantizer: str = "symmetric_mse_clip",
) -> tuple[SparseCodec | None, LowRankCodec | None, dict[str, object]]:
    """Fit a byte-identical sparse/low-rank allocation in a declared order.

    The two orders use the same ``nnz`` and rank ledger.  They differ only in
    which component sees the unmodelled residual first, making non-commutation
    an explicit experimental factor rather than an undocumented convention.
    """

    if sparse_refit not in {"naive", "obs"}:
        raise ValueError(f"unsupported sparse refit: {sparse_refit}")
    if sparse_refit == "obs" and (covariance is None or metric is None):
        raise ValueError("OBS sparse refit requires covariance and metric")

    if residual_order == "s_then_l":
        sparse = sparse_codec_from_residual(
            residual_q,
            covariance_tensor,
            nonzero,
            method=sparse_method,
            torch_dtype=torch_dtype,
        )
        diagnostics: dict[str, object] = {"strict_sparse_refit": sparse_refit}
        if sparse_refit == "obs":
            assert covariance is not None and metric is not None
            sparse, obs_diagnostics = obs_refit_sparse(
                residual_q,
                sparse,
                covariance,
                metric=metric,
                rcond=obs_rcond,
            )
            diagnostics.update(obs_diagnostics)
        sparse_decoded = np.zeros_like(residual_q) if sparse is None else sparse.decode()
        lowrank = quantize_lowrank_factors(
            factorizer.factorize(residual_q - sparse_decoded, rank),
            bits=lowrank_factor_bits,
            quantizer=lowrank_factor_quantizer,
        )
        return sparse, lowrank, diagnostics
    if residual_order == "l_then_s":
        lowrank = quantize_lowrank_factors(
            factorizer.factorize(residual_q, rank),
            bits=lowrank_factor_bits,
            quantizer=lowrank_factor_quantizer,
        )
        lowrank_decoded = np.zeros_like(residual_q) if lowrank is None else lowrank.decode()
        sparse_target = residual_q - lowrank_decoded
        sparse = sparse_codec_from_residual(
            sparse_target,
            covariance_tensor,
            nonzero,
            method=sparse_method,
            torch_dtype=torch_dtype,
        )
        diagnostics = {"strict_sparse_refit": sparse_refit}
        if sparse_refit == "obs":
            assert covariance is not None and metric is not None
            sparse, obs_diagnostics = obs_refit_sparse(
                sparse_target,
                sparse,
                covariance,
                metric=metric,
                rcond=obs_rcond,
            )
            diagnostics.update(obs_diagnostics)
        return sparse, lowrank, diagnostics
    raise ValueError(f"unsupported residual order: {residual_order}")


class HessianMetric:
    """Reusable torch implementation of the input-covariance metric.

    The experiment constructs many discrete S/L allocations. Keeping ``C`` on
    the SVD device avoids repeated multi-billion-op float64 CPU products for
    wide MLP projections while preserving the same ``tr(d C d^T)`` proxy.
    """

    def __init__(self, covariance: torch.Tensor, *, device: str) -> None:
        self.device = str(device)
        self.covariance = covariance.detach().float().to(self.device)

    def inner(self, left: np.ndarray, right: np.ndarray) -> float:
        a = torch.from_numpy(np.asarray(left, dtype=np.float32)).to(self.device)
        b = torch.from_numpy(np.asarray(right, dtype=np.float32)).to(self.device)
        return float(torch.sum((a @ self.covariance) * b).detach().cpu())

    def cost(self, delta: np.ndarray) -> float:
        return 0.5 * self.inner(delta, delta)

    def gram(self, components: list[np.ndarray]) -> np.ndarray:
        stacked = torch.stack(
            [torch.from_numpy(np.asarray(component, dtype=np.float32)) for component in components],
            dim=0,
        ).to(self.device)
        projected = stacked @ self.covariance
        gram = torch.einsum("krc,jrc->kj", projected, stacked)
        return gram.detach().cpu().double().numpy()


def _encode_symmetric_block(
    work: np.ndarray,
    *,
    bits: int,
    quantizer: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a row-by-column block with FP16 stored scales."""

    qmax = max(1, 2 ** (int(bits) - 1) - 1)
    maximum = np.max(np.abs(work), axis=1)
    clip_ratios = (
        (1.0,)
        if quantizer == "symmetric_rtn"
        else (0.75, 0.875, 1.0)
        if quantizer == "symmetric_mse_clip"
        else ()
    )
    if not clip_ratios:
        raise ValueError(f"unsupported symmetric quantizer: {quantizer}")
    best_error = np.full(work.shape[0], np.inf, dtype=np.float64)
    best_scales = np.ones(work.shape[0], dtype=np.float16)
    best_codes = np.zeros(work.shape, dtype=np.int8)
    for ratio in clip_ratios:
        raw_scale = maximum * float(ratio) / float(qmax)
        raw_scale = np.where(raw_scale > 0.0, raw_scale, 1.0)
        stored_scale = np.asarray(raw_scale, dtype=np.float16)
        safe_scale = np.maximum(
            stored_scale.astype(np.float32), np.finfo(np.float16).tiny
        )
        codes = np.clip(
            np.rint(work / safe_scale[:, None]), -qmax, qmax
        ).astype(np.int8)
        decoded = codes.astype(np.float32) * safe_scale[:, None]
        error = np.mean((decoded - work) ** 2, axis=1, dtype=np.float64)
        improved = error < best_error
        if np.any(improved):
            best_error[improved] = error[improved]
            best_scales[improved] = stored_scale[improved]
            best_codes[improved] = codes[improved]
    return best_codes, best_scales


def encode_symmetric_quantizer(
    weight: np.ndarray,
    bits: int,
    *,
    col_block_size: int | None = None,
    quantizer: str = "symmetric_rtn",
) -> QuantCodec:
    """Encode row-wise or row-by-column-group symmetric integer weights."""

    work = np.asarray(weight, dtype=np.float32)
    if work.ndim != 2:
        raise ValueError("quantized weight must be a matrix")
    if col_block_size is None:
        codes, scales = _encode_symmetric_block(
            work,
            bits=bits,
            quantizer=quantizer,
        )
        return QuantCodec(
            codes=codes,
            scales=scales,
            bits=int(bits),
            quantizer=quantizer,
        )
    block_size = int(col_block_size)
    if block_size <= 0:
        raise ValueError("col_block_size must be positive")
    rows, cols = work.shape
    groups = (cols + block_size - 1) // block_size
    codes = np.empty((rows, cols), dtype=np.int8)
    scales = np.empty((rows, groups), dtype=np.float16)
    for group in range(groups):
        start = group * block_size
        stop = min((group + 1) * block_size, cols)
        group_codes, group_scales = _encode_symmetric_block(
            work[:, start:stop],
            bits=bits,
            quantizer=quantizer,
        )
        codes[:, start:stop] = group_codes
        scales[:, group] = group_scales
    return QuantCodec(
        codes=codes,
        scales=scales,
        bits=int(bits),
        col_block_size=block_size,
        quantizer=quantizer,
    )


def encode_row_rtn(weight: np.ndarray, bits: int) -> QuantCodec:
    return encode_symmetric_quantizer(weight, bits, quantizer="symmetric_rtn")


def build_quantizer_candidate_codecs(
    weight: np.ndarray,
    *,
    bit_widths: list[int],
    group_sizes: list[int],
    quantizers: list[str],
) -> list[QuantCodec]:
    """Build a deterministic heterogeneous base-quantizer candidate grid."""

    codecs: list[QuantCodec] = []
    seen: set[tuple[int, int, str]] = set()
    for bits in sorted(set(map(int, bit_widths))):
        validate_research_codec_bits(bits)
        for group_size in sorted(set(map(int, group_sizes))):
            if group_size < 0:
                raise ValueError("candidate Q group sizes must be non-negative")
            col_block_size = None if group_size == 0 else group_size
            for quantizer in sorted(set(map(str, quantizers))):
                key = (bits, group_size, quantizer)
                if key in seen:
                    continue
                seen.add(key)
                codecs.append(
                    encode_symmetric_quantizer(
                        weight,
                        bits,
                        col_block_size=col_block_size,
                        quantizer=quantizer,
                    )
                )
    return codecs


def screen_heterogeneous_candidate_families(
    *,
    layer: str,
    weight: np.ndarray,
    quantizer_codecs: list[QuantCodec],
    lowrank_factor_bits: list[int],
    default_q: QuantCodec,
    metric: HessianMetric,
    support_encoding: str,
    target_ratio: float,
    top_k: int,
) -> tuple[list[tuple[QuantCodec, int]], list[dict[str, object]]]:
    """Use a cheap Q-only proxy before expensive S/L family construction."""

    records: list[
        tuple[
            tuple[float, int, int, int, str],
            QuantCodec,
            int,
            dict[str, object],
        ]
    ] = []
    for q in quantizer_codecs:
        q_candidate = Candidate("Q", target_ratio, layer, weight, q)
        q_cost = metric.cost(q.decode() - weight)
        q_payload = q_candidate.payload(support_encoding=support_encoding)
        group_size = 0 if q.col_block_size is None else int(q.col_block_size)
        for factor_bits in lowrank_factor_bits:
            row = {
                "search_family": "heterogeneous_family_q_only_prescreen",
                "strategy": "global_candidate_family",
                "target_ratio": target_ratio,
                "layer": layer,
                "q_bits": q.bits,
                "q_quantizer": q.quantizer,
                "q_col_block_size": group_size,
                "lowrank_factor_bits": int(factor_bits),
                "q_only_hessian_cost": q_cost,
                "q_only_payload_bits": q_payload.total_bits,
                "q_only_payload_ratio": q_payload.ratio,
                "selected_for_expensive_family_expansion": False,
            }
            key = (
                q_cost,
                q_payload.total_bits,
                int(factor_bits),
                group_size,
                q.quantizer,
            )
            records.append((key, q, int(factor_bits), row))
    records.sort(key=lambda item: item[0])
    if int(top_k) <= 0:
        selected_indices = set(range(len(records)))
    else:
        selected_indices = set(range(min(int(top_k), len(records))))
        default_index = next(
            index
            for index, (_key, q, factor_bits, _row) in enumerate(records)
            if q is default_q and factor_bits == 16
        )
        selected_indices.add(default_index)
        for factor_bits in sorted(set(map(int, lowrank_factor_bits))):
            selected_indices.add(
                next(
                    index
                    for index, (_key, _q, bits, _row) in enumerate(records)
                    if bits == factor_bits
                )
            )
    selected: list[tuple[QuantCodec, int]] = []
    rows: list[dict[str, object]] = []
    for proxy_rank, (_key, q, factor_bits, row) in enumerate(records, start=1):
        chosen = proxy_rank - 1 in selected_indices
        row["q_only_proxy_rank"] = proxy_rank
        row["selected_for_expensive_family_expansion"] = chosen
        row["forced_default_family"] = q is default_q and factor_bits == 16
        if chosen:
            selected.append((q, factor_bits))
        rows.append(row)
    return selected, rows


def quantize_lowrank_factors(
    source: LowRankCodec | None,
    *,
    bits: int,
    quantizer: str = "symmetric_mse_clip",
) -> LowRankCodec | None:
    """Store both low-rank factors with packed integer codes and FP16 row scales."""

    if source is None:
        return None
    factor_bits = int(bits)
    if factor_bits == 16:
        if source.factor_bits == 16:
            return source
        return LowRankCodec(
            np.asarray(source._decode_factor(source.left, source.left_scales), dtype=np.float16),
            np.asarray(source._decode_factor(source.right, source.right_scales), dtype=np.float16),
        )
    validate_research_codec_bits(factor_bits)
    left_values = source._decode_factor(source.left, source.left_scales)
    right_values = source._decode_factor(source.right, source.right_scales)
    left = encode_symmetric_quantizer(
        left_values,
        factor_bits,
        quantizer=quantizer,
    )
    right = encode_symmetric_quantizer(
        right_values,
        factor_bits,
        quantizer=quantizer,
    )
    return LowRankCodec(
        left=left.codes,
        right=right.codes,
        factor_bits=factor_bits,
        left_scales=left.scales,
        right_scales=right.scales,
        quantizer=quantizer,
    )


def sparse_codec_from_residual(
    residual: np.ndarray,
    covariance: torch.Tensor,
    nonzero: int,
    *,
    method: str,
    torch_dtype: torch.dtype,
) -> SparseCodec | None:
    nonzero = max(0, min(int(nonzero), int(residual.size)))
    if nonzero <= 0:
        return None
    tensor = torch.from_numpy(np.asarray(residual, dtype=np.float32)).to(dtype=torch_dtype)
    keep_fraction = float(nonzero) / float(max(tensor.numel(), 1))
    projected = base.residual_sparse_project(tensor, covariance, keep_fraction, method)
    values = projected.detach().cpu().float().numpy()
    # top-k in the reused projector produces exactly floor(fraction * N) slots.
    mask = values != 0.0
    if int(np.count_nonzero(mask)) != nonzero:
        scores = np.abs(residual)
        if method == "wanda":
            diag = np.sqrt(np.maximum(np.diag(covariance.detach().cpu().numpy()), 0.0))
            scores = scores * diag.reshape(1, -1)
        indices = np.argpartition(scores.reshape(-1), -nonzero)[-nonzero:]
        mask = np.zeros(residual.size, dtype=bool)
        mask[indices] = True
        mask = mask.reshape(residual.shape)
        values = np.where(mask, residual, 0.0)
    stored = np.asarray(np.where(mask, values, 0.0), dtype=np.float16)
    return SparseCodec(values=stored, mask=mask)


def obs_refit_sparse(
    residual: np.ndarray,
    sparse: SparseCodec | None,
    covariance: PreparedInputCovariance,
    *,
    metric: HessianMetric,
    rcond: float,
) -> tuple[SparseCodec | None, dict[str, object]]:
    if sparse is None or sparse.nonzero_count <= 0:
        return None, {
            "obs_applied": False,
            "obs_reason": "empty_support",
            "obs_relative_stationarity_continuous": float("nan"),
            "obs_relative_stationarity_stored": float("nan"),
        }
    result = obs_retained_support_correction(
        residual,
        sparse.mask,
        covariance,
        rcond=float(rcond),
    )
    stored_values = np.asarray(np.where(sparse.mask, result.corrected_weight, 0.0), dtype=np.float16)
    stored = SparseCodec(stored_values, sparse.mask)
    stored_error = stored.decode() - residual
    gradient = stored_error.astype(np.float64) @ covariance.matrix
    retained_gradient = np.abs(gradient[sparse.mask])
    stored_stationarity = float(np.max(retained_gradient, initial=0.0) / max(float(np.linalg.norm(gradient)), EPS))
    continuous_cost = float(result.corrected_cost)
    stored_cost = metric.cost(stored_error)
    return stored, {
        "obs_applied": True,
        "obs_unique_support_count": result.unique_support_count,
        "obs_relative_stationarity_continuous": result.relative_stationarity,
        "obs_relative_stationarity_stored": stored_stationarity,
        "obs_continuous_cost": continuous_cost,
        "obs_stored_cost": stored_cost,
        "obs_fp16_rounding_cost_gap": stored_cost - continuous_cost,
        "obs_rhs_null_residual_max": result.rhs_null_residual_max,
    }


def _hessian_inner_np(left: np.ndarray, right: np.ndarray, covariance: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(np.sum((a @ covariance) * b))


def _hessian_cost_np(delta: np.ndarray, covariance: np.ndarray) -> float:
    return 0.5 * _hessian_inner_np(delta, delta, covariance)


def _hessian_rho(inner: float, self_left: float, self_right: float) -> float:
    denom = math.sqrt(max(2.0 * self_left, 0.0) * max(2.0 * self_right, 0.0))
    if denom <= EPS:
        return float("nan")
    return float(max(-1.0, min(1.0, inner / denom)))


def _rho_kind(value: float, threshold: float) -> str:
    if not math.isfinite(value):
        return "inactive"
    if abs(value) <= float(threshold):
        return "hessian_orthogonal"
    if value < 0.0:
        return "repair_cancellation"
    return "positive_conflict"


def _payload_for_counts(
    shape: tuple[int, int],
    *,
    bits: int,
    scale_count: int,
    sparse_nonzero: int,
    rank: int,
    support_encoding: str,
    lowrank_factor_bits: int = 16,
) -> PayloadBreakdown:
    factor_bits = int(lowrank_factor_bits)
    lowrank_scale_count = (
        int(shape[0]) + int(rank)
        if int(rank) > 0 and factor_bits != 16
        else 0
    )
    return exact_payload_accounting(
        shape,
        base_code_bits=int(bits),
        base_scale_count=int(scale_count),
        sparse_nonzero=int(sparse_nonzero),
        sparse_value_bits=16,
        support_encoding=support_encoding,
        lowrank_rank=int(rank),
        lowrank_factor_bits=factor_bits,
        lowrank_scale_count=lowrank_scale_count,
    )


def max_sparse_under_target(
    shape: tuple[int, int],
    *,
    bits: int,
    scale_count: int,
    rank: int,
    target_ratio: float,
    support_encoding: str,
    lowrank_factor_bits: int = 16,
) -> int:
    reference = shape[0] * shape[1] * 16
    budget = int(math.floor(float(target_ratio) * reference + 1e-9))
    return max_sparse_under_budget_bits(
        shape,
        bits=bits,
        scale_count=scale_count,
        rank=rank,
        budget_bits=budget,
        support_encoding=support_encoding,
        lowrank_factor_bits=lowrank_factor_bits,
    )


def max_sparse_under_budget_bits(
    shape: tuple[int, int],
    *,
    bits: int,
    scale_count: int,
    rank: int,
    budget_bits: int,
    support_encoding: str,
    lowrank_factor_bits: int = 16,
) -> int:
    if budget_bits < 0:
        raise ValueError("budget_bits must be non-negative")
    low, high = 0, shape[0] * shape[1]
    while low < high:
        middle = (low + high + 1) // 2
        payload = _payload_for_counts(
            shape,
            bits=bits,
            scale_count=scale_count,
            sparse_nonzero=middle,
            rank=rank,
            support_encoding=support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        if payload.total_bits <= int(budget_bits):
            low = middle
        else:
            high = middle - 1
    return int(low)


def max_rank_under_target(
    shape: tuple[int, int],
    *,
    bits: int,
    scale_count: int,
    sparse_nonzero: int,
    target_ratio: float,
    support_encoding: str,
    lowrank_factor_bits: int = 16,
) -> int:
    reference = shape[0] * shape[1] * 16
    budget = int(math.floor(float(target_ratio) * reference + 1e-9))
    low, high = 0, min(shape)
    while low < high:
        middle = (low + high + 1) // 2
        payload = _payload_for_counts(
            shape,
            bits=bits,
            scale_count=scale_count,
            sparse_nonzero=sparse_nonzero,
            rank=middle,
            support_encoding=support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        if payload.total_bits <= budget:
            low = middle
        else:
            high = middle - 1
    return int(low)


def max_rank_under_budget_bits(
    shape: tuple[int, int],
    *,
    bits: int,
    scale_count: int,
    sparse_nonzero: int,
    budget_bits: int,
    support_encoding: str,
    lowrank_factor_bits: int = 16,
) -> int:
    """Return the largest realizable rank below an explicit logical-bit cap."""

    if budget_bits < 0:
        raise ValueError("budget_bits must be non-negative")
    low, high = 0, min(shape)
    while low < high:
        middle = (low + high + 1) // 2
        payload = _payload_for_counts(
            shape,
            bits=bits,
            scale_count=scale_count,
            sparse_nonzero=sparse_nonzero,
            rank=middle,
            support_encoding=support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        if payload.total_bits <= int(budget_bits):
            low = middle
        else:
            high = middle - 1
    return int(low)


def candidate_geometry(
    candidate: Candidate,
    metric: HessianMetric,
    activation_samples: torch.Tensor | None,
    *,
    support_encoding: str,
    rho_threshold: float,
    q_reference_cost: float,
    q_reference_bits: int,
) -> dict[str, object]:
    final = candidate.final
    sparse = candidate.sparse_decoded
    lowrank = candidate.lowrank_decoded
    # Absorb the final FP16 accumulation residual into the Q bucket so that the
    # reported decomposition is exactly equal to the deployed perturbation.
    q_effective = final - sparse - lowrank
    q_delta = q_effective - candidate.weight
    total_delta = final - candidate.weight
    accumulation_rounding = final - candidate.pre_runtime_sum

    gram = metric.gram([q_delta, sparse, lowrank])
    self_q = 0.5 * float(gram[0, 0])
    self_s = 0.5 * float(gram[1, 1])
    self_l = 0.5 * float(gram[2, 2])
    cross_qs = float(gram[0, 1]) if candidate.sparse is not None else 0.0
    cross_ql = float(gram[0, 2]) if candidate.lowrank is not None else 0.0
    cross_sl = float(gram[1, 2]) if candidate.sparse is not None and candidate.lowrank is not None else 0.0
    decomposed = self_q + self_s + self_l + cross_qs + cross_ql + cross_sl
    total_cost = metric.cost(total_delta)
    rho_qs = _hessian_rho(cross_qs, self_q, self_s)
    rho_ql = _hessian_rho(cross_ql, self_q, self_l)
    rho_sl = _hessian_rho(cross_sl, self_s, self_l)
    cancellation_gain_qs = -cross_qs / self_q if candidate.sparse is not None and self_q > EPS else float("nan")
    cancellation_gain_ql = -cross_ql / self_q if candidate.lowrank is not None and self_q > EPS else float("nan")
    baseline_energy = max(metric.inner(candidate.weight, candidate.weight), EPS)

    if activation_samples is None or activation_samples.numel() == 0:
        activation_error = 2.0 * total_cost / baseline_energy
        worst_token = activation_error
        p95 = activation_error
    else:
        samples = activation_samples.detach().cpu().float().numpy()
        err = samples @ total_delta.astype(np.float32).T
        ref = samples @ candidate.weight.astype(np.float32).T
        token_err = np.sum(err * err, axis=1)
        token_ref = np.maximum(np.sum(ref * ref, axis=1), EPS)
        ratios = token_err / token_ref
        activation_error = float(np.sum(token_err) / max(float(np.sum(token_ref)), EPS))
        worst_token = float(np.max(ratios, initial=0.0))
        p95 = float(np.quantile(ratios, 0.95)) if ratios.size else 0.0

    payload = candidate.payload(support_encoding=support_encoding)
    fields = payload.as_dict()
    added_bits = int(payload.total_bits - q_reference_bits)
    hessian_gain = float(q_reference_cost - total_cost)
    added_scalars = max(candidate.q.scale_count - candidate.weight.shape[0], 0)
    added_scalars += candidate.sparse_nnz + candidate.rank * sum(candidate.weight.shape)
    return {
        "strategy": candidate.strategy,
        "target_ratio": candidate.target_ratio,
        "layer": candidate.layer,
        "rows": candidate.weight.shape[0],
        "cols": candidate.weight.shape[1],
        **fields,
        "target_gap": payload.ratio - candidate.target_ratio,
        "under_target": payload.ratio <= candidate.target_ratio + 1e-12,
        "sparse_nnz": candidate.sparse_nnz,
        "sparse_fraction": candidate.sparse_nnz / float(candidate.weight.size),
        "lowrank_rank": candidate.rank,
        "lowrank_factor_bits": (
            0 if candidate.lowrank is None else candidate.lowrank.factor_bits
        ),
        "lowrank_factor_scale_count": (
            0 if candidate.lowrank is None else candidate.lowrank.scale_count
        ),
        "lowrank_factor_quantizer": (
            "inactive" if candidate.lowrank is None else candidate.lowrank.quantizer
        ),
        "q_bits": candidate.q.bits,
        "q_quantizer": candidate.q.quantizer,
        "q_scale_count": candidate.q.scale_count,
        "q_col_block_size": candidate.q.col_block_size or 0,
        "q_active": True,
        "s_active": candidate.sparse_nnz > 0,
        "l_active": candidate.rank > 0,
        "all_requested_components_active": (
            ("S" not in candidate.strategy or candidate.sparse_nnz > 0)
            and ("L" not in candidate.strategy or candidate.rank > 0)
        ),
        "hessian_cost": total_cost,
        "normalized_hessian_cost": 2.0 * total_cost / baseline_energy,
        "baseline_hessian_energy": baseline_energy,
        "hessian_self_q": self_q,
        "hessian_self_s": self_s,
        "hessian_self_l": self_l,
        "hessian_cross_qs": cross_qs,
        "hessian_cross_ql": cross_ql,
        "hessian_cross_sl": cross_sl,
        "hessian_decomposition_residual": total_cost - decomposed,
        "rho_qs": rho_qs,
        "rho_ql": rho_ql,
        "rho_sl": rho_sl,
        "cancellation_gain_qs_over_q": cancellation_gain_qs,
        "cancellation_gain_ql_over_q": cancellation_gain_ql,
        "rho_qs_kind": _rho_kind(rho_qs, rho_threshold),
        "rho_ql_kind": _rho_kind(rho_ql, rho_threshold),
        "rho_sl_kind": _rho_kind(rho_sl, rho_threshold),
        "activation_reconstruction_error": activation_error,
        "worst_token_risk": worst_token,
        "token_risk_p95": p95,
        "fp16_accumulation_rounding_l2": float(np.linalg.norm(accumulation_rounding)),
        "q_reference_cost": q_reference_cost,
        "q_reference_bits": q_reference_bits,
        "added_bits_over_q": added_bits,
        "added_scalars_over_q": added_scalars,
        "hessian_gain_over_q": hessian_gain,
        "hessian_gain_per_added_bit": hessian_gain / added_bits if added_bits > 0 else float("nan"),
        "folded_zero_bit_hessian_gain": hessian_gain if added_bits <= 0 and candidate.repair_dof > 0 else 0.0,
        "hessian_gain_per_added_scalar": hessian_gain / added_scalars if added_scalars > 0 else float("nan"),
        "hessian_recovery_fraction_over_q": hessian_gain / max(q_reference_cost, EPS),
        "folded_repair_dof": candidate.repair_dof,
        **candidate.diagnostics,
    }


def make_global_scaled_q(
    layer: str,
    target: float,
    weight: np.ndarray,
    q: QuantCodec,
    covariance: PreparedInputCovariance,
    bounds: tuple[float, float],
) -> Candidate:
    q_decoded = q.decode()
    result = hessian_basis_repair(q_decoded - weight, np.stack([q_decoded]), covariance)
    continuous_multiplier = float(1.0 + result.coefficients[0])
    multiplier = float(np.clip(continuous_multiplier, bounds[0], bounds[1]))
    repaired_q = q.scaled(multiplier)
    return Candidate(
        "Q_global_scale",
        target,
        layer,
        weight,
        repaired_q,
        diagnostics={
            "scale_repair_kind": "global_folded_into_fp16_q_scales",
            "scale_multiplier_continuous": continuous_multiplier,
            "scale_multiplier_applied": multiplier,
            "scale_cost_before_continuous": result.cost_before,
            "scale_cost_after_continuous": result.cost_after,
            "scale_gram_rank": result.gram_rank,
            "scale_max_basis_stationarity": result.max_basis_stationarity,
        },
        repair_dof=1,
    )


def make_block_scaled_q_candidates(
    layer: str,
    weight: np.ndarray,
    q: QuantCodec,
    covariance: PreparedInputCovariance,
    block_sizes: list[int],
    bounds: tuple[float, float],
) -> list[Candidate]:
    decoded = q.decode()
    out: list[Candidate] = []
    for block_size in sorted(set(block_sizes)):
        result = hessian_row_block_scale_repair(
            weight,
            decoded,
            covariance,
            col_block_size=int(block_size),
            scale_bounds=bounds,
            storage_dtype=np.float16,
        )
        groups = result.scales.shape[1]
        combined_scales = np.asarray(q.scales.astype(np.float32)[:, None] * result.scales, dtype=np.float16)
        block_codec = QuantCodec(
            q.codes,
            combined_scales,
            q.bits,
            int(block_size),
            q.quantizer,
        )
        out.append(
            Candidate(
                "Q_block_scale",
                0.0,
                layer,
                weight,
                block_codec,
                diagnostics={
                    "scale_repair_kind": "row_x_column_block_fp16_scales",
                    "scale_col_block_size": block_size,
                    "scale_groups_per_row": groups,
                    "scale_cost_before_continuous": result.cost_before,
                    "scale_cost_after_continuous": result.cost_after,
                    "scale_max_basis_stationarity": result.max_relative_stationarity,
                },
                repair_dof=result.stored_scale_count,
            )
        )
    return out


def make_component_scaled_candidate(
    source: Candidate,
    covariance: PreparedInputCovariance,
    bounds: tuple[float, float],
    *,
    strategy: str = "Q+S+L_component_scale",
) -> Candidate:
    components: list[np.ndarray] = [source.q_decoded]
    labels = ["q"]
    if source.sparse is not None and source.sparse_nnz > 0:
        components.append(source.sparse_decoded)
        labels.append("s")
    if source.lowrank is not None and source.rank > 0:
        components.append(source.lowrank_decoded)
        labels.append("l")
    result = hessian_basis_repair(source.final - source.weight, np.stack(components), covariance)
    multipliers = np.clip(1.0 + result.coefficients, bounds[0], bounds[1])
    mapping = {label: float(multiplier) for label, multiplier in zip(labels, multipliers)}
    q = source.q.scaled(mapping["q"])
    sparse = source.sparse.scaled(mapping.get("s", 1.0)) if source.sparse is not None else None
    lowrank = source.lowrank.scaled(mapping.get("l", 1.0)) if source.lowrank is not None else None
    return Candidate(
        strategy,
        source.target_ratio,
        source.layer,
        source.weight,
        q,
        sparse,
        lowrank,
        diagnostics={
            **source.diagnostics,
            "scale_repair_kind": "component_multipliers_folded_into_existing_fp16_storage",
            "component_scale_labels": "+".join(labels),
            "component_scale_multipliers": json.dumps(mapping, sort_keys=True),
            "scale_cost_before_continuous": result.cost_before,
            "scale_cost_after_continuous": result.cost_after,
            "scale_gram_rank": result.gram_rank,
            "scale_gram_condition": result.gram_condition,
            "scale_max_basis_stationarity": result.max_basis_stationarity,
        },
        repair_dof=len(labels),
    )


def _candidate_fast_cost(candidate: Candidate, metric: HessianMetric) -> float:
    return metric.cost(candidate.final - candidate.weight)


def parse_allocation_rank_grid(value: str | None) -> list[int]:
    """Parse the optional grid with the base runner's one-argument helper."""

    return sorted(set(base.parse_int_csv(value)))


def resolve_allocation_rank_grid(
    max_rank: int,
    *,
    max_dense_rank: int,
    configured_grid: Iterable[int] = (),
) -> list[int]:
    """Return the disclosed rank states used by QSL allocation searches.

    The legacy behavior densely enumerates ``0..max_dense_rank``.  A supplied
    non-uniform grid allows large matrices to reach their true rate-feasible
    endpoint without paying for every intermediate SVD/refit.  Rank zero and
    the maximum feasible rank are always retained as auditable endpoints.
    """

    feasible = int(max_rank)
    dense_limit = int(max_dense_rank)
    if feasible < 0 or dense_limit < 0:
        raise ValueError("allocation rank limits must be non-negative")
    configured = [int(rank) for rank in configured_grid]
    if any(rank < 0 for rank in configured):
        raise ValueError("allocation rank grid must be non-negative")
    if configured:
        ranks = {0, feasible}
        ranks.update(rank for rank in configured if rank <= feasible)
        return sorted(ranks)
    return list(range(0, min(feasible, dense_limit) + 1))


def _candidate_with_strategy(
    source: Candidate,
    strategy: str,
    *,
    diagnostics: dict[str, object] | None = None,
) -> Candidate:
    """Clone an endpoint without changing any serialized component."""

    return Candidate(
        strategy,
        source.target_ratio,
        source.layer,
        source.weight,
        source.q,
        source.sparse,
        source.lowrank,
        diagnostics={**source.diagnostics, **(diagnostics or {})},
        repair_dof=source.repair_dof,
    )


def build_global_single_component_option_pools(
    *,
    layer: str,
    weight_tensor: torch.Tensor,
    covariance_tensor: torch.Tensor,
    covariance: PreparedInputCovariance,
    factorizer: LowRankFactorizer,
    metric: HessianMetric,
    target_ratio: float,
    q_candidate: Candidate,
    ql_candidate: Candidate,
    qs_obs_candidate: Candidate,
    allocation_ranks: list[int],
    lowrank_factor_bits: int,
    args: argparse.Namespace,
) -> dict[str, list[Candidate]]:
    """Enumerate pure-L, pure-S/OBS and their no-joint union control.

    Both controls use the same local Q+L repair allowance, support fractions,
    budget multipliers and configured rank-grid envelope as the QSL search.  A
    Pure-L refits the unchanged Q residual independently at each enumerated
    rank, while the original local maximum-rank Q+L factor is reused exactly.
    Thus a one-layer cap cannot improve merely by changing the SVD call used at
    the same rank; multi-layer gains isolate the enumerated rank allocation.
    """

    weight = weight_tensor.detach().cpu().float().numpy()
    residual_q = weight - q_candidate.q.decode()
    shape = tuple(map(int, weight.shape))
    q_bits = q_candidate.payload(support_encoding=args.support_encoding).total_bits
    ql_bits = ql_candidate.payload(support_encoding=args.support_encoding).total_bits
    repair_budget_bits = max(0, int(ql_bits) - int(q_bits))
    band_multipliers = [1.0, *map(float, args.global_frontier_budget_multipliers)]
    band_budgets = {
        multiplier: int(q_bits)
        + int(math.floor(multiplier * repair_budget_bits))
        for multiplier in band_multipliers
    }
    maximum_band_budget = max(band_budgets.values(), default=int(ql_bits))

    band_max_ranks = {
        multiplier: max_rank_under_budget_bits(
            shape,
            bits=q_candidate.q.bits,
            scale_count=q_candidate.q.scale_count,
            sparse_nonzero=0,
            budget_bits=budget,
            support_encoding=args.support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        for multiplier, budget in band_budgets.items()
    }
    maximum_rank = max(band_max_ranks.values(), default=0)
    pure_l_ranks = set(allocation_ranks)
    pure_l_ranks.update(band_max_ranks.values())
    pure_l_ranks.update(
        resolve_allocation_rank_grid(
            maximum_rank,
            max_dense_rank=int(args.max_allocation_ranks),
            configured_grid=getattr(args, "allocation_rank_grid", ()),
        )
    )
    pure_l_ranks = {rank for rank in pure_l_ranks if 0 <= rank <= maximum_rank}
    ql_options = [
        _candidate_with_strategy(
            q_candidate,
            "Q+L_global",
            diagnostics={
                "global_single_component_control": "lowrank",
                "global_control_allocation_state": "nested_Q",
            },
        ),
        _candidate_with_strategy(
            ql_candidate,
            "Q+L_global",
            diagnostics={
                "global_single_component_control": "lowrank",
                "global_control_allocation_state": "original_local_Q+L",
            },
        ),
    ]
    for rank in sorted(pure_l_ranks):
        factor = (
            ql_candidate.lowrank
            if rank == ql_candidate.rank
            else quantize_lowrank_factors(
                factorizer.factorize(residual_q, rank),
                bits=lowrank_factor_bits,
            )
        )
        if rank > 0 and (factor is None or factor.rank != rank):
            raise RuntimeError(f"pure-L global control failed to fit rank {rank}")
        candidate = Candidate(
            "Q+L_global",
            target_ratio,
            layer,
            weight,
            q_candidate.q,
            lowrank=factor,
            diagnostics={
                "global_single_component_control": "lowrank",
                "global_control_allocation_state": "independent_rank_family",
                "global_control_family_max_rank": maximum_rank,
                "global_control_rank": rank,
                "global_control_budget_multipliers": json.dumps(band_multipliers),
                "global_control_decomposition_policy": (
                    "independent_same_Q_residual_factorization_per_enumerated_rank; "
                    "original_local_Q+L_factor_reused_at_its_rank"
                ),
            },
        )
        if candidate.payload(support_encoding=args.support_encoding).total_bits > maximum_band_budget:
            raise AssertionError("pure-L global candidate exceeded its largest logical budget band")
        ql_options.append(candidate)

    band_max_supports = {
        multiplier: max_sparse_under_budget_bits(
            shape,
            bits=q_candidate.q.bits,
            scale_count=q_candidate.q.scale_count,
            rank=0,
            budget_bits=budget,
            support_encoding=args.support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        for multiplier, budget in band_budgets.items()
    }
    local_max_support = band_max_supports[1.0]
    support_counts = {0, qs_obs_candidate.sparse_nnz, *band_max_supports.values()}
    support_counts.update(
        int(math.floor(float(fraction) * maximum))
        for maximum in band_max_supports.values()
        for fraction in args.global_frontier_support_fractions
    )
    support_counts = {
        nonzero for nonzero in support_counts if 0 <= nonzero <= shape[0] * shape[1]
    }
    qs_obs_options = [
        _candidate_with_strategy(
            q_candidate,
            "Q+S_OBS_global",
            diagnostics={
                "global_single_component_control": "sparse_obs",
                "global_control_allocation_state": "nested_Q",
                "strict_sparse_refit": "not_applicable",
            },
        ),
        _candidate_with_strategy(
            qs_obs_candidate,
            "Q+S_OBS_global",
            diagnostics={
                "global_single_component_control": "sparse_obs",
                "global_control_allocation_state": "original_local_Q+S_OBS",
                "strict_sparse_refit": "obs",
            },
        ),
    ]
    for nonzero in sorted(support_counts):
        selected_sparse = sparse_codec_from_residual(
            residual_q,
            covariance_tensor,
            nonzero,
            method=args.s_method,
            torch_dtype=weight_tensor.dtype,
        )
        sparse_obs, obs_diagnostics = obs_refit_sparse(
            residual_q,
            selected_sparse,
            covariance,
            metric=metric,
            rcond=args.obs_rcond,
        )
        if nonzero > 0 and not bool(obs_diagnostics.get("obs_applied", False)):
            raise RuntimeError("pure-S global control could not apply its required OBS refit")
        candidate = Candidate(
            "Q+S_OBS_global",
            target_ratio,
            layer,
            weight,
            q_candidate.q,
            sparse=sparse_obs,
            diagnostics={
                "global_single_component_control": "sparse_obs",
                "global_control_allocation_state": "enumerated_support_family",
                "global_control_support_nnz": nonzero,
                "global_control_local_max_support_nnz": local_max_support,
                "global_control_budget_multipliers": json.dumps(band_multipliers),
                "strict_sparse_refit": "obs" if nonzero > 0 else "not_applicable",
                **obs_diagnostics,
            },
            repair_dof=0 if sparse_obs is None else sparse_obs.nonzero_count,
        )
        if candidate.payload(support_encoding=args.support_encoding).total_bits > maximum_band_budget:
            raise AssertionError("pure-S global candidate exceeded its largest logical budget band")
        qs_obs_options.append(candidate)

    # The union control is the clean counterfactual for the joint QSL pool: it
    # may choose Q, pure S/OBS or pure L independently at every layer, but it
    # can never activate S and L together in one layer.  Every option is
    # relabelled while preserving its exact payload and fitted values.
    union_options = [
        _candidate_with_strategy(
            candidate,
            GLOBAL_NONJOINT_CONTROL_STRATEGY,
            diagnostics={
                "global_nonjoint_source_family": candidate.strategy,
                "global_nonjoint_control": True,
            },
        )
        for candidate in (*qs_obs_options, *ql_options)
    ]
    return {
        "Q+L_global": ql_options,
        "Q+S_OBS_global": qs_obs_options,
        GLOBAL_NONJOINT_CONTROL_STRATEGY: union_options,
    }


def build_layer_candidates(
    *,
    layer: str,
    weight_tensor: torch.Tensor,
    covariance_tensor: torch.Tensor,
    prepared_covariance: PreparedInputCovariance,
    activation_samples: torch.Tensor | None,
    factorizer: LowRankFactorizer,
    metric: HessianMetric,
    target_ratio: float,
    q: QuantCodec,
    global_q: Candidate,
    block_q_options: list[Candidate],
    lowrank_factor_bits: int,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Candidate],
    list[dict[str, object]],
    dict[str, list[Candidate]],
]:
    weight = weight_tensor.detach().cpu().float().numpy()
    covariance = prepared_covariance
    shape = tuple(map(int, weight.shape))
    q_candidate = Candidate("Q", target_ratio, layer, weight, q)
    q_payload = q_candidate.payload(support_encoding=args.support_encoding)
    q_cost = _candidate_fast_cost(q_candidate, metric)
    selected: dict[str, Candidate] = {"Q": q_candidate}
    search_rows: list[dict[str, object]] = []

    scaled = Candidate(
        global_q.strategy,
        target_ratio,
        layer,
        weight,
        global_q.q,
        diagnostics=dict(global_q.diagnostics),
        repair_dof=global_q.repair_dof,
    )
    selected[scaled.strategy] = scaled

    best_block: Candidate | None = None
    if not bool(getattr(args, "skip_block_scale", False)):
        feasible_blocks: list[Candidate] = []
        for option in block_q_options:
            candidate = Candidate(
                option.strategy,
                target_ratio,
                layer,
                weight,
                option.q,
                diagnostics=dict(option.diagnostics),
                repair_dof=option.repair_dof,
            )
            payload = candidate.payload(support_encoding=args.support_encoding)
            cost = _candidate_fast_cost(candidate, metric)
            search_rows.append(
                {
                    "search_family": "Q_block_scale",
                    "strategy": candidate.strategy,
                    "target_ratio": target_ratio,
                    "layer": layer,
                    "allocation_id": f"block={candidate.q.col_block_size}",
                    "payload_ratio": payload.ratio,
                    "payload_bits": payload.total_bits,
                    "under_target": payload.ratio <= target_ratio + 1e-12,
                    "hessian_cost": cost,
                    "selected_within_layer_target": False,
                }
            )
            if payload.ratio <= target_ratio + 1e-12:
                feasible_blocks.append(candidate)
        if feasible_blocks:
            best_block = min(feasible_blocks, key=lambda item: _candidate_fast_cost(item, metric))
        else:
            best_block = Candidate(
                "Q_block_scale",
                target_ratio,
                layer,
                weight,
                scaled.q,
                diagnostics={
                    **scaled.diagnostics,
                    "allocation_fallback": "global_scale_no_feasible_block_fit",
                },
                repair_dof=scaled.repair_dof,
            )
        selected["Q_block_scale"] = best_block

    # Q+S and its frozen-support OBS refit use the maximum realizable CSR
    # survivor count below the requested payload.
    max_sparse = max_sparse_under_target(
        shape,
        bits=q.bits,
        scale_count=q.scale_count,
        rank=0,
        target_ratio=target_ratio,
        support_encoding=args.support_encoding,
        lowrank_factor_bits=lowrank_factor_bits,
    )
    residual_q = weight - q.decode()
    sparse = sparse_codec_from_residual(
        residual_q,
        covariance_tensor,
        max_sparse,
        method=args.s_method,
        torch_dtype=weight_tensor.dtype,
    )
    qs = Candidate("Q+S", target_ratio, layer, weight, q, sparse)
    selected["Q+S"] = qs
    sparse_obs, obs_diag = obs_refit_sparse(
        residual_q,
        sparse,
        covariance,
        metric=metric,
        rcond=args.obs_rcond,
    )
    qs_obs = Candidate(
        "Q+S_OBS",
        target_ratio,
        layer,
        weight,
        q,
        sparse_obs,
        diagnostics=obs_diag,
        repair_dof=0 if sparse_obs is None else sparse_obs.nonzero_count,
    )
    selected["Q+S_OBS"] = qs_obs

    max_rank = max_rank_under_target(
        shape,
        bits=q.bits,
        scale_count=q.scale_count,
        sparse_nonzero=0,
        target_ratio=target_ratio,
        support_encoding=args.support_encoding,
        lowrank_factor_bits=lowrank_factor_bits,
    )
    ql_factor = quantize_lowrank_factors(
        factorizer.factorize(residual_q, max_rank),
        bits=lowrank_factor_bits,
    )
    ql = Candidate("Q+L", target_ratio, layer, weight, q, lowrank=ql_factor)
    selected["Q+L"] = ql

    # Enumerate the exact discrete rank allocation. For each rank, all remaining
    # realizable bits are assigned to CSR sparse survivors. A locally inactive
    # S or L is allowed and explicitly flagged; this exposes discrete metadata
    # barriers rather than silently using a nominal rate.
    allocation_ranks = resolve_allocation_rank_grid(
        max_rank,
        max_dense_rank=int(args.max_allocation_ranks),
        configured_grid=getattr(args, "allocation_rank_grid", ()),
    )
    qsl_options: list[Candidate] = []
    qsl_row_indices: list[int] = []
    for rank in allocation_ranks:
        nonzero = max_sparse_under_target(
            shape,
            bits=q.bits,
            scale_count=q.scale_count,
            rank=rank,
            target_ratio=target_ratio,
            support_encoding=args.support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        sparse_part, factor, fit_diagnostics = fit_sparse_lowrank_components(
            residual_q=residual_q,
            covariance_tensor=covariance_tensor,
            factorizer=factorizer,
            nonzero=nonzero,
            rank=rank,
            sparse_method=args.s_method,
            residual_order=args.residual_order,
            torch_dtype=weight_tensor.dtype,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        candidate = Candidate(
            "Q+S+L",
            target_ratio,
            layer,
            weight,
            q,
            sparse_part,
            factor,
            diagnostics={"residual_order": args.residual_order, **fit_diagnostics},
        )
        payload = candidate.payload(support_encoding=args.support_encoding)
        cost = _candidate_fast_cost(candidate, metric)
        row_index = len(search_rows)
        search_rows.append(
            {
                "search_family": "Q+S+L_exact_allocation",
                "strategy": candidate.strategy,
                "target_ratio": target_ratio,
                "layer": layer,
                "allocation_id": f"rank={rank},nnz={candidate.sparse_nnz}",
                "payload_ratio": payload.ratio,
                "payload_bits": payload.total_bits,
                "under_target": payload.ratio <= target_ratio + 1e-12,
                "sparse_nnz": candidate.sparse_nnz,
                "lowrank_rank": candidate.rank,
                "both_s_and_l_active": candidate.sparse_nnz > 0 and candidate.rank > 0,
                "hessian_cost": cost,
                "selected_within_layer_target": False,
            }
        )
        qsl_options.append(candidate)
        qsl_row_indices.append(row_index)
    jointly_active = [
        index
        for index, candidate in enumerate(qsl_options)
        if candidate.sparse_nnz > 0 and candidate.rank > 0
    ]
    eligible_indices = jointly_active or list(range(len(qsl_options)))
    best_qsl_index = min(eligible_indices, key=lambda index: _candidate_fast_cost(qsl_options[index], metric))
    best_qsl = qsl_options[best_qsl_index]
    best_qsl.diagnostics["joint_component_constraint"] = (
        "both_S_and_L_required" if jointly_active else "discrete_budget_prevented_joint_S_and_L"
    )
    search_rows[qsl_row_indices[best_qsl_index]]["selected_within_layer_target"] = True
    selected["Q+S+L"] = best_qsl

    # A stricter answer to the fixed-rate question: cap every layer's combined
    # codec by the exact number of bits used by that layer's Q+L endpoint.  The
    # aggregate combination can therefore never buy its gain with more storage
    # than the strongest single-repair comparator.  We re-enumerate rank and
    # CSR survivors under this discrete cap instead of rescaling nominal rates.
    # When requested, the cap includes the real codec header, descriptors and
    # the requested stream alignment, not only the declared value streams.
    ql_budget_bits = ql.payload(support_encoding=args.support_encoding).total_bits
    enforce_serialized_cap = bool(getattr(args, "enforce_serialized_rate_cap", False))
    artifact_alignment = int(getattr(args, "artifact_alignment", 64))
    ql_budget_file_bytes = (
        codec_artifact_natural_file_bytes([_artifact_layer(ql)], alignment=artifact_alignment)
        if enforce_serialized_cap
        else 0
    )
    # A one-layer comparison does not capture the final multi-layer manifest's
    # different stream-descriptor count. Reserve one alignment unit per layer;
    # the final aggregate artifact is still checked exactly before evaluation.
    serialized_allocation_budget_file_bytes = (
        max(0, ql_budget_file_bytes - artifact_alignment) if enforce_serialized_cap else 0
    )
    matched_options: list[Candidate] = []
    matched_row_indices: list[int] = []
    global_matched_options: list[Candidate] = []
    for rank in allocation_ranks:
        logical_nonzero = max_sparse_under_budget_bits(
            shape,
            bits=q.bits,
            scale_count=q.scale_count,
            rank=rank,
            budget_bits=ql_budget_bits,
            support_encoding=args.support_encoding,
            lowrank_factor_bits=lowrank_factor_bits,
        )
        global_candidate: Candidate | None = None
        if args.rate_allocation == "global_exact":
            global_sparse, global_factor, global_fit_diagnostics = fit_sparse_lowrank_components(
                residual_q=residual_q,
                covariance_tensor=covariance_tensor,
                factorizer=factorizer,
                nonzero=logical_nonzero,
                rank=rank,
                sparse_method=args.s_method,
                residual_order=args.residual_order,
                torch_dtype=weight_tensor.dtype,
                sparse_refit=args.strict_sparse_refit,
                covariance=covariance,
                metric=metric,
                obs_rcond=args.obs_rcond,
                lowrank_factor_bits=lowrank_factor_bits,
            )
            global_candidate = Candidate(
                "Q+S+L_QL_budget",
                target_ratio,
                layer,
                weight,
                q,
                global_sparse,
                global_factor,
                diagnostics={
                    "residual_order": args.residual_order,
                    "global_local_ql_repair_budget_multiplier": 1.0,
                    "global_cross_layer_borrowing_candidate": False,
                    **global_fit_diagnostics,
                },
            )
            global_matched_options.append(global_candidate)
        nonzero = logical_nonzero
        if enforce_serialized_cap:
            nonzero = _max_sparse_under_serialized_budget(
                layer=layer,
                q=q,
                shape=shape,
                rank=rank,
                logical_max_nonzero=logical_nonzero,
                budget_file_bytes=serialized_allocation_budget_file_bytes,
                lowrank_factor_bits=lowrank_factor_bits,
                alignment=artifact_alignment,
            )
            if nonzero < 0:
                continue
        if global_candidate is not None and nonzero == logical_nonzero:
            candidate = global_candidate
        else:
            sparse_part, factor, fit_diagnostics = fit_sparse_lowrank_components(
                residual_q=residual_q,
                covariance_tensor=covariance_tensor,
                factorizer=factorizer,
                nonzero=nonzero,
                rank=rank,
                sparse_method=args.s_method,
                residual_order=args.residual_order,
                torch_dtype=weight_tensor.dtype,
                sparse_refit=args.strict_sparse_refit,
                covariance=covariance,
                metric=metric,
                obs_rcond=args.obs_rcond,
                lowrank_factor_bits=lowrank_factor_bits,
            )
            candidate = Candidate(
                "Q+S+L_QL_budget",
                target_ratio,
                layer,
                weight,
                q,
                sparse_part,
                factor,
                diagnostics={"residual_order": args.residual_order, **fit_diagnostics},
            )
        payload = candidate.payload(support_encoding=args.support_encoding)
        if payload.total_bits > ql_budget_bits:
            raise AssertionError("Q+S+L_QL_budget exceeded its exact Q+L bit cap")
        artifact_file_bytes = (
            codec_artifact_natural_file_bytes([_artifact_layer(candidate)], alignment=artifact_alignment)
            if enforce_serialized_cap
            else 0
        )
        if enforce_serialized_cap and artifact_file_bytes > serialized_allocation_budget_file_bytes:
            raise AssertionError("Q+S+L_QL_budget exceeded its guarded serialized allocation cap")
        row_index = len(search_rows)
        search_rows.append(
            {
                "search_family": "Q+S+L_exact_QL_bit_cap",
                "strategy": candidate.strategy,
                "target_ratio": target_ratio,
                "layer": layer,
                "allocation_id": f"rank={rank},nnz={candidate.sparse_nnz}",
                "comparison_budget_bits": ql_budget_bits,
                "logical_max_sparse_nnz_before_serialized_cap": logical_nonzero,
                "comparison_budget_file_bytes": ql_budget_file_bytes or "",
                "serialized_allocation_budget_file_bytes": serialized_allocation_budget_file_bytes or "",
                "artifact_file_bytes": artifact_file_bytes or "",
                "serialized_rate_cap_satisfied": (
                    artifact_file_bytes <= ql_budget_file_bytes if enforce_serialized_cap else "not_evaluated"
                ),
                "payload_ratio": payload.ratio,
                "payload_bits": payload.total_bits,
                "under_target": payload.ratio <= target_ratio + 1e-12,
                "sparse_nnz": candidate.sparse_nnz,
                "lowrank_rank": candidate.rank,
                "both_s_and_l_active": candidate.sparse_nnz > 0 and candidate.rank > 0,
                "hessian_cost": _candidate_fast_cost(candidate, metric),
                "selected_within_layer_target": False,
            }
        )
        matched_options.append(candidate)
        matched_row_indices.append(row_index)
    if not matched_options:
        # At very low/discrete rates the historical one-alignment-unit guard
        # can be smaller than base Q itself.  Preserve a nested, aggregate-
        # feasible endpoint instead of taking min() over an empty list.
        nested_q_fallback = Candidate(
            "Q+S+L_QL_budget",
            target_ratio,
            layer,
            weight,
            q,
            diagnostics={
                "residual_order": args.residual_order,
                "allocation_fallback": "nested_Q_after_empty_local_alignment_guard",
                "strict_sparse_refit": "not_applicable",
            },
        )
        fallback_payload = nested_q_fallback.payload(support_encoding=args.support_encoding)
        if fallback_payload.total_bits > ql_budget_bits:
            raise AssertionError("nested Q fallback exceeds the Q+L logical payload")
        fallback_file_bytes = (
            codec_artifact_natural_file_bytes(
                [_artifact_layer(nested_q_fallback)], alignment=artifact_alignment
            )
            if enforce_serialized_cap
            else 0
        )
        if enforce_serialized_cap and fallback_file_bytes > ql_budget_file_bytes:
            raise AssertionError("nested Q fallback exceeds the Q+L file cap")
        row_index = len(search_rows)
        search_rows.append(
            {
                "search_family": "Q+S+L_empty_guard_nested_Q_fallback",
                "strategy": nested_q_fallback.strategy,
                "target_ratio": target_ratio,
                "layer": layer,
                "allocation_id": "rank=0,nnz=0,nested_Q_fallback",
                "comparison_budget_bits": ql_budget_bits,
                "logical_max_sparse_nnz_before_serialized_cap": 0,
                "comparison_budget_file_bytes": ql_budget_file_bytes or "",
                "serialized_allocation_budget_file_bytes": serialized_allocation_budget_file_bytes or "",
                "artifact_file_bytes": fallback_file_bytes or "",
                "serialized_rate_cap_satisfied": (
                    fallback_file_bytes <= ql_budget_file_bytes
                    if enforce_serialized_cap
                    else "not_evaluated"
                ),
                "payload_ratio": fallback_payload.ratio,
                "payload_bits": fallback_payload.total_bits,
                "under_target": fallback_payload.ratio <= target_ratio + 1e-12,
                "sparse_nnz": 0,
                "lowrank_rank": 0,
                "both_s_and_l_active": False,
                "hessian_cost": _candidate_fast_cost(nested_q_fallback, metric),
                "selected_within_layer_target": False,
            }
        )
        matched_options.append(nested_q_fallback)
        matched_row_indices.append(row_index)
    if args.rate_allocation == "global_exact" and args.global_frontier_top_ranks > 0:
        ranked_global = sorted(
            global_matched_options,
            key=lambda item: (_candidate_fast_cost(item, metric), item.rank, -item.sparse_nnz),
        )[: int(args.global_frontier_top_ranks)]
        existing_allocations = {
            (candidate.rank, candidate.sparse_nnz) for candidate in global_matched_options
        }
        for base_candidate in ranked_global:
            maximum = base_candidate.sparse_nnz
            for fraction in args.global_frontier_support_fractions:
                nonzero = int(math.floor(float(fraction) * maximum))
                key = (base_candidate.rank, nonzero)
                if nonzero <= 0 or key in existing_allocations:
                    continue
                sparse_part, factor, fit_diagnostics = fit_sparse_lowrank_components(
                    residual_q=residual_q,
                    covariance_tensor=covariance_tensor,
                    factorizer=factorizer,
                    nonzero=nonzero,
                    rank=base_candidate.rank,
                    sparse_method=args.s_method,
                    residual_order=args.residual_order,
                    torch_dtype=weight_tensor.dtype,
                    lowrank_factor_bits=lowrank_factor_bits,
                    sparse_refit=args.strict_sparse_refit,
                    covariance=covariance,
                    metric=metric,
                    obs_rcond=args.obs_rcond,
                )
                refined = Candidate(
                    "Q+S+L_QL_budget",
                    target_ratio,
                    layer,
                    weight,
                    q,
                    sparse_part,
                    factor,
                    diagnostics={
                        "residual_order": args.residual_order,
                        "global_support_fraction": float(fraction),
                        "global_support_parent_nnz": maximum,
                        **fit_diagnostics,
                    },
                )
                if refined.payload(support_encoding=args.support_encoding).total_bits > ql_budget_bits:
                    raise AssertionError("global support refinement exceeded its Q+L logical cap")
                global_matched_options.append(refined)
                existing_allocations.add(key)
            # The legacy pool stops at this layer's own Q+L logical payload.
            # Enumerating modest >1 budget bands permits the aggregate DP to
            # borrow bytes from other layers.  We refit from the residual for
            # every band because CSR metadata is nonlinear in nnz and, for
            # S->L, the low-rank target changes with the selected support.
            for multiplier in args.global_frontier_budget_multipliers:
                base_q_bits = int(q_payload.total_bits)
                local_repair_budget_bits = max(0, int(ql_budget_bits) - base_q_bits)
                band_budget_bits = base_q_bits + int(
                    math.floor(float(multiplier) * local_repair_budget_bits)
                )
                zero_sparse_payload = _payload_for_counts(
                    shape,
                    bits=q.bits,
                    scale_count=q.scale_count,
                    sparse_nonzero=0,
                    rank=base_candidate.rank,
                    support_encoding=args.support_encoding,
                    lowrank_factor_bits=lowrank_factor_bits,
                )
                if zero_sparse_payload.total_bits > band_budget_bits:
                    continue
                nonzero = max_sparse_under_budget_bits(
                    shape,
                    bits=q.bits,
                    scale_count=q.scale_count,
                    rank=base_candidate.rank,
                    budget_bits=band_budget_bits,
                    support_encoding=args.support_encoding,
                    lowrank_factor_bits=lowrank_factor_bits,
                )
                key = (base_candidate.rank, nonzero)
                if nonzero <= 0 or key in existing_allocations:
                    continue
                sparse_part, factor, fit_diagnostics = fit_sparse_lowrank_components(
                    residual_q=residual_q,
                    covariance_tensor=covariance_tensor,
                    factorizer=factorizer,
                    nonzero=nonzero,
                    rank=base_candidate.rank,
                    sparse_method=args.s_method,
                    residual_order=args.residual_order,
                    torch_dtype=weight_tensor.dtype,
                    sparse_refit=args.strict_sparse_refit,
                    covariance=covariance,
                    metric=metric,
                    obs_rcond=args.obs_rcond,
                    lowrank_factor_bits=lowrank_factor_bits,
                )
                borrowed = Candidate(
                    "Q+S+L_QL_budget",
                    target_ratio,
                    layer,
                    weight,
                    q,
                    sparse_part,
                    factor,
                    diagnostics={
                        "residual_order": args.residual_order,
                        "global_local_ql_repair_budget_multiplier": float(multiplier),
                        "global_local_base_q_bits": base_q_bits,
                        "global_local_ql_repair_budget_bits": local_repair_budget_bits,
                        "global_local_budget_bits": band_budget_bits,
                        "global_cross_layer_borrowing_candidate": True,
                        **fit_diagnostics,
                    },
                )
                if borrowed.payload(support_encoding=args.support_encoding).total_bits > band_budget_bits:
                    raise AssertionError("global budget-band candidate exceeded its logical band")
                global_matched_options.append(borrowed)
                existing_allocations.add(key)
    global_option_pools: dict[str, list[Candidate]] = {
        "Q+S+L_QL_budget": global_matched_options
    }
    if (
        args.rate_allocation == "global_exact"
        and args.include_global_single_component_controls
        and abs(float(target_ratio) - float(args.endpoint_target)) <= 1e-12
    ):
        global_option_pools.update(
            build_global_single_component_option_pools(
                layer=layer,
                weight_tensor=weight_tensor,
                covariance_tensor=covariance_tensor,
                covariance=covariance,
                factorizer=factorizer,
                metric=metric,
                target_ratio=target_ratio,
                q_candidate=q_candidate,
                ql_candidate=ql,
                qs_obs_candidate=qs_obs,
                allocation_ranks=allocation_ranks,
                lowrank_factor_bits=lowrank_factor_bits,
                args=args,
            )
        )
        for strategy in GLOBAL_CONTROL_STRATEGIES:
            for candidate in global_option_pools[strategy]:
                payload = candidate.payload(support_encoding=args.support_encoding)
                search_rows.append(
                    {
                        "search_family": "global_nonjoint_control_candidate_pool",
                        "strategy": strategy,
                        "target_ratio": target_ratio,
                        "layer": layer,
                        "allocation_id": (
                            f"rank={candidate.rank},nnz={candidate.sparse_nnz}"
                        ),
                        "payload_ratio": payload.ratio,
                        "payload_bits": payload.total_bits,
                        "sparse_nnz": candidate.sparse_nnz,
                        "lowrank_rank": candidate.rank,
                        "hessian_cost": _candidate_fast_cost(candidate, metric),
                        "selected_within_layer_target": False,
                    }
                )
    matched_joint = [
        index
        for index, candidate in enumerate(matched_options)
        if candidate.sparse_nnz > 0 and candidate.rank > 0
    ]
    matched_eligible = matched_joint or list(range(len(matched_options)))
    matched_index = min(matched_eligible, key=lambda index: _candidate_fast_cost(matched_options[index], metric))
    matched_qsl = matched_options[matched_index]
    matched_qsl.diagnostics.update(
        {
            "rate_cap_strategy": (
                "per_layer_exact_Q+L_serialized_bytes"
                if enforce_serialized_cap
                else "per_layer_exact_Q+L_payload_bits"
            ),
            "comparison_budget_bits": ql_budget_bits,
            "comparison_budget_file_bytes": ql_budget_file_bytes or "",
            "serialized_allocation_budget_file_bytes": serialized_allocation_budget_file_bytes or "",
            "serialized_alignment_guard_bytes": artifact_alignment if enforce_serialized_cap else "",
            "joint_component_constraint": (
                "both_S_and_L_required" if matched_joint else "discrete_budget_prevented_joint_S_and_L"
            ),
        }
    )
    search_rows[matched_row_indices[matched_index]]["selected_within_layer_target"] = True
    selected[matched_qsl.strategy] = matched_qsl
    matched_scaled = make_component_scaled_candidate(
        matched_qsl,
        covariance,
        args.scale_bounds,
        strategy="Q+S+L_QL_budget_component_scale",
    )
    matched_scaled.diagnostics.update(
        {
            "rate_cap_strategy": (
                "per_layer_exact_Q+L_serialized_bytes"
                if enforce_serialized_cap
                else "per_layer_exact_Q+L_payload_bits"
            ),
            "comparison_budget_bits": ql_budget_bits,
            "comparison_budget_file_bytes": ql_budget_file_bytes or "",
            "serialized_allocation_budget_file_bytes": serialized_allocation_budget_file_bytes or "",
            "serialized_alignment_guard_bytes": artifact_alignment if enforce_serialized_cap else "",
        }
    )
    selected[matched_scaled.strategy] = matched_scaled

    # Isolate OBS at the exact same discrete S/L allocation selected above.
    if args.residual_order == "s_then_l":
        obs_target = residual_q
        fixed_obs_factor = None
    elif args.residual_order == "l_then_s":
        fixed_obs_factor = quantize_lowrank_factors(
            factorizer.factorize(residual_q, best_qsl.rank),
            bits=lowrank_factor_bits,
        )
        fixed_lowrank = (
            np.zeros_like(weight) if fixed_obs_factor is None else fixed_obs_factor.decode()
        )
        obs_target = residual_q - fixed_lowrank
    else:  # guarded by argparse, retained as a fail-closed internal check
        raise ValueError(f"unsupported residual order: {args.residual_order}")
    if best_qsl.sparse is not None:
        obs_sparse, qsl_obs_diag = obs_refit_sparse(
            obs_target,
            best_qsl.sparse,
            covariance,
            metric=metric,
            rcond=args.obs_rcond,
        )
    else:
        obs_sparse, qsl_obs_diag = None, {"obs_applied": False, "obs_reason": "empty_support"}
    if args.residual_order == "s_then_l":
        residual_after_obs = residual_q - (
            np.zeros_like(weight) if obs_sparse is None else obs_sparse.decode()
        )
        obs_factor = quantize_lowrank_factors(
            factorizer.factorize(residual_after_obs, best_qsl.rank),
            bits=lowrank_factor_bits,
        )
    else:
        obs_factor = fixed_obs_factor
    qsl_obs = Candidate(
        "Q+S_OBS+L",
        target_ratio,
        layer,
        weight,
        q,
        obs_sparse,
        obs_factor,
        diagnostics={
            **qsl_obs_diag,
            "allocation_source": "same_rank_and_support_as_selected_Q+S+L",
            "residual_order": args.residual_order,
        },
        repair_dof=0 if obs_sparse is None else obs_sparse.nonzero_count,
    )
    selected["Q+S_OBS+L"] = qsl_obs

    component_scaled = make_component_scaled_candidate(best_qsl, covariance, args.scale_bounds)
    selected["Q+S+L_component_scale"] = component_scaled

    # Mark the winning block row after selection.
    if best_block is not None:
        best_block_id = f"block={best_block.q.col_block_size}"
        for row in search_rows:
            if (
                row.get("search_family") == "Q_block_scale"
                and row.get("allocation_id") == best_block_id
            ):
                row["selected_within_layer_target"] = True

    # Full codec-faithful metrics are emitted only for selected candidates; the
    # allocation search rows above retain the cheaper proxy for every option.
    for strategy, candidate in selected.items():
        metrics = candidate_geometry(
            candidate,
            metric,
            activation_samples,
            support_encoding=args.support_encoding,
            rho_threshold=args.rho_threshold,
            q_reference_cost=q_cost,
            q_reference_bits=q_payload.total_bits,
        )
        metrics["search_family"] = "selected_endpoint"
        metrics["selected_within_layer_target"] = True
        search_rows.append(metrics)
    return selected, search_rows, global_option_pools


def _select_global_exact_pareto_allocation(
    *,
    cap_candidates: dict[str, Candidate],
    option_pools: dict[str, list[Candidate]],
    metrics: dict[str, HessianMetric],
    alignment: int,
    endpoint_label: str,
    optimality_scope: str,
) -> tuple[dict[str, Candidate], dict[str, object]]:
    """Choose the single proxy-best allocation, preserving the historical API."""

    ranked, report = _rank_global_exact_pareto_allocations(
        cap_candidates=cap_candidates,
        option_pools=option_pools,
        metrics=metrics,
        alignment=alignment,
        endpoint_label=endpoint_label,
        optimality_scope=optimality_scope,
        top_k=1,
    )
    return ranked[0].candidates, report


def _allocation_digest(
    selected: dict[str, Candidate],
    *,
    layers: list[str],
    choices: tuple[int, ...],
) -> str:
    payload = []
    for layer, choice in zip(layers, choices, strict=True):
        allocation = _artifact_allocation(selected[layer])
        candidate = selected[layer]
        payload.append(
            {
                "option_index": int(choice),
                "name": allocation.name,
                "shape": list(allocation.shape),
                "q_bits": allocation.q_bits,
                "q_quantizer": candidate.q.quantizer,
                "q_scale_shape": list(allocation.q_scale_shape),
                "q_col_block_size": allocation.q_col_block_size,
                "sparse_nnz": allocation.sparse_nnz,
                "lowrank_rank": allocation.lowrank_rank,
                "lowrank_factor_bits": allocation.lowrank_factor_bits,
                "lowrank_left_scale_shape": list(
                    allocation.lowrank_left_scale_shape
                ),
                "lowrank_right_scale_shape": list(
                    allocation.lowrank_right_scale_shape
                ),
            }
        )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _rank_global_exact_pareto_allocations(
    *,
    cap_candidates: dict[str, Candidate],
    option_pools: dict[str, list[Candidate]],
    metrics: dict[str, HessianMetric],
    alignment: int,
    endpoint_label: str,
    optimality_scope: str,
    top_k: int,
    required_natural_file_bytes: int | None = None,
) -> tuple[list[RankedGlobalAllocation], dict[str, object]]:
    """Choose a globally feasible endpoint under one canonical file-byte cap.

    The dynamic program constructs a safe three-dimensional Pareto frontier in
    canonical JSON-header bytes, payload-end bytes and Hessian cost.  These are
    sufficient monotone resources when the same sorted layer suffix is added;
    unlike sums of one-layer files, they cannot reverse order through container
    alignment.  The Q+L cap and selected endpoint are cross-checked by the real
    serializer.  Optimality is exact inside the enumerated candidate pool.
    """

    requested_top_k = int(top_k)
    if requested_top_k <= 0:
        raise ValueError("global allocation top_k must be positive")
    layers = sorted(cap_candidates)
    if (
        not layers
        or set(option_pools) != set(layers)
        or set(metrics) != set(layers)
    ):
        raise AssertionError(
            f"global {endpoint_label} allocation requires identical non-empty layer sets"
        )
    for layer in layers:
        if cap_candidates[layer].layer != layer or any(
            candidate.layer != layer for candidate in option_pools[layer]
        ):
            raise AssertionError(
                f"global {endpoint_label} candidate layer differs from its mapping key"
            )
    cap_bytes = codec_artifact_allocations_natural_file_bytes(
        [_artifact_allocation(cap_candidates[layer]) for layer in layers],
        alignment=alignment,
    )
    serialized_cap_bytes = codec_artifact_natural_file_bytes(
        [_artifact_layer(cap_candidates[layer]) for layer in layers],
        alignment=alignment,
    )
    if serialized_cap_bytes != cap_bytes:
        raise AssertionError("metadata Q+L byte oracle differs from the real serializer")
    required_natural = (
        None
        if required_natural_file_bytes is None
        else int(required_natural_file_bytes)
    )
    if required_natural is not None and (
        required_natural <= 0 or required_natural > cap_bytes
    ):
        raise ValueError(
            "required natural file bytes must be positive and no larger than the Q+L cap"
        )
    active_cap_bytes = cap_bytes if required_natural is None else required_natural
    full_serializer_cross_checks = 1
    normalized: dict[
        str, list[tuple[Candidate, int, float, LayerCodecAllocation]]
    ] = {}
    for layer in layers:
        eligible = list(option_pools[layer])
        if not eligible:
            raise AssertionError(f"global {endpoint_label} allocation has no options for {layer}")
        deduplicated: dict[
            tuple[LayerCodecAllocation, int, int, int],
            tuple[Candidate, int, float, LayerCodecAllocation],
        ] = {}
        for candidate in eligible:
            allocation = _artifact_allocation(candidate)
            size = codec_artifact_allocations_natural_file_bytes(
                [allocation], alignment=alignment
            )
            cost = metrics[layer].cost(candidate.final - candidate.weight)
            if not math.isfinite(cost):
                raise ValueError(
                    f"global {endpoint_label} candidate cost is non-finite for {layer}"
                )
            value_identity = (
                allocation,
                id(candidate.q),
                id(candidate.sparse),
                id(candidate.lowrank),
            )
            previous = deduplicated.get(value_identity)
            if previous is None or cost < previous[2]:
                deduplicated[value_identity] = (candidate, size, cost, allocation)
        normalized[layer] = sorted(
            deduplicated.values(), key=lambda item: (item[1], item[2], item[0].rank)
        )

    # (raw canonical header bytes, payload-end bytes, natural bytes,
    #  summed Hessian cost, chosen option indices).  Componentwise dominance
    # in the first two resources is safe under a common sorted layer suffix.
    states: list[tuple[int, int, int, float, tuple[int, ...]]] = [(0, 0, 0, 0.0, ())]
    peak_state_count = 1
    partial_layouts_checked = 0
    frontier_state_limit = GLOBAL_ALLOCATOR_FRONTIER_STATE_LIMIT
    expanded_state_limit = GLOBAL_ALLOCATOR_EXPANDED_STATE_LIMIT
    for layer_index, layer in enumerate(layers):
        expanded: list[tuple[int, int, int, float, tuple[int, ...]]] = []
        prefix_layers = layers[: layer_index + 1]
        planned_expansions = len(states) * len(normalized[layer])
        if planned_expansions > expanded_state_limit:
            message = (
                "exact global allocator expansion exceeded the fail-closed state limit"
            )
            if required_natural is not None:
                raise ExactNaturalMatchSearchLimitError(
                    message,
                    diagnostics={
                        "limit_kind": "expanded_states",
                        "layer_index": layer_index,
                        "layer": layer,
                        "input_state_count": len(states),
                        "layer_option_count": len(normalized[layer]),
                        "planned_expansions": planned_expansions,
                        "state_limit": expanded_state_limit,
                    },
                )
            raise RuntimeError(message)
        for _header, _payload_end, _natural, state_cost, choices in states:
            for option_index, (_candidate, _size, option_cost, _allocation) in enumerate(
                normalized[layer]
            ):
                next_choices = (*choices, option_index)
                allocations = [
                    normalized[prefix_layer][choice][3]
                    for prefix_layer, choice in zip(
                        prefix_layers, next_choices, strict=True
                    )
                ]
                layout = codec_artifact_allocations_layout(
                    allocations, alignment=alignment
                )
                partial_layouts_checked += 1
                # Appending streams and manifest records cannot reduce either
                # resource, so an over-cap prefix can be rejected permanently.
                if layout.natural_file_bytes > active_cap_bytes:
                    continue
                combined_cost = state_cost + option_cost
                if not math.isfinite(combined_cost):
                    raise ValueError(
                        f"global {endpoint_label} accumulated cost is non-finite"
                    )
                expanded.append(
                    (
                        layout.header_bytes,
                        layout.payload_end_bytes,
                        layout.natural_file_bytes,
                        combined_cost,
                        next_choices,
                    )
                )

        frontier: list[tuple[int, int, int, float, tuple[int, ...]]] = []
        if required_natural is not None:
            # For equality-constrained allocation, a lower-byte state cannot
            # dominate a larger state because only the latter may reach the
            # requested final natural size.  Retaining top-K choices for each
            # exact canonical-layout state is safe: a common sorted suffix sees
            # the same header length and payload end, hence the same future
            # offsets and final byte count.
            expanded.sort(
                key=lambda item: (item[0], item[1], item[2], item[3], item[4])
            )
            retained_per_layout: dict[tuple[int, int, int], int] = {}
            for state in expanded:
                layout_key = (state[0], state[1], state[2])
                retained = retained_per_layout.get(layout_key, 0)
                if retained >= requested_top_k:
                    continue
                frontier.append(state)
                retained_per_layout[layout_key] = retained + 1
        else:
            expanded.sort(key=lambda item: (item[0], item[1], item[3], item[4]))
            payload_values = sorted({state[1] for state in expanded})
            prefix_best_costs: list[list[float]] = [
                [] for _ in range(len(payload_values) + 1)
            ]

            def best_costs_through(index: int) -> list[float]:
                best_costs: list[float] = []
                while index > 0:
                    best_costs.extend(prefix_best_costs[index])
                    index -= index & -index
                best_costs.sort()
                return best_costs[:requested_top_k]

            def record_cost(index: int, cost: float) -> None:
                while index < len(prefix_best_costs):
                    bucket = prefix_best_costs[index]
                    bucket.append(cost)
                    bucket.sort()
                    del bucket[requested_top_k:]
                    index += index & -index

            for state in expanded:
                payload_index = bisect.bisect_left(payload_values, state[1]) + 1
                dominating_costs = best_costs_through(payload_index)
                if (
                    len(dominating_costs) < requested_top_k
                    or state[3] < dominating_costs[-1]
                ):
                    frontier.append(state)
                    record_cost(payload_index, state[3])
        if len(frontier) > frontier_state_limit:
            message = (
                "exact global allocator Pareto frontier exceeded the fail-closed state limit"
            )
            if required_natural is not None:
                raise ExactNaturalMatchSearchLimitError(
                    message,
                    diagnostics={
                        "limit_kind": "pareto_frontier",
                        "layer_index": layer_index,
                        "layer": layer,
                        "frontier_state_count": len(frontier),
                        "state_limit": frontier_state_limit,
                    },
                )
            raise RuntimeError(message)
        states = frontier
        peak_state_count = max(peak_state_count, len(states))

    exact_checked = len(states)
    ranked: list[RankedGlobalAllocation] = []
    for _header, _payload_end, natural, cost, choices in sorted(
        states, key=lambda item: (item[3], -item[2], item[4])
    ):
        selected = {
            layer: normalized[layer][choice][0]
            for layer, choice in zip(layers, choices, strict=True)
        }
        if natural > cap_bytes:
            continue
        if required_natural is not None and natural != required_natural:
            continue
        serialized_selected_natural = codec_artifact_natural_file_bytes(
            [_artifact_layer(selected[layer]) for layer in layers], alignment=alignment
        )
        full_serializer_cross_checks += 1
        if serialized_selected_natural != natural:
            raise AssertionError("metadata selected-byte oracle differs from the real serializer")
        ranked.append(
            RankedGlobalAllocation(
                candidates=selected,
                natural_file_bytes=int(natural),
                hessian_cost=float(cost),
                choices=choices,
                allocation_digest=_allocation_digest(
                    selected,
                    layers=layers,
                    choices=choices,
                ),
            )
        )
        if len(ranked) >= requested_top_k:
            break

    match_available = bool(ranked)
    if not ranked and required_natural is None:
        raise RuntimeError(
            f"global {endpoint_label} allocator found no feasible canonical layout; "
            "fallback is forbidden"
        )
    selected_cost = ranked[0].hessian_cost if ranked else None
    selected_natural = ranked[0].natural_file_bytes if ranked else None
    selection_source = (
        "global_exact_canonical_layout_exact_natural_dynamic_program"
        if required_natural is not None
        else "global_exact_canonical_layout_pareto_frontier"
    )

    return ranked, {
        "mode": "global_exact",
        "endpoint_label": endpoint_label,
        "selection_source": selection_source,
        "optimality_scope": optimality_scope,
        "layer_count": len(layers),
        "candidate_count": sum(len(items) for items in normalized.values()),
        "final_pareto_state_count": len(states),
        "peak_pareto_state_count": peak_state_count,
        "frontier_coarsening_events": 0,
        "maximum_pre_coarsening_state_count": 0,
        "frontier_fail_closed_state_limit": frontier_state_limit,
        "expansion_fail_closed_state_limit": expanded_state_limit,
        "exact_combinations_checked": exact_checked,
        "metadata_exact_combinations_checked": exact_checked,
        "partial_metadata_layouts_checked": partial_layouts_checked,
        "full_serializer_cross_checks": full_serializer_cross_checks,
        "byte_oracle": (
            "canonical_manifest_header_payload_metadata_pareto_with_full_serializer_cross_check"
        ),
        "q_l_cap_natural_file_bytes": cap_bytes,
        "selected_natural_file_bytes": selected_natural,
        "unused_natural_bytes_before_tail_padding": (
            None if selected_natural is None else cap_bytes - selected_natural
        ),
        "selected_hessian_cost": selected_cost,
        "strict_file_byte_feasible": (
            False if selected_natural is None else selected_natural <= cap_bytes
        ),
        "required_natural_file_bytes": required_natural,
        "exact_natural_file_byte_match_required": required_natural is not None,
        "exact_natural_file_byte_match_available": match_available,
        "exact_natural_file_byte_match": (
            required_natural is not None
            and selected_natural is not None
            and selected_natural == required_natural
        ),
        "proxy_top_k_requested": requested_top_k,
        "proxy_top_k_returned": len(ranked),
        "proxy_top_k": [
            {
                "proxy_rank": proxy_rank,
                "natural_file_bytes": allocation.natural_file_bytes,
                "hessian_cost": allocation.hessian_cost,
                "allocation_digest": allocation.allocation_digest,
            }
            for proxy_rank, allocation in enumerate(ranked, start=1)
        ],
    }


def rank_global_exact_qsl_allocations(
    *,
    ql_candidates: dict[str, Candidate],
    degenerate_candidates: dict[str, list[Candidate]],
    option_pools: dict[str, list[Candidate]],
    fallback_candidates: dict[str, Candidate],
    metrics: dict[str, HessianMetric],
    alignment: int,
    top_k: int,
) -> tuple[list[RankedGlobalAllocation], dict[str, object]]:
    """Return the proxy top-K QSL allocations under the exact shared cap."""

    layers = sorted(ql_candidates)
    if (
        not layers
        or set(degenerate_candidates) != set(layers)
        or set(option_pools) != set(layers)
        or set(fallback_candidates) != set(layers)
        or set(metrics) != set(layers)
    ):
        raise AssertionError("global QSL allocation requires identical non-empty layer sets")
    pooled: dict[str, list[Candidate]] = {}
    for layer in layers:
        candidates = [
            ql_candidates[layer],
            fallback_candidates[layer],
            *degenerate_candidates[layer],
            *option_pools[layer],
        ]
        if any(candidate.layer != layer for candidate in candidates):
            raise AssertionError("global QSL candidate layer differs from its mapping key")
        pooled[layer] = list(option_pools[layer])
        pooled[layer].append(
            _candidate_with_strategy(
                fallback_candidates[layer],
                "Q+S+L_QL_budget",
                diagnostics={
                    "global_local_guard_sentinel": True,
                    "global_allocator_candidate_role": "eligible_audit_sentinel",
                },
            )
        )
        for source in degenerate_candidates[layer]:
            pooled[layer].append(
                _candidate_with_strategy(
                    source,
                    "Q+S+L_QL_budget",
                    diagnostics={
                        "global_degenerate_state": source.strategy,
                        "strict_sparse_refit": (
                            "obs" if "OBS" in source.strategy else "not_applicable"
                        ),
                    },
                )
            )

    optimality_scope = (
        "enumerated_rank_support_budget_band_and_supplied_degenerate_pure_candidate_pool_with_"
        "safe_header_payload_cost_pareto_and_exact_final_serialization"
    )
    ranked, report = _rank_global_exact_pareto_allocations(
        cap_candidates=ql_candidates,
        option_pools=pooled,
        metrics=metrics,
        alignment=alignment,
        endpoint_label="Q+S+L_QL_budget",
        optimality_scope=optimality_scope,
        top_k=top_k,
    )
    # The historical local-guard endpoint is retained only as a protocol audit
    # sentinel.  It must be finite and feasible, and the exact Pareto result
    # must weakly dominate it; otherwise stop instead of silently falling back.
    fallback_natural = codec_artifact_allocations_natural_file_bytes(
        [_artifact_allocation(fallback_candidates[layer]) for layer in layers],
        alignment=alignment,
    )
    cap_bytes = int(report["q_l_cap_natural_file_bytes"])
    if fallback_natural > cap_bytes:
        raise AssertionError("legacy guarded QSL fallback exceeds the aggregate Q+L file cap")
    fallback_cost = sum(
        metrics[layer].cost(
            fallback_candidates[layer].final - fallback_candidates[layer].weight
        )
        for layer in layers
    )
    if not math.isfinite(fallback_cost):
        raise ValueError("global QSL fallback cost is non-finite")
    selected_cost = ranked[0].hessian_cost
    selected_natural = ranked[0].natural_file_bytes
    if (fallback_cost, -fallback_natural) < (selected_cost, -selected_natural):
        raise RuntimeError(
            "exact global allocator lost a feasible local-guard sentinel; fallback is forbidden"
        )
    report["selected_qsl_natural_file_bytes"] = selected_natural
    report["fallback_policy"] = "audit_only_fail_closed_never_selected"
    return ranked, report


def _decorate_global_qsl_selection(
    selected: dict[str, Candidate],
    *,
    ql_candidates: dict[str, Candidate],
    report: dict[str, object],
    optimality_scope: str,
) -> None:
    cap_bytes = int(report["q_l_cap_natural_file_bytes"])
    selected_natural = int(report["selected_natural_file_bytes"])
    for layer, candidate in selected.items():
        candidate.diagnostics.update(
            {
                "rate_cap_strategy": "aggregate_exact_Q+L_serialized_bytes",
                "comparison_budget_bits": ql_candidates[layer].payload(
                    support_encoding="csr_fixed"
                ).total_bits,
                "global_allocator_selection_source": report["selection_source"],
                "global_allocator_optimality_scope": optimality_scope,
                "global_q_l_cap_natural_file_bytes": cap_bytes,
                "global_qsl_natural_file_bytes": selected_natural,
                "global_qsl_unused_natural_bytes": cap_bytes - selected_natural,
                "global_allocator_frontier_coarsened": False,
            }
        )


def select_global_exact_qsl_allocation(
    *,
    ql_candidates: dict[str, Candidate],
    degenerate_candidates: dict[str, list[Candidate]],
    option_pools: dict[str, list[Candidate]],
    fallback_candidates: dict[str, Candidate],
    metrics: dict[str, HessianMetric],
    alignment: int,
) -> tuple[dict[str, Candidate], dict[str, object]]:
    """Choose the QSL endpoint with the shared exact canonical-layout allocator."""

    ranked, report = rank_global_exact_qsl_allocations(
        ql_candidates=ql_candidates,
        degenerate_candidates=degenerate_candidates,
        option_pools=option_pools,
        fallback_candidates=fallback_candidates,
        metrics=metrics,
        alignment=alignment,
        top_k=1,
    )
    selected = ranked[0].candidates
    optimality_scope = str(report["optimality_scope"])
    _decorate_global_qsl_selection(
        selected,
        ql_candidates=ql_candidates,
        report=report,
        optimality_scope=optimality_scope,
    )
    return selected, report


def rank_global_exact_component_allocations(
    *,
    strategy: str,
    ql_candidates: dict[str, Candidate],
    option_pools: dict[str, list[Candidate]],
    metrics: dict[str, HessianMetric],
    alignment: int,
    optimality_scope: str,
    candidate_pool_asymmetry: str,
    top_k: int,
    required_natural_file_bytes: int | None = None,
) -> tuple[list[RankedGlobalAllocation], dict[str, object]]:
    """Return proxy top-K allocations for a pure or no-joint control."""

    if strategy not in GLOBAL_CONTROL_STRATEGIES:
        raise ValueError(f"unsupported global control strategy: {strategy}")
    for layer, candidates in option_pools.items():
        for candidate in candidates:
            if candidate.strategy != strategy:
                raise AssertionError(
                    f"{strategy} pool contains candidate labelled {candidate.strategy}"
                )
            if (
                strategy == GLOBAL_NONJOINT_CONTROL_STRATEGY
                and candidate.sparse_nnz > 0
                and candidate.rank > 0
            ):
                raise AssertionError(
                    f"{strategy} pool contains a forbidden joint S+L candidate for {layer}"
                )
            if candidate.diagnostics.get("allocation_fallback"):
                raise RuntimeError(
                    f"{strategy} pool contains a fallback candidate for {layer}; fallback is forbidden"
                )
    ranked, report = _rank_global_exact_pareto_allocations(
        cap_candidates=ql_candidates,
        option_pools=option_pools,
        metrics=metrics,
        alignment=alignment,
        endpoint_label=strategy,
        optimality_scope=optimality_scope,
        top_k=top_k,
        required_natural_file_bytes=required_natural_file_bytes,
    )
    if strategy == GLOBAL_NONJOINT_CONTROL_STRATEGY:
        for allocation in ranked:
            for layer, candidate in allocation.candidates.items():
                if candidate.sparse_nnz > 0 and candidate.rank > 0:
                    raise RuntimeError(
                        f"no-joint global control selected a joint S+L state for {layer}"
                    )
    expected_selection_source = (
        "global_exact_canonical_layout_exact_natural_dynamic_program"
        if required_natural_file_bytes is not None
        else "global_exact_canonical_layout_pareto_frontier"
    )
    if report["selection_source"] != expected_selection_source:
        raise RuntimeError(f"{strategy} did not resolve through the exact Pareto allocator")
    report.update(
        {
            "strategy": strategy,
            "candidate_pool_asymmetry": candidate_pool_asymmetry,
            "fallback_policy": "forbidden_fail_closed",
        }
    )
    return ranked, report


def attempt_exact_natural_component_allocations(
    *,
    strategy: str,
    ql_candidates: dict[str, Candidate],
    option_pools: dict[str, list[Candidate]],
    metrics: dict[str, HessianMetric],
    alignment: int,
    optimality_scope: str,
    candidate_pool_asymmetry: str,
    top_k: int,
    required_natural_file_bytes: int,
) -> tuple[list[RankedGlobalAllocation], dict[str, object]]:
    """Attempt the optional exact-byte control without weakening claim gating."""

    try:
        return rank_global_exact_component_allocations(
            strategy=strategy,
            ql_candidates=ql_candidates,
            option_pools=option_pools,
            metrics=metrics,
            alignment=alignment,
            optimality_scope=optimality_scope,
            candidate_pool_asymmetry=candidate_pool_asymmetry,
            top_k=top_k,
            required_natural_file_bytes=required_natural_file_bytes,
        )
    except ExactNaturalMatchSearchLimitError as exc:
        layers = sorted(ql_candidates)
        cap_bytes = codec_artifact_allocations_natural_file_bytes(
            [_artifact_allocation(ql_candidates[layer]) for layer in layers],
            alignment=alignment,
        )
        return [], {
            "mode": "global_exact",
            "strategy": strategy,
            "endpoint_label": strategy,
            "selection_source": (
                "global_exact_canonical_layout_exact_natural_dynamic_program"
            ),
            "optimality_scope": optimality_scope,
            "candidate_pool_asymmetry": candidate_pool_asymmetry,
            "layer_count": len(layers),
            "candidate_count": sum(len(option_pools[layer]) for layer in layers),
            "q_l_cap_natural_file_bytes": cap_bytes,
            "selected_natural_file_bytes": None,
            "selected_hessian_cost": None,
            "required_natural_file_bytes": int(required_natural_file_bytes),
            "exact_natural_file_byte_match_required": True,
            "exact_natural_file_byte_match_available": False,
            "exact_natural_file_byte_match": False,
            "search_completed": False,
            "search_status": "state_limit_exceeded",
            "search_error": str(exc),
            "search_limit_diagnostics": exc.diagnostics,
            "frontier_fail_closed_state_limit": (
                GLOBAL_ALLOCATOR_FRONTIER_STATE_LIMIT
            ),
            "expansion_fail_closed_state_limit": (
                GLOBAL_ALLOCATOR_EXPANDED_STATE_LIMIT
            ),
            "fallback_policy": (
                "retain_cap_best_nojoint_for_description_disable_joint_claim"
            ),
        }


def _decorate_global_component_selection(
    selected: dict[str, Candidate],
    *,
    strategy: str,
    ql_candidates: dict[str, Candidate],
    report: dict[str, object],
    optimality_scope: str,
    candidate_pool_asymmetry: str,
) -> None:
    cap_bytes = int(report["q_l_cap_natural_file_bytes"])
    selected_natural = int(report["selected_natural_file_bytes"])
    for layer, candidate in selected.items():
        candidate.diagnostics.update(
            {
                "rate_cap_strategy": "aggregate_exact_Q+L_serialized_bytes",
                "comparison_budget_bits": ql_candidates[layer].payload(
                    support_encoding="csr_fixed"
                ).total_bits,
                "global_allocator_selection_source": report["selection_source"],
                "global_allocator_optimality_scope": optimality_scope,
                "global_allocator_candidate_pool_asymmetry": candidate_pool_asymmetry,
                "global_q_l_cap_natural_file_bytes": cap_bytes,
                "global_control_natural_file_bytes": selected_natural,
                "global_control_unused_natural_bytes": cap_bytes - selected_natural,
                "global_allocator_frontier_coarsened": False,
            }
        )


def _apply_validation_selection_to_allocator_report(
    report: dict[str, object],
    *,
    allocation: RankedGlobalAllocation,
    selection_report: dict[str, object],
) -> None:
    """Preserve proxy evidence while making the validation winner explicit."""

    report["proxy_best_hessian_cost"] = report["selected_hessian_cost"]
    report["proxy_best_natural_file_bytes"] = report["selected_natural_file_bytes"]
    report["proxy_best_unused_natural_bytes_before_tail_padding"] = report[
        "unused_natural_bytes_before_tail_padding"
    ]
    allocator_selection_source = report["selection_source"]
    report.update(
        {
            key: value
            for key, value in selection_report.items()
            if key != "selection_source"
        }
    )
    report["selection_source"] = allocator_selection_source
    report["final_selection_source"] = selection_report["selection_source"]
    report["selected_hessian_cost"] = allocation.hessian_cost
    report["selected_natural_file_bytes"] = allocation.natural_file_bytes
    cap_bytes = int(report["q_l_cap_natural_file_bytes"])
    report["unused_natural_bytes_before_tail_padding"] = (
        cap_bytes - allocation.natural_file_bytes
    )
    report["strict_file_byte_feasible"] = allocation.natural_file_bytes <= cap_bytes
    if report.get("endpoint_label") == "Q+S+L_QL_budget":
        report["selected_qsl_natural_file_bytes"] = allocation.natural_file_bytes


def select_global_exact_component_allocation(
    *,
    strategy: str,
    ql_candidates: dict[str, Candidate],
    option_pools: dict[str, list[Candidate]],
    metrics: dict[str, HessianMetric],
    alignment: int,
    optimality_scope: str,
    candidate_pool_asymmetry: str,
) -> tuple[dict[str, Candidate], dict[str, object]]:
    """Allocate a pure-family or no-joint control under the shared byte cap."""

    ranked, report = rank_global_exact_component_allocations(
        strategy=strategy,
        ql_candidates=ql_candidates,
        option_pools=option_pools,
        metrics=metrics,
        alignment=alignment,
        optimality_scope=optimality_scope,
        candidate_pool_asymmetry=candidate_pool_asymmetry,
        top_k=1,
    )
    selected = ranked[0].candidates
    _decorate_global_component_selection(
        selected,
        strategy=strategy,
        ql_candidates=ql_candidates,
        report=report,
        optimality_scope=optimality_scope,
        candidate_pool_asymmetry=candidate_pool_asymmetry,
    )
    return selected, report


def _empty_aggregate(strategy: str, target: float) -> dict[str, object]:
    return {
        "strategy": strategy,
        "target_ratio": target,
        "layers": 0,
        "layers_s_active": 0,
        "layers_l_active": 0,
        "layers_both_s_l_active": 0,
        "reference_bits": 0,
        "payload_bits": 0,
        "hessian_cost": 0.0,
        "baseline_hessian_energy": 0.0,
        "hessian_self_q": 0.0,
        "hessian_self_s": 0.0,
        "hessian_self_l": 0.0,
        "hessian_cross_qs": 0.0,
        "hessian_cross_ql": 0.0,
        "hessian_cross_sl": 0.0,
        "activation_error_weighted": 0.0,
        "worst_token_risk": 0.0,
        "token_risk_p95": 0.0,
        "sparse_nnz": 0,
        "lowrank_rank_sum": 0,
        "q_scale_count": 0,
        "folded_repair_dof": 0,
        "comparison_budget_bits": 0,
        "rate_cap_strategy": "",
        "_q_bits_by_layer": {},
        "_q_quantizers_by_layer": {},
        "_q_group_sizes_by_layer": {},
        "_lowrank_factor_bits_by_layer": {},
    }


def update_aggregate(aggregate: dict[str, object], metrics: dict[str, object]) -> None:
    aggregate["layers"] = int(aggregate["layers"]) + 1
    aggregate["layers_s_active"] = int(aggregate["layers_s_active"]) + int(bool(metrics["s_active"]))
    aggregate["layers_l_active"] = int(aggregate["layers_l_active"]) + int(bool(metrics["l_active"]))
    aggregate["layers_both_s_l_active"] = int(aggregate["layers_both_s_l_active"]) + int(
        bool(metrics["s_active"]) and bool(metrics["l_active"])
    )
    for key in (
        "reference_bits",
        "payload_bits",
        "hessian_cost",
        "baseline_hessian_energy",
        "hessian_self_q",
        "hessian_self_s",
        "hessian_self_l",
        "hessian_cross_qs",
        "hessian_cross_ql",
        "hessian_cross_sl",
        "sparse_nnz",
        "q_scale_count",
        "folded_repair_dof",
    ):
        aggregate[key] = float(aggregate[key]) + float(metrics[key])
    aggregate["lowrank_rank_sum"] = int(aggregate["lowrank_rank_sum"]) + int(metrics["lowrank_rank"])
    aggregate["activation_error_weighted"] = float(aggregate["activation_error_weighted"]) + float(
        metrics["activation_reconstruction_error"]
    ) * float(metrics["baseline_hessian_energy"])
    aggregate["worst_token_risk"] = max(float(aggregate["worst_token_risk"]), float(metrics["worst_token_risk"]))
    aggregate["token_risk_p95"] = max(float(aggregate["token_risk_p95"]), float(metrics["token_risk_p95"]))
    if metrics.get("comparison_budget_bits") not in {None, ""}:
        aggregate["comparison_budget_bits"] = int(aggregate["comparison_budget_bits"]) + int(
            float(metrics["comparison_budget_bits"])
        )
    if metrics.get("rate_cap_strategy"):
        aggregate["rate_cap_strategy"] = str(metrics["rate_cap_strategy"])
    layer = str(metrics["layer"])
    q_group_size = int(float(metrics.get("q_col_block_size", 0)))
    lowrank_factor_bits = int(float(metrics.get("lowrank_factor_bits", 0)))
    for field, value in (
        ("_q_bits_by_layer", int(float(metrics["q_bits"]))),
        ("_q_quantizers_by_layer", str(metrics["q_quantizer"])),
        ("_q_group_sizes_by_layer", q_group_size),
        ("_lowrank_factor_bits_by_layer", lowrank_factor_bits),
    ):
        mapping = aggregate[field]
        if not isinstance(mapping, dict):
            raise TypeError(f"aggregate field {field} must remain a mapping")
        mapping[layer] = value


def _finalize_layer_codec_assignment(
    row: dict[str, object],
    *,
    source_key: str,
    output_key: str,
    distribution_key: str,
) -> None:
    assignment = row.pop(source_key, {})
    if not isinstance(assignment, dict):
        raise TypeError(f"aggregate field {source_key} must be a mapping")
    ordered = {str(layer): assignment[layer] for layer in sorted(assignment)}
    counts: dict[str, int] = {}
    for value in ordered.values():
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    row[output_key] = json.dumps(
        ordered,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    row[distribution_key] = json.dumps(
        counts,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def finalize_aggregate(aggregate: dict[str, object], *, rate_tolerance: float, rho_threshold: float) -> dict[str, object]:
    row = dict(aggregate)
    for source_key, output_key, distribution_key in (
        ("_q_bits_by_layer", "q_bits_by_layer", "q_bits_distribution"),
        (
            "_q_quantizers_by_layer",
            "q_quantizers_by_layer",
            "q_quantizer_distribution",
        ),
        (
            "_q_group_sizes_by_layer",
            "q_group_sizes_by_layer",
            "q_group_size_distribution",
        ),
        (
            "_lowrank_factor_bits_by_layer",
            "lowrank_factor_bits_by_layer",
            "lowrank_factor_bits_distribution",
        ),
    ):
        _finalize_layer_codec_assignment(
            row,
            source_key=source_key,
            output_key=output_key,
            distribution_key=distribution_key,
        )
    reference_bits = max(float(row["reference_bits"]), 1.0)
    payload_bits = float(row["payload_bits"])
    target = float(row["target_ratio"])
    baseline_energy = max(float(row["baseline_hessian_energy"]), EPS)
    row["payload_ratio"] = payload_bits / reference_bits
    row["compression_ratio"] = reference_bits / max(payload_bits, 1.0)
    row["target_gap"] = float(row["payload_ratio"]) - target
    row["under_target"] = float(row["payload_ratio"]) <= target + 1e-12
    row["strict_rate_match"] = abs(float(row["target_gap"])) <= float(rate_tolerance) * target
    comparison_budget = int(float(row.get("comparison_budget_bits", 0)))
    row["rate_cap_satisfied"] = payload_bits <= comparison_budget if comparison_budget > 0 else "not_applicable"
    row["normalized_hessian_cost"] = 2.0 * float(row["hessian_cost"]) / baseline_energy
    row["activation_reconstruction_error"] = float(row["activation_error_weighted"]) / baseline_energy
    row.pop("activation_error_weighted", None)
    for left, right, cross_key in (
        ("hessian_self_q", "hessian_self_s", "hessian_cross_qs"),
        ("hessian_self_q", "hessian_self_l", "hessian_cross_ql"),
        ("hessian_self_s", "hessian_self_l", "hessian_cross_sl"),
    ):
        suffix = cross_key.rsplit("_", 1)[-1]
        rho = _hessian_rho(float(row[cross_key]), float(row[left]), float(row[right]))
        row[f"rho_{suffix}"] = rho
        row[f"rho_{suffix}_kind"] = _rho_kind(rho, rho_threshold)
    q_self = float(row["hessian_self_q"])
    row["cancellation_gain_qs_over_q"] = (
        -float(row["hessian_cross_qs"]) / q_self if float(row["hessian_self_s"]) > EPS and q_self > EPS else float("nan")
    )
    row["cancellation_gain_ql_over_q"] = (
        -float(row["hessian_cross_ql"]) / q_self if float(row["hessian_self_l"]) > EPS and q_self > EPS else float("nan")
    )
    row["heldout_evaluated"] = False
    row["heldout_nll"] = float("nan")
    row["heldout_perplexity"] = float("nan")
    row["nll_delta"] = float("nan")
    row["perplexity_delta"] = float("nan")
    return row


def add_parameter_efficiency(endpoint_rows: list[dict[str, object]]) -> None:
    by_target = {(float(row["target_ratio"]), str(row["strategy"])): row for row in endpoint_rows}
    for row in endpoint_rows:
        q = by_target.get((float(row["target_ratio"]), "Q"))
        if q is None:
            continue
        added_bits = int(float(row["payload_bits"]) - float(q["payload_bits"]))
        gain = float(q["hessian_cost"]) - float(row["hessian_cost"])
        row["q_reference_payload_bits"] = int(float(q["payload_bits"]))
        row["q_reference_hessian_cost"] = float(q["hessian_cost"])
        row["added_bits_over_q"] = added_bits
        row["hessian_gain_over_q"] = gain
        row["hessian_gain_per_added_bit"] = gain / added_bits if added_bits > 0 else float("nan")
        row["folded_zero_bit_hessian_gain"] = (
            gain if added_bits <= 0 and int(float(row.get("folded_repair_dof", 0))) > 0 else 0.0
        )
        row["hessian_recovery_fraction_over_q"] = gain / max(float(q["hessian_cost"]), EPS)


def validate_endpoint_serialized_rate_cap(
    endpoint_candidates: dict[str, dict[str, Candidate]],
    *,
    alignment: int,
) -> dict[str, int]:
    """Fail before task evaluation if the aggregate QSL file exceeds QL."""

    strategies = [
        "Q+L",
        "Q+S+L_QL_budget",
        "Q+S+L_QL_budget_component_scale",
    ]
    strategies.extend(
        strategy
        for strategy in GLOBAL_CONTROL_STRATEGIES
        if endpoint_candidates.get(strategy)
    )
    sizes: dict[str, int] = {}
    expected_layers: set[str] | None = None
    for strategy in strategies:
        candidates = endpoint_candidates.get(strategy, {})
        if not candidates:
            raise AssertionError(f"serialized cap validation is missing {strategy}")
        layer_keys = set(candidates)
        if any(candidate.layer != layer for layer, candidate in candidates.items()):
            raise AssertionError(f"{strategy} candidate keys do not match candidate.layer")
        if expected_layers is None:
            expected_layers = layer_keys
        elif layer_keys != expected_layers:
            missing = sorted(expected_layers - layer_keys)
            extra = sorted(layer_keys - expected_layers)
            raise AssertionError(
                f"serialized cap validation layer-set mismatch for {strategy}: "
                f"missing={missing}, extra={extra}"
            )
        layers = [_artifact_layer(candidates[layer]) for layer in sorted(candidates)]
        sizes[strategy] = codec_artifact_natural_file_bytes(layers, alignment=alignment)
    ql_bytes = sizes["Q+L"]
    for strategy in strategies[1:]:
        if sizes[strategy] > ql_bytes:
            raise AssertionError(
                f"{strategy} aggregate natural artifact ({sizes[strategy]}) exceeds Q+L ({ql_bytes})"
            )
    return sizes


def emit_endpoint_codec_artifacts(
    output_dir: Path,
    *,
    baseline_weights: dict[str, torch.Tensor],
    endpoint_candidates: dict[str, dict[str, Candidate]],
    endpoint_rows: list[dict[str, object]],
    endpoint_target: float,
    alignment: int,
    enforce_serialized_rate_cap: bool,
    rate_allocation: str = "local_guard",
) -> list[dict[str, object]]:
    """Serialize every endpoint, decode it, and attach real-byte evidence."""

    expected_layers = set(baseline_weights)
    if not expected_layers:
        raise AssertionError("codec artifact emission requires non-empty baseline weights")
    for strategy, candidates in endpoint_candidates.items():
        if not candidates:
            continue
        layer_keys = set(candidates)
        if layer_keys != expected_layers:
            missing = sorted(expected_layers - layer_keys)
            extra = sorted(layer_keys - expected_layers)
            raise AssertionError(
                f"codec artifact layer-set mismatch for {strategy}: missing={missing}, extra={extra}"
            )
        if any(candidate.layer != layer for layer, candidate in candidates.items()):
            raise AssertionError(f"{strategy} candidate keys do not match candidate.layer")

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reference_path = artifact_dir / "selected_linear_fp16_reference.hrc"
    reference_weights = {
        layer: tensor.detach().cpu().float().numpy() for layer, tensor in baseline_weights.items()
    }
    reference_result = write_fp16_reference_artifact(
        reference_path,
        reference_weights,
        alignment=alignment,
    )
    reference_decoded = read_fp16_reference_artifact(reference_path)
    for layer, weight in reference_weights.items():
        expected = np.asarray(weight, dtype=np.float16).astype(np.float32)
        if not np.array_equal(reference_decoded.layers[layer], expected):
            raise AssertionError(f"FP16 reference artifact roundtrip changed {layer}")

    layers_by_strategy = {
        strategy: [_artifact_layer(candidates[layer]) for layer in sorted(candidates)]
        for strategy, candidates in endpoint_candidates.items()
        if candidates
    }
    if "Q+L" not in layers_by_strategy:
        raise AssertionError("serialized rate comparison requires the Q+L endpoint")
    natural_bytes = {
        strategy: codec_artifact_natural_file_bytes(layers, alignment=alignment)
        for strategy, layers in layers_by_strategy.items()
    }
    ql_budget_file_bytes = int(natural_bytes["Q+L"])
    capped_strategies = {
        "Q+S+L_QL_budget",
        "Q+S+L_QL_budget_component_scale",
        *GLOBAL_CONTROL_STRATEGIES,
    }
    row_lookup = {
        str(row["strategy"]): row
        for row in endpoint_rows
        if abs(float(row["target_ratio"]) - float(endpoint_target)) <= 1e-12
    }
    artifact_rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        if strategy not in layers_by_strategy:
            continue
        natural = int(natural_bytes[strategy])
        under_ql_cap: bool | str = "not_applicable"
        target_file_bytes: int | None = None
        if strategy in capped_strategies:
            under_ql_cap = natural <= ql_budget_file_bytes
            if enforce_serialized_rate_cap and not under_ql_cap:
                raise AssertionError(
                    f"{strategy} natural artifact ({natural}) exceeds Q+L ({ql_budget_file_bytes})"
                )
            if under_ql_cap:
                target_file_bytes = ql_budget_file_bytes
        safe_name = "".join(character if character.isalnum() else "_" for character in strategy).strip("_")
        artifact_path = artifact_dir / f"{safe_name}.hrc"
        result = write_codec_artifact(
            artifact_path,
            layers_by_strategy[strategy],
            alignment=alignment,
            target_file_bytes=target_file_bytes,
        )
        decoded = read_codec_artifact(artifact_path)
        for layer, candidate in endpoint_candidates[strategy].items():
            if not np.array_equal(decoded.layers[layer], candidate.final):
                raise AssertionError(f"codec artifact roundtrip changed {strategy}/{layer}")
        physical_exact_match = result.file_bytes == ql_budget_file_bytes
        row: dict[str, object] = {
            "strategy": strategy,
            "target_ratio": endpoint_target,
            "artifact_path": artifact_path.relative_to(output_dir).as_posix(),
            "artifact_sha256": result.sha256,
            "artifact_file_bytes": result.file_bytes,
            "artifact_natural_file_bytes": natural,
            "artifact_logical_payload_bits": result.logical_payload_bits,
            "artifact_stream_bytes": result.stream_bytes,
            "artifact_container_bytes": result.container_bytes,
            "artifact_alignment_padding_bytes": result.alignment_padding_bytes,
            "artifact_tail_padding_bytes": result.tail_padding_bytes,
            "artifact_total_overhead_bytes": result.file_bytes
            - int(math.ceil(result.logical_payload_bits / 8.0)),
            "reference_artifact_file_bytes": reference_result.file_bytes,
            "artifact_to_reference_file_ratio": result.file_bytes / max(reference_result.file_bytes, 1),
            "artifact_physical_compression_ratio": reference_result.file_bytes / max(result.file_bytes, 1),
            "ql_budget_file_bytes": ql_budget_file_bytes,
            "under_ql_serialized_cap_before_padding": under_ql_cap,
            "same_physical_bytes_as_ql": physical_exact_match,
            "roundtrip_exact_fp16_endpoint": True,
            "artifact_scope": "selected_linear_weights_only",
            "production_backend": False,
        }
        artifact_rows.append(row)
        if strategy in row_lookup:
            row_lookup[strategy].update(row)

    manifest = {
        "format": "llm_spectral_dynamics_research_codec",
        "scope": "selected_linear_weights_only",
        "production_backend": False,
        "alignment_bytes": int(alignment),
        "serialized_rate_cap_enforced": bool(enforce_serialized_rate_cap),
        "rate_cap_policy": (
            (
                "global enumerated Pareto allocation is checked with the complete multi-layer serializer; "
                "a feasible natural artifact is padded to exact Q+L file bytes"
                if rate_allocation == "global_exact"
                else "per-layer QSL search reserves one alignment unit below corresponding Q+L; "
                "aggregate QSL is checked then padded to exact Q+L bytes"
            )
            if enforce_serialized_rate_cap
            else "measurement only"
        ),
        "reference": {
            "path": reference_path.relative_to(output_dir).as_posix(),
            "sha256": reference_result.sha256,
            "file_bytes": reference_result.file_bytes,
            "logical_payload_bits": reference_result.logical_payload_bits,
            "roundtrip_exact_fp16": True,
        },
        "strategies": artifact_rows,
    }
    base.write_csv(output_dir / "artifact_payloads.csv", artifact_rows)
    base.write_json(output_dir / "artifact_manifest.json", manifest)
    return artifact_rows


def annotate_joint_value_claim(
    endpoint_rows: list[dict[str, object]],
    *,
    endpoint_target: float,
    rate_allocator_report: dict[str, object],
) -> dict[str, object]:
    """Gate the joint-value claim on equal natural bytes and final-test NLL."""

    row_lookup = {
        str(row["strategy"]): row
        for row in endpoint_rows
        if abs(float(row["target_ratio"]) - float(endpoint_target)) <= 1e-12
    }
    qsl = row_lookup.get("Q+S+L_QL_budget")
    nojoint = row_lookup.get(GLOBAL_NONJOINT_CONTROL_STRATEGY)
    if qsl is None or nojoint is None:
        result = {
            "evaluated": False,
            "supported": False,
            "reason": "QSL or no-joint endpoint is unavailable",
        }
        rate_allocator_report["joint_value_claim"] = result
        return result

    qsl_natural = qsl.get("artifact_natural_file_bytes")
    nojoint_natural = nojoint.get("artifact_natural_file_bytes")
    same_natural_bytes = (
        qsl_natural is not None
        and nojoint_natural is not None
        and int(qsl_natural) == int(nojoint_natural)
    )
    qsl_physical = qsl.get("artifact_file_bytes")
    nojoint_physical = nojoint.get("artifact_file_bytes")
    same_physical_bytes = (
        qsl_physical is not None
        and nojoint_physical is not None
        and int(qsl_physical) == int(nojoint_physical)
    )
    test_evaluated = bool(qsl.get("heldout_evaluated")) and bool(
        nojoint.get("heldout_evaluated")
    )
    test_gain = (
        float(nojoint["heldout_nll"]) - float(qsl["heldout_nll"])
        if test_evaluated
        else float("nan")
    )
    qsl_wins_test = test_evaluated and test_gain > 0.0
    exact_match_search_succeeded = (
        rate_allocator_report.get("joint_control_natural_match_available") is True
    )
    supported = exact_match_search_succeeded and same_natural_bytes and qsl_wins_test
    if not exact_match_search_succeeded:
        reason = (
            "the exact-natural no-joint counterfactual search did not complete "
            "with a matched endpoint"
        )
    elif not same_natural_bytes:
        reason = "natural serialized bytes differ; padded equality is insufficient"
    elif not test_evaluated:
        reason = "final test NLL was not evaluated for both endpoints"
    elif not qsl_wins_test:
        reason = "QSL did not beat the no-joint control on final test NLL"
    else:
        reason = "QSL beat no-joint at identical natural serialized bytes on final test"
    result = {
        "evaluated": bool(test_evaluated and qsl_natural is not None and nojoint_natural is not None),
        "supported": supported,
        "reason": reason,
        "exact_natural_match_search_succeeded": exact_match_search_succeeded,
        "same_natural_file_bytes": same_natural_bytes,
        "same_physical_file_bytes_after_padding": same_physical_bytes,
        "qsl_natural_file_bytes": qsl_natural,
        "nojoint_natural_file_bytes": nojoint_natural,
        "qsl_test_nll": qsl.get("heldout_nll"),
        "nojoint_test_nll": nojoint.get("heldout_nll"),
        "qsl_test_nll_gain_over_nojoint": test_gain,
    }
    for row in (qsl, nojoint):
        row.update(
            {
                "joint_counterfactual_strategy": GLOBAL_NONJOINT_CONTROL_STRATEGY,
                "joint_same_natural_file_bytes": same_natural_bytes,
                "joint_same_physical_file_bytes_after_padding": same_physical_bytes,
                "joint_exact_natural_match_search_succeeded": (
                    exact_match_search_succeeded
                ),
                "joint_test_nll_gain_over_nojoint": test_gain,
                "joint_value_claim_supported": supported,
                "joint_value_claim_reason": reason,
            }
        )
    rate_allocator_report["joint_value_claim"] = result
    return result


def evaluate_current_model_with_windows(
    model: torch.nn.Module,
    tokenizer: object,
    *,
    strategy: str,
    texts: list[str],
    sequence_length: int,
    batch_size: int,
    device: str,
    eval_limit: int,
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    """Evaluate aggregate NLL while retaining fixed-window paired evidence."""

    nll_total = 0.0
    token_total = 0
    window_rows: list[dict[str, object]] = []
    window_index = 0
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(
            base.token_batches(
                tokenizer,
                texts,
                sequence_length=sequence_length,
                batch_size=batch_size,
                limit=eval_limit,
            )
        ):
            batch = batch.to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch[:, 1:]
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            for sequence_index in range(token_losses.shape[0]):
                window_sum = float(token_losses[sequence_index].sum().detach().cpu())
                window_tokens = int(token_losses.shape[1])
                window_nll = window_sum / max(window_tokens, 1)
                window_rows.append(
                    {
                        "strategy": strategy,
                        "window_index": window_index,
                        "batch_index": batch_index,
                        "sequence_index": sequence_index,
                        "tokens": window_tokens,
                        "nll_sum": window_sum,
                        "nll": window_nll,
                        "perplexity": float(math.exp(min(window_nll, 50.0))),
                    }
                )
                window_index += 1
                nll_total += window_sum
                token_total += window_tokens
    mean_nll = nll_total / max(token_total, 1)
    return (
        {
            "nll": mean_nll,
            "perplexity": float(math.exp(min(mean_nll, 50.0))),
            "tokens": token_total,
        },
        window_rows,
    )


def rerank_global_allocations_by_validation(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    modules: dict[str, torch.nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    ranked_by_strategy: dict[str, list[RankedGlobalAllocation]],
    args: argparse.Namespace,
    validation_baseline_metrics: dict[str, float | int] | None = None,
) -> tuple[
    dict[str, dict[str, Candidate]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, float | int],
    dict[str, dict[str, object]],
]:
    """Rerank proxy-screened allocations on validation without touching test."""

    if not args.two_stage_selection:
        raise ValueError("validation reranking requires --two-stage-selection")
    if not ranked_by_strategy:
        raise ValueError("validation reranking requires at least one global strategy")
    if validation_baseline_metrics is None:
        baseline_metrics, baseline_window_rows = evaluate_current_model_with_windows(
            model,
            tokenizer,
            strategy="dense_validation",
            texts=args.selection_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            eval_limit=args.selection_limit,
        )
        for row in baseline_window_rows:
            row.update(
                {
                    "evidence_role": "allocation_validation",
                    "proxy_rank": 0,
                    "allocation_digest": "dense",
                }
            )
        selection_rows: list[dict[str, object]] = [
            {
                "strategy": "dense_validation",
                "proxy_rank": 0,
                "allocation_digest": "dense",
                "hessian_cost": 0.0,
                "natural_file_bytes": 0,
                "validation_nll": float(baseline_metrics["nll"]),
                "validation_perplexity": float(baseline_metrics["perplexity"]),
                "validation_tokens": int(baseline_metrics["tokens"]),
                "validation_nll_delta": 0.0,
                "selected_by_validation": False,
                "evidence_role": "allocation_validation_baseline",
            }
        ]
        all_window_rows = list(baseline_window_rows)
    else:
        baseline_metrics = dict(validation_baseline_metrics)
        selection_rows = []
        all_window_rows = []
    selected_by_strategy: dict[str, dict[str, Candidate]] = {}
    selection_reports: dict[str, dict[str, object]] = {}
    expected_layers = set(modules)

    for strategy in STRATEGY_ORDER:
        allocations = ranked_by_strategy.get(strategy)
        if not allocations:
            continue
        evaluated: list[
            tuple[float, float, int, int, RankedGlobalAllocation, dict[str, float | int]]
        ] = []
        strategy_rows: list[dict[str, object]] = []
        for proxy_rank, allocation in enumerate(allocations, start=1):
            if set(allocation.candidates) != expected_layers:
                raise AssertionError(
                    f"validation allocation layer-set mismatch for {strategy}"
                )
            replacements = {
                layer: torch.from_numpy(candidate.final).to(
                    dtype=baseline_weights[layer].dtype
                )
                for layer, candidate in allocation.candidates.items()
            }
            base.restore_weights(modules, baseline_weights)
            base.apply_replacements(modules, replacements)
            try:
                metrics, window_rows = evaluate_current_model_with_windows(
                    model,
                    tokenizer,
                    strategy=f"{strategy}__proxy_rank_{proxy_rank}",
                    texts=args.selection_texts,
                    sequence_length=args.sequence_length,
                    batch_size=args.batch_size,
                    device=args.device,
                    eval_limit=args.selection_limit,
                )
            finally:
                base.restore_weights(modules, baseline_weights)
            for row in window_rows:
                row.update(
                    {
                        "evidence_role": "allocation_validation",
                        "base_strategy": strategy,
                        "proxy_rank": proxy_rank,
                        "allocation_digest": allocation.allocation_digest,
                    }
                )
            all_window_rows.extend(window_rows)
            validation_nll = float(metrics["nll"])
            evaluated.append(
                (
                    validation_nll,
                    allocation.hessian_cost,
                    allocation.natural_file_bytes,
                    proxy_rank,
                    allocation,
                    metrics,
                )
            )
            row = {
                "strategy": strategy,
                "proxy_rank": proxy_rank,
                "allocation_digest": allocation.allocation_digest,
                "hessian_cost": allocation.hessian_cost,
                "natural_file_bytes": allocation.natural_file_bytes,
                "validation_nll": validation_nll,
                "validation_perplexity": float(metrics["perplexity"]),
                "validation_tokens": int(metrics["tokens"]),
                "validation_nll_delta": validation_nll
                - float(baseline_metrics["nll"]),
                "selected_by_validation": False,
                "evidence_role": "allocation_validation_candidate",
            }
            strategy_rows.append(row)
            selection_rows.append(row)

        winner = min(evaluated, key=lambda item: item[:4])
        (
            winner_nll,
            winner_cost,
            winner_natural,
            winner_proxy_rank,
            winner_allocation,
            winner_metrics,
        ) = winner
        strategy_rows[winner_proxy_rank - 1]["selected_by_validation"] = True
        selected_by_strategy[strategy] = winner_allocation.candidates
        selection_reports[strategy] = {
            "selection_source": "validation_nll_rerank_of_exact_proxy_top_k",
            "proxy_top_k_evaluated": len(allocations),
            "validation_selected_proxy_rank": winner_proxy_rank,
            "validation_selected_allocation_digest": winner_allocation.allocation_digest,
            "validation_selected_hessian_cost": winner_cost,
            "validation_selected_natural_file_bytes": winner_natural,
            "validation_selected_nll": winner_nll,
            "validation_selected_perplexity": float(winner_metrics["perplexity"]),
            "validation_selected_tokens": int(winner_metrics["tokens"]),
            "validation_selected_nll_delta": winner_nll
            - float(baseline_metrics["nll"]),
        }
    return (
        selected_by_strategy,
        selection_rows,
        all_window_rows,
        baseline_metrics,
        selection_reports,
    )


def evaluate_endpoint_strategies(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    modules: dict[str, torch.nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, dict[str, torch.Tensor]],
    endpoint_rows: list[dict[str, object]],
    endpoint_target: float,
    baseline_metrics: dict[str, float | int],
    baseline_window_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    row_lookup = {
        str(row["strategy"]): row
        for row in endpoint_rows
        if abs(float(row["target_ratio"]) - float(endpoint_target)) <= 1e-12
    }
    if args.proxy_only:
        return []
    all_window_rows: list[dict[str, object]] = []
    baseline_by_window = {int(row["window_index"]): float(row["nll"]) for row in baseline_window_rows}
    for strategy in STRATEGY_ORDER:
        if strategy not in replacements or strategy not in row_lookup:
            continue
        base.restore_weights(modules, baseline_weights)
        base.apply_replacements(modules, replacements[strategy])
        try:
            metrics, window_rows = evaluate_current_model_with_windows(
                model,
                tokenizer,
                strategy=strategy,
                texts=args.eval_texts,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                device=args.device,
                eval_limit=args.eval_limit,
            )
        finally:
            base.restore_weights(modules, baseline_weights)
        row = row_lookup[strategy]
        row["heldout_evaluated"] = True
        row["heldout_nll"] = float(metrics["nll"])
        row["heldout_perplexity"] = float(metrics["perplexity"])
        row["heldout_tokens"] = int(metrics["tokens"])
        row["nll_delta"] = float(metrics["nll"]) - float(baseline_metrics["nll"])
        row["perplexity_delta"] = float(metrics["perplexity"]) - float(baseline_metrics["perplexity"])
        paired_deltas = np.asarray(
            [float(item["nll"]) - baseline_by_window[int(item["window_index"])] for item in window_rows],
            dtype=np.float64,
        )
        for item in window_rows:
            item["evidence_role"] = (
                "final_test" if args.two_stage_selection else "endpoint_evaluation"
            )
        paired_mean = float(np.mean(paired_deltas)) if paired_deltas.size else float("nan")
        paired_se = (
            float(np.std(paired_deltas, ddof=1) / math.sqrt(paired_deltas.size))
            if paired_deltas.size > 1
            else float("nan")
        )
        row["paired_window_count"] = int(paired_deltas.size)
        row["paired_window_nll_delta_mean"] = paired_mean
        row["paired_window_nll_delta_se"] = paired_se
        row["paired_window_nll_delta_ci95_low"] = paired_mean - 1.96 * paired_se
        row["paired_window_nll_delta_ci95_high"] = paired_mean + 1.96 * paired_se
        all_window_rows.extend(window_rows)
    return all_window_rows


def evaluate_comfort_paths(
    *,
    model: torch.nn.Module,
    tokenizer: object,
    modules: dict[str, torch.nn.Linear],
    baseline_weights: dict[str, torch.Tensor],
    replacements: dict[str, dict[str, torch.Tensor]],
    endpoint_rows: list[dict[str, object]],
    endpoint_target: float,
    baseline_metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if args.proxy_only or args.skip_comfort:
        return [], []
    comfort_texts = getattr(args, "comfort_texts", args.eval_texts)
    comfort_eval_limit = int(getattr(args, "comfort_eval_limit", args.eval_limit))
    comfort_role = str(getattr(args, "comfort_evidence_role", "endpoint_evaluation"))
    same_as_endpoint_evaluation = (
        comfort_texts is args.eval_texts and comfort_eval_limit == int(args.eval_limit)
    )
    if same_as_endpoint_evaluation:
        comfort_baseline_metrics = baseline_metrics
    else:
        comfort_baseline_metrics, _comfort_baseline_rows = evaluate_current_model_with_windows(
            model,
            tokenizer,
            strategy="dense_comfort_baseline",
            texts=comfort_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            eval_limit=comfort_eval_limit,
        )
    row_lookup = {
        str(row["strategy"]): row
        for row in endpoint_rows
        if abs(float(row["target_ratio"]) - float(endpoint_target)) <= 1e-12
    }
    strategies = [item for item in args.comfort_strategies if item in replacements and item in row_lookup]
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        endpoint = row_lookup[strategy]
        for epsilon in args.comfort_epsilons:
            value = float(epsilon)
            if abs(value) <= 1e-12:
                metrics = comfort_baseline_metrics
            elif (
                same_as_endpoint_evaluation
                and abs(value - 1.0) <= 1e-12
                and bool(endpoint.get("heldout_evaluated"))
            ):
                metrics = {
                    "nll": endpoint["heldout_nll"],
                    "perplexity": endpoint["heldout_perplexity"],
                    "tokens": endpoint.get("heldout_tokens", baseline_metrics["tokens"]),
                }
            else:
                interpolated = {
                    layer: (
                        baseline_weights[layer].float()
                        + value * (replacements[strategy][layer].float() - baseline_weights[layer].float())
                    ).to(dtype=baseline_weights[layer].dtype)
                    for layer in modules
                }
                base.restore_weights(modules, baseline_weights)
                base.apply_replacements(modules, interpolated)
                try:
                    metrics, _comfort_rows = evaluate_current_model_with_windows(
                        model,
                        tokenizer,
                        strategy=f"{strategy}@epsilon={value:g}",
                        texts=comfort_texts,
                        sequence_length=args.sequence_length,
                        batch_size=args.batch_size,
                        device=args.device,
                        eval_limit=comfort_eval_limit,
                    )
                finally:
                    base.restore_weights(modules, baseline_weights)
            rows.append(
                {
                    "strategy": strategy,
                    "target_ratio": endpoint_target,
                    "payload_ratio_at_codec_endpoint": endpoint["payload_ratio"],
                    "epsilon": value,
                    "path_kind": "codec_endpoint" if abs(value - 1.0) <= 1e-12 else "noncodec_interpolation",
                    "deployable": abs(value - 1.0) <= 1e-12,
                    "evidence_role": comfort_role,
                    "hessian_cost": value * value * float(endpoint["hessian_cost"]),
                    "normalized_hessian_cost": value * value * float(endpoint["normalized_hessian_cost"]),
                    "nll": float(metrics["nll"]),
                    "perplexity": float(metrics["perplexity"]),
                    "tokens": int(metrics["tokens"]),
                    "nll_delta": float(metrics["nll"]) - float(comfort_baseline_metrics["nll"]),
                    "perplexity_delta": float(metrics["perplexity"])
                    - float(comfort_baseline_metrics["perplexity"]),
                }
            )

    summaries: list[dict[str, object]] = []
    for strategy in strategies:
        group = sorted((row for row in rows if row["strategy"] == strategy), key=lambda item: float(item["epsilon"]))
        fit_rows = [row for row in group if 0.0 < float(row["epsilon"]) <= float(args.comfort_fit_max_epsilon)]
        if len(fit_rows) >= 2:
            eps = np.asarray([float(row["epsilon"]) for row in fit_rows], dtype=np.float64)
            values = np.asarray([float(row["nll_delta"]) for row in fit_rows], dtype=np.float64)
            design = np.stack([eps, eps * eps], axis=1)
            linear, quadratic = np.linalg.lstsq(design, values, rcond=None)[0]
        else:
            linear, quadratic = float("nan"), float("nan")
        max_comfort = 0.0
        prefix_ok = True
        for row in group:
            epsilon = float(row["epsilon"])
            predicted = linear * epsilon + quadratic * epsilon * epsilon
            actual = float(row["nll_delta"])
            tolerance = max(
                float(args.comfort_absolute_tolerance),
                float(args.comfort_relative_tolerance) * max(abs(actual), abs(predicted), EPS),
            )
            error = abs(actual - predicted)
            passed = bool(math.isfinite(predicted) and error <= tolerance)
            row["taylor_fit_nll_delta"] = predicted
            row["taylor_fit_absolute_error"] = error
            row["comfort_tolerance"] = tolerance
            row["within_local_comfort_fit"] = passed
            if prefix_ok and passed:
                max_comfort = epsilon
            elif epsilon > 0.0:
                prefix_ok = False
        endpoint = max(group, key=lambda item: float(item["epsilon"]))
        proxy = np.asarray([float(row["normalized_hessian_cost"]) for row in group], dtype=np.float64)
        task = np.asarray([float(row["nll_delta"]) for row in group], dtype=np.float64)
        if len(group) >= 2 and np.std(proxy) > EPS and np.std(task) > EPS:
            proxy_task_corr = float(np.corrcoef(proxy, task)[0, 1])
        else:
            proxy_task_corr = float("nan")
        summaries.append(
            {
                "strategy": strategy,
                "target_ratio": endpoint_target,
                "small_epsilon_fit_max": args.comfort_fit_max_epsilon,
                "taylor_linear_coefficient": linear,
                "taylor_quadratic_coefficient": quadratic,
                "max_contiguous_comfort_epsilon": max_comfort,
                "codec_endpoint_within_comfort_fit": bool(endpoint["within_local_comfort_fit"]),
                "codec_endpoint_nll_delta": endpoint["nll_delta"],
                "codec_endpoint_fit_error": endpoint["taylor_fit_absolute_error"],
                "hessian_proxy_nll_correlation": proxy_task_corr,
                "interpretation": (
                    "local_fit_reaches_codec_endpoint"
                    if bool(endpoint["within_local_comfort_fit"])
                    else "local_comfort_zone_does_not_certify_codec_endpoint"
                ),
            }
        )
    return rows, summaries


def plot_results(
    output_dir: Path,
    endpoint_rows: list[dict[str, object]],
    comfort_rows: list[dict[str, object]],
    *,
    endpoint_target: float,
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {strategy: plt.cm.tab10(index % 10) for index, strategy in enumerate(STRATEGY_ORDER)}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)

    ax = axes[0, 0]
    for strategy in STRATEGY_ORDER:
        group = sorted(
            (row for row in endpoint_rows if row["strategy"] == strategy),
            key=lambda row: float(row["payload_ratio"]),
        )
        if group:
            ax.plot(
                [float(row["payload_ratio"]) for row in group],
                [float(row["normalized_hessian_cost"]) for row in group],
                marker="o",
                linewidth=1.4,
                label=_plot_strategy_label(strategy),
                color=colors[strategy],
            )
    ax.set_xlabel("real payload / FP16 reference")
    ax.set_ylabel("normalized Hessian cost")
    ax.set_title("(a) Exact-rate local distortion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    evaluated = [
        row
        for row in endpoint_rows
        if abs(float(row["target_ratio"]) - endpoint_target) <= 1e-12 and bool(row.get("heldout_evaluated"))
    ]
    ax.barh(
        np.arange(len(evaluated)),
        [float(row["perplexity_delta"]) for row in evaluated],
        color=[colors[str(row["strategy"])] for row in evaluated],
    )
    ax.set_yticks(
        np.arange(len(evaluated)),
        [_plot_strategy_label(row["strategy"]) for row in evaluated],
        fontsize=8,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("held-out PPL delta")
    ax.set_title(f"(b) Codec endpoints at target {endpoint_target:.3f}")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1, 0]
    for strategy in STRATEGY_ORDER:
        group = sorted(
            (row for row in comfort_rows if row["strategy"] == strategy),
            key=lambda row: float(row["epsilon"]),
        )
        if group:
            ax.plot(
                [float(row["epsilon"]) for row in group],
                [float(row["nll_delta"]) for row in group],
                marker="o",
                linewidth=1.4,
                label=_plot_strategy_label(strategy),
                color=colors[strategy],
            )
            ax.plot(
                [float(row["epsilon"]) for row in group],
                [float(row.get("taylor_fit_nll_delta", float("nan"))) for row in group],
                linestyle="--",
                linewidth=0.8,
                color=colors[strategy],
                alpha=0.75,
            )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("epsilon along compression perturbation")
    ax.set_ylabel("held-out NLL delta")
    ax.set_title("(c) Loss landscape: measured (solid), local fit (dashed)")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    endpoint = [
        row for row in endpoint_rows if abs(float(row["target_ratio"]) - endpoint_target) <= 1e-12
    ]
    x = np.arange(len(endpoint))
    width = 0.25
    for offset, pair in zip((-width, 0.0, width), ("qs", "ql", "sl")):
        ax.bar(
            x + offset,
            [float(row.get(f"rho_{pair}", float("nan"))) for row in endpoint],
            width=width,
            label=pair.upper(),
        )
    ax.axhspan(-0.1, 0.1, color="gray", alpha=0.15, label="|rho| <= 0.1")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(
        x,
        [_plot_strategy_label(row["strategy"]) for row in endpoint],
        rotation=35,
        ha="right",
        fontsize=7,
    )
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("signed Hessian cosine")
    ax.set_title("(d) Orthogonality vs cancellation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"pretrained_hessian_repair_probe.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    baseline_metrics: dict[str, float | int],
    endpoint_rows: list[dict[str, object]],
    comfort_summary: list[dict[str, object]],
) -> None:
    endpoint = [
        row for row in endpoint_rows if abs(float(row["target_ratio"]) - args.endpoint_target) <= 1e-12
    ]
    payload_scope_line = (
        "- Payload is serialized for the selected tensors in deterministic research artifacts: packed Q codes + "
        "FP16 scales + FP16 sparse values + fixed-width CSR support + packed 4/8-bit or FP16 low-rank "
        "factors with their stored scales, including the "
        f"manifest, descriptors and {int(args.artifact_alignment)}-byte alignment. This is byte-audit evidence, not a production inference backend."
        if args.emit_codec_artifacts
        else "- Payload covers selected tensor value streams only; no serialized-byte claim is available because "
        "artifact emission was disabled."
    )
    data_scope_line = (
        f"- Data: `{args.text_source_used}` with independent train calibration, "
        "validation allocation selection and final-test splits; exact duplicate content "
        "is removed across roles and test is reserved until validation selection completes."
        if args.two_stage_selection
        else f"- Data: `{args.text_source_used}` with content-disjoint calibration/evaluation "
        "source texts; fallback is disabled."
    )
    lines = [
        "# Pretrained exact-rate Hessian repair probe",
        "",
        "## Scope and design expectations",
        "",
        f"- Model: `{args.model}`; selected tensors: {len(args.selected_layers)} MLP linears.",
        data_scope_line,
        payload_scope_line,
        "- Expected before running: block scales should buy Hessian reduction per extra scale bit; OBS should make the frozen sparse support stationary; a Q/S/L combination should win only when its marginal Hessian reduction per real bit exceeds metadata overhead and its component directions are complementary.",
        "- The Hessian proxy is activation MSE (`C ⊗ I_out`), not the full task Hessian. Held-out NLL/PPL is therefore the endpoint arbiter.",
        "",
        "## Held-out baseline",
        "",
        f"NLL = {_format_float(baseline_metrics['nll'])}, PPL = {_format_float(baseline_metrics['perplexity'])}, tokens = {int(baseline_metrics['tokens'])}.",
        "",
        f"## Codec endpoints near target {args.endpoint_target:.3f}",
        "",
        "| strategy | logical value ratio | artifact/reference ratio | file bytes | logical target match | norm. H cost | H gain / added bit | rho(S,L) | cancel Q<-S | cancel Q<-L | PPL delta |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint:
        lines.append(
            "| {strategy} | {payload} | {artifact_ratio} | {file_bytes} | {match} | {cost} | {eff} | {rsl} | {cqs} | {cql} | {ppl} |".format(
                strategy=row["strategy"],
                payload=_format_float(row["payload_ratio"]),
                artifact_ratio=_format_float(row.get("artifact_to_reference_file_ratio")),
                file_bytes=(
                    str(int(float(row["artifact_file_bytes"])))
                    if "artifact_file_bytes" in row
                    else "n/a"
                ),
                match="yes" if bool(row["strict_rate_match"]) else "no",
                cost=_format_float(row["normalized_hessian_cost"]),
                eff=_format_float(row.get("hessian_gain_per_added_bit"), 9),
                rsl=_format_float(row.get("rho_sl")),
                cqs=_format_float(row.get("cancellation_gain_qs_over_q")),
                cql=_format_float(row.get("cancellation_gain_ql_over_q")),
                ppl=_format_float(row.get("perplexity_delta")),
            )
        )
    endpoint_by_strategy = {str(row["strategy"]): row for row in endpoint}
    ql = endpoint_by_strategy.get("Q+L")
    matched = endpoint_by_strategy.get("Q+S+L_QL_budget_component_scale")
    if ql is not None and matched is not None:
        bit_delta = int(float(matched["payload_bits"]) - float(ql["payload_bits"]))
        file_delta = (
            int(float(matched["artifact_file_bytes"]) - float(ql["artifact_file_bytes"]))
            if "artifact_file_bytes" in matched and "artifact_file_bytes" in ql
            else None
        )
        ppl_gain = float(ql["perplexity_delta"]) - float(matched["perplexity_delta"])
        hessian_gain = float(ql["normalized_hessian_cost"]) - float(matched["normalized_hessian_cost"])
        allocation_description = (
            "selected from enumerated per-layer support/rank budget bands by an aggregate "
            "additive frontier, then checked against the complete natural Q+L file-byte cap"
            if args.rate_allocation == "global_exact"
            else "capped independently per layer by the legacy guarded Q+L codec budget"
        )
        natural_delta = (
            int(
                float(matched["artifact_natural_file_bytes"])
                - float(ql["artifact_natural_file_bytes"])
            )
            if "artifact_natural_file_bytes" in matched
            and "artifact_natural_file_bytes" in ql
            else None
        )
        lines.extend(
            [
                "",
                "### Q+L fixed-rate control",
                "",
                f"`Q+S+L_QL_budget_component_scale` is {allocation_description}. "
                f"Aggregate value-stream bit delta versus Q+L = `{bit_delta}`; "
                + (
                    f"serialized file-byte delta = `{file_delta}`; "
                    if file_delta is not None
                    else "serialized file-byte delta was not measured; "
                )
                + (
                    f"natural file-byte delta = `{natural_delta}`; "
                    if natural_delta is not None
                    else "natural file-byte delta was not measured; "
                )
                + f"PPL-delta improvement = `{ppl_gain:.6f}`; "
                f"normalized-Hessian-cost improvement = `{hessian_gain:.9f}`. A positive improvement means the "
                "combination wins without using more measured storage only when the reported file-byte delta is non-positive.",
            ]
        )
    lines.extend(
        [
            "",
            "`rho ≈ 0` means second-order additivity, not that a combination is better. Negative rho is reported as repair/cancellation; positive rho is conflict. A combination has a fixed-rate advantage only if the saved self loss plus favorable cross terms outweigh the real sparse-index/factor/scale payload.",
            "",
            "`endpoint_window_nll.csv` retains paired fixed-window NLL for the dense model and every endpoint. The endpoint CSV reports a descriptive normal 95% interval over those fixed windows; contiguous language-model windows are not independent samples, so this interval is an uncertainty diagnostic rather than a population-level confidence claim.",
            "",
            "## Comfort-zone / loss-landscape check",
            "",
            "Only epsilon = 1 is a deployable codec. Smaller epsilon values diagnose whether each method has a locally comfortable perturbation regime; they are not compressed checkpoints.",
            "",
            "| strategy | max contiguous fitted epsilon | endpoint fitted? | endpoint NLL delta | proxy/NLL correlation |",
            "|---|---:|:---:|---:|---:|",
        ]
    )
    for row in comfort_summary:
        lines.append(
            "| {strategy} | {epsilon} | {endpoint} | {delta} | {corr} |".format(
                strategy=row["strategy"],
                epsilon=_format_float(row["max_contiguous_comfort_epsilon"], 3),
                endpoint="yes" if bool(row["codec_endpoint_within_comfort_fit"]) else "no",
                delta=_format_float(row["codec_endpoint_nll_delta"]),
                corr=_format_float(row["hessian_proxy_nll_correlation"]),
            )
        )
    lines.extend(
        [
            "",
            "## Theory–experiment contract",
            "",
            "For perturbations `d_a,d_b`, the local prediction is `ΔL ≈ ½<d_a,d_a>_H + ½<d_b,d_b>_H + <d_a,d_b>_H`. The CSVs expose every term. OBS and scale repair are tested first by their stationarity/cost identities after FP16 rounding, then by held-out PPL. Disagreement between the proxy and PPL is evidence that the local input-covariance geometry is insufficient at that endpoint, not evidence to overwrite the endpoint result.",
            "",
            "See `candidate_ablation.csv` for discrete payload/allocation choices, `strategy_endpoints.csv` for aggregated comparisons, `comfort_sweep.csv` for the measured loss landscape, and (when emitted) `artifact_manifest.json` plus `artifact_payloads.csv` for independently decoded physical-byte evidence.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", default="")
    parser.add_argument("--model-snapshot-manifest", default="")
    parser.add_argument("--model-snapshot-manifest-sha256", default="")
    parser.add_argument("--model-snapshot-aggregate-sha256", default="")
    parser.add_argument("--resource-gate-manifest", default="")
    parser.add_argument("--output-dir", default="results/pretrained_hessian_repair_pythia70m_20260713")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--svd-device", default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--subset", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--selection-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--backup-name", default="")
    parser.add_argument("--text-source", choices=["dataset"], default="dataset")
    parser.add_argument("--calib-limit", type=int, default=4)
    parser.add_argument("--selection-limit", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--texts-per-batch-window", type=int, default=8)
    parser.add_argument("--selector-activation-sample-rows", type=int, default=256)
    parser.add_argument("--module-types", default="dense_h_to_4h,dense_4h_to_h")
    parser.add_argument("--layer-positions", default="first,middle,last")
    parser.add_argument("--layers", default="")
    parser.add_argument("--max-modules", type=int, default=6)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--candidate-bits",
        default="",
        help="Comma-separated base bit widths admitted to the global heterogeneous pool.",
    )
    parser.add_argument(
        "--candidate-q-group-sizes",
        default="0",
        help="Comma-separated column group sizes; 0 denotes one scale per output row.",
    )
    parser.add_argument(
        "--candidate-quantizers",
        default="symmetric_rtn",
        help="Comma-separated quantizer rules: symmetric_rtn,symmetric_mse_clip.",
    )
    parser.add_argument(
        "--candidate-lowrank-factor-bits",
        default="16",
        help="Comma-separated stored low-rank factor widths; 16 keeps FP16 factors.",
    )
    parser.add_argument(
        "--candidate-family-top-k",
        type=int,
        default=0,
        help=(
            "Cheap Q-only Hessian pre-screen size for expensive quantizer x "
            "low-rank-factor family expansion; 0 keeps the full Cartesian grid."
        ),
    )
    parser.add_argument("--target-ratios", default="0.258,0.275,0.300")
    parser.add_argument("--endpoint-target", type=float, default=0.258)
    parser.add_argument("--support-encoding", choices=["csr_fixed"], default="csr_fixed")
    parser.add_argument("--emit-codec-artifacts", action="store_true")
    parser.add_argument("--enforce-serialized-rate-cap", action="store_true")
    parser.add_argument("--artifact-alignment", type=int, default=64)
    parser.add_argument("--s-method", choices=["wanda", "magnitude"], default="wanda")
    parser.add_argument("--l-method", choices=["whitened_svd", "svd"], default="whitened_svd")
    parser.add_argument(
        "--residual-order",
        choices=["s_then_l", "l_then_s"],
        default="s_then_l",
        help="Fit the sparse or low-rank component first while holding the same byte allocation.",
    )
    parser.add_argument(
        "--covariance-mode",
        choices=["full", "diagonal", "identity"],
        default="full",
        help="Ablate the activation-Gram geometry without changing the serialized codec.",
    )
    parser.add_argument(
        "--covariance-damping-ratio",
        type=float,
        default=0.0,
        help="Add ratio times the selected covariance mean diagonal before fitting repairs.",
    )
    parser.add_argument(
        "--whitening-floor-ratio",
        type=float,
        default=1e-5,
        help="Factorizer-only eigenvalue floor for whitened SVD; the scoring metric is unchanged.",
    )
    parser.add_argument(
        "--lowrank-svd-solver",
        choices=["auto", "full", "randomized"],
        default="auto",
        help="Use exact full SVD or a seeded low-rank randomized solver for each residual fit.",
    )
    parser.add_argument("--lowrank-svd-oversampling", type=int, default=4)
    parser.add_argument("--lowrank-svd-niter", type=int, default=2)
    parser.add_argument(
        "--rate-allocation",
        choices=["local_guard", "global_exact"],
        default="local_guard",
        help="Use legacy per-layer guards or a global exact-file-byte QSL frontier allocator.",
    )
    parser.add_argument(
        "--include-global-single-component-controls",
        action="store_true",
        help=(
            "Add pure global Q+L, pure Q+S_OBS and their per-layer no-joint "
            "union control under the same exact aggregate Q+L physical-file-byte "
            "cap as the global QSL endpoint."
        ),
    )
    parser.add_argument(
        "--two-stage-selection",
        action="store_true",
        help=(
            "Screen exact-file-feasible global allocations by the Hessian proxy, "
            "rerank the proxy top-K on a separate validation split, and reserve "
            "the test split for the selected endpoints only."
        ),
    )
    parser.add_argument(
        "--selection-top-k",
        type=int,
        default=1,
        help="Number of proxy-ranked exact global allocations evaluated on validation.",
    )
    parser.add_argument(
        "--strict-sparse-refit",
        choices=["naive", "obs"],
        default="naive",
        help="Refit the sparse values inside the same-byte strict QSL candidate without changing support.",
    )
    parser.add_argument("--global-frontier-top-ranks", type=int, default=2)
    parser.add_argument(
        "--global-frontier-support-fractions",
        default="0.5,0.75,0.9,0.97",
        help="Additional support sizes for the best local rank states in global allocation mode.",
    )
    parser.add_argument(
        "--global-frontier-budget-multipliers",
        default="1.25,1.5,2.0",
        help=(
            "Multiples (>1) of each layer's Q+L repair allowance beyond base Q, enumerated for the best rank states; "
            "these candidates let the aggregate exact-feasibility search borrow bytes across layers."
        ),
    )
    parser.add_argument("--repair-block-sizes", default="32,64,128,256,512")
    parser.add_argument(
        "--skip-block-scale",
        action="store_true",
        help=(
            "Explicitly omit the row-by-column-block scale baseline. This is a "
            "resource guard for large full-covariance tensors and never emits a "
            "silent Q_block_scale fallback endpoint."
        ),
    )
    parser.add_argument("--max-allocation-ranks", type=int, default=32)
    parser.add_argument(
        "--allocation-rank-grid",
        default="",
        help=(
            "Optional non-uniform QSL rank grid. When set, rank 0 and the true "
            "rate-feasible maximum are included automatically; this avoids a "
            "low-rank truncation bias on 3B/7B tensors."
        ),
    )
    parser.add_argument("--obs-rcond", type=float, default=1e-10)
    parser.add_argument("--scale-min", type=float, default=0.0)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--rho-threshold", type=float, default=0.1)
    parser.add_argument("--rate-tolerance", type=float, default=0.01)
    parser.add_argument("--comfort-epsilons", default="0,0.125,0.25,0.5,0.75,1")
    parser.add_argument(
        "--comfort-strategies",
        default="Q,Q_block_scale,Q+S_OBS,Q+L,Q+S+L_QL_budget_component_scale,Q+S+L_component_scale",
    )
    parser.add_argument("--comfort-fit-max-epsilon", type=float, default=0.25)
    parser.add_argument("--comfort-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--comfort-absolute-tolerance", type=float, default=1e-4)
    parser.add_argument("--skip-comfort", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def validate_research_codec_bits(bits: int) -> int:
    """Reject bit widths that the signed research codec cannot represent exactly."""

    if bits < 2 or bits > 8:
        raise ValueError(
            "--bits must be within [2, 8]; the current signed 1-bit stream cannot "
            "represent encode_row_rtn's {-1, 0, +1} code alphabet"
        )
    return bits


def main() -> None:
    args = build_arg_parser().parse_args()
    base.set_seed(args.seed)
    args.module_types = base.parse_csv(args.module_types, [])
    args.layer_positions = base.parse_csv(args.layer_positions, ["first", "middle", "last"])
    args.layers = base.parse_int_csv(args.layers)
    args.target_ratios = sorted(set(base.parse_float_csv(args.target_ratios, [0.258])))
    args.candidate_bits = sorted(
        set((base.parse_int_csv(args.candidate_bits) or []) + [int(args.bits)])
    )
    args.candidate_q_group_sizes = sorted(
        set(base.parse_int_csv(args.candidate_q_group_sizes) + [0])
    )
    args.candidate_quantizers = sorted(
        set(
            base.parse_csv(args.candidate_quantizers, ["symmetric_rtn"])
            + ["symmetric_rtn"]
        )
    )
    args.candidate_lowrank_factor_bits = sorted(
        set(base.parse_int_csv(args.candidate_lowrank_factor_bits) + [16])
    )
    args.repair_block_sizes = base.parse_int_csv(args.repair_block_sizes)
    args.allocation_rank_grid = parse_allocation_rank_grid(args.allocation_rank_grid)
    args.comfort_epsilons = sorted(set(base.parse_float_csv(args.comfort_epsilons, [0.0, 1.0])))
    args.comfort_strategies = base.parse_csv(args.comfort_strategies, [])
    args.global_frontier_support_fractions = sorted(
        set(base.parse_float_csv(args.global_frontier_support_fractions, []))
    )
    args.global_frontier_budget_multipliers = sorted(
        set(base.parse_float_csv(args.global_frontier_budget_multipliers, []))
    )
    args.scale_bounds = (float(args.scale_min), float(args.scale_max))
    validate_research_codec_bits(args.bits)
    for bits in args.candidate_bits:
        validate_research_codec_bits(bits)
    for bits in args.candidate_lowrank_factor_bits:
        if bits != 16:
            validate_research_codec_bits(bits)
    if any(group_size < 0 for group_size in args.candidate_q_group_sizes):
        raise ValueError("--candidate-q-group-sizes must contain non-negative integers")
    if not args.candidate_q_group_sizes:
        raise ValueError("--candidate-q-group-sizes must be non-empty")
    supported_quantizers = {"symmetric_rtn", "symmetric_mse_clip"}
    unknown_quantizers = set(args.candidate_quantizers) - supported_quantizers
    if unknown_quantizers:
        raise ValueError(
            f"unsupported --candidate-quantizers: {sorted(unknown_quantizers)}"
        )
    if args.artifact_alignment <= 0 or args.artifact_alignment & (args.artifact_alignment - 1):
        raise ValueError("--artifact-alignment must be a positive power of two")
    if args.enforce_serialized_rate_cap and not args.emit_codec_artifacts:
        raise ValueError("--enforce-serialized-rate-cap requires --emit-codec-artifacts")
    if not args.target_ratios or any(value <= 0.0 or value > 1.0 for value in args.target_ratios):
        raise ValueError("--target-ratios must be non-empty values in (0, 1]")
    if args.scale_min > args.scale_max:
        raise ValueError("--scale-min must not exceed --scale-max")
    if not math.isfinite(args.covariance_damping_ratio) or args.covariance_damping_ratio < 0.0:
        raise ValueError("--covariance-damping-ratio must be finite and non-negative")
    if not math.isfinite(args.whitening_floor_ratio) or args.whitening_floor_ratio < 0.0:
        raise ValueError("--whitening-floor-ratio must be finite and non-negative")
    if args.lowrank_svd_oversampling < 0 or args.lowrank_svd_niter < 0:
        raise ValueError("--lowrank-svd-oversampling and --lowrank-svd-niter must be non-negative")
    if args.global_frontier_top_ranks < 0:
        raise ValueError("--global-frontier-top-ranks must be non-negative")
    if args.calib_limit <= 0 or args.selection_limit <= 0 or args.eval_limit <= 0:
        raise ValueError("--calib-limit, --selection-limit and --eval-limit must be positive")
    if args.selection_top_k <= 0:
        raise ValueError("--selection-top-k must be positive")
    if args.candidate_family_top_k < 0:
        raise ValueError("--candidate-family-top-k must be non-negative")
    if args.two_stage_selection:
        if args.rate_allocation != "global_exact":
            raise ValueError("--two-stage-selection requires --rate-allocation global_exact")
        if args.selection_top_k < 2:
            raise ValueError("--two-stage-selection requires --selection-top-k >= 2")
        if args.proxy_only:
            raise ValueError("--two-stage-selection requires validation NLL, not --proxy-only")
    if args.include_global_single_component_controls and args.rate_allocation != "global_exact":
        raise ValueError(
            "--include-global-single-component-controls requires --rate-allocation global_exact"
        )
    snapshot_values = (
        args.model_snapshot_manifest,
        args.model_snapshot_manifest_sha256,
        args.model_snapshot_aggregate_sha256,
    )
    if any(snapshot_values) and not all(snapshot_values):
        raise ValueError(
            "model snapshot manifest path, file SHA-256 and aggregate SHA-256 "
            "must be provided together"
        )
    if args.max_allocation_ranks < 0:
        raise ValueError("--max-allocation-ranks must be non-negative")
    if any(rank < 0 for rank in args.allocation_rank_grid):
        raise ValueError("--allocation-rank-grid must contain non-negative ranks")
    if not args.skip_block_scale and not args.repair_block_sizes:
        raise ValueError("--repair-block-sizes must be non-empty unless --skip-block-scale is set")
    if any(value <= 0.0 or value >= 1.0 for value in args.global_frontier_support_fractions):
        raise ValueError("--global-frontier-support-fractions must lie strictly inside (0, 1)")
    if any(
        not math.isfinite(value) or value <= 1.0 or value > 16.0
        for value in args.global_frontier_budget_multipliers
    ):
        raise ValueError("--global-frontier-budget-multipliers must lie in (1, 16]")
    if any(value < 0.0 or value > 1.0 for value in args.comfort_epsilons):
        raise ValueError("--comfort-epsilons must be in [0, 1]")
    if 0.0 not in args.comfort_epsilons:
        args.comfort_epsilons.insert(0, 0.0)
    if 1.0 not in args.comfort_epsilons:
        args.comfort_epsilons.append(1.0)
    if not args.skip_comfort:
        if not math.isfinite(args.comfort_fit_max_epsilon) or args.comfort_fit_max_epsilon <= 0.0:
            raise ValueError("--comfort-fit-max-epsilon must be positive when comfort paths are enabled")
        fitted_positive = [
            value
            for value in args.comfort_epsilons
            if 0.0 < value <= float(args.comfort_fit_max_epsilon)
        ]
        if len(fitted_positive) < 2:
            raise ValueError(
                "comfort paths require at least two positive epsilons at or below "
                "--comfort-fit-max-epsilon"
            )
    args.endpoint_target = min(args.target_ratios, key=lambda value: abs(value - float(args.endpoint_target)))

    model_snapshot_evidence: dict[str, object] | None = None
    if args.model_snapshot_manifest:
        model_snapshot_evidence = model_snapshot_manifest.verify_model_snapshot_manifest(
            args.model_snapshot_manifest,
            args.model,
            expected_manifest_sha256=args.model_snapshot_manifest_sha256,
            expected_aggregate_sha256=args.model_snapshot_aggregate_sha256,
        )

    resource_gate_evidence: dict[str, object] | None = None
    if args.resource_gate_manifest:
        gate_path = Path(args.resource_gate_manifest).expanduser().resolve(strict=True)
        gate_raw = gate_path.read_bytes()
        gate = json.loads(gate_raw.decode("utf-8"))
        if (
            not isinstance(gate, dict)
            or gate.get("schema_version") != "large_scale_hessian_resource_gate.v1"
            or gate.get("gate_passed") is not True
            or gate.get("lock_acquired") is not True
        ):
            raise ValueError("resource gate manifest does not prove gate_passed=true")
        physical_gpu = gate.get("selected_physical_gpu")
        if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int):
            raise ValueError("resource gate manifest has no physical GPU index")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES differs from the resource-gated physical GPU"
            )
        if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise ValueError("CUDA_DEVICE_ORDER must be PCI_BUS_ID for physical GPU binding")
        resource_gate_evidence = {
            "schema_version": str(gate["schema_version"]),
            "path": str(gate_path),
            "sha256": hashlib.sha256(gate_raw).hexdigest(),
            "selected_physical_gpu": physical_gpu,
            "consumed_before_model_load": True,
        }

    args.zero_shot_tasks = []
    args.spq_lora_train_limit = 0
    args.spq_lora_steps = 0
    args.disjoint_text_splits = True
    args.data_cfg = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "backup_name": args.backup_name,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "allow_fallback": False,
    }
    if args.two_stage_selection:
        load_two_stage_text_windows(args)
    else:
        required_texts = (
            int(args.calib_limit) + int(args.eval_limit) + 1
        ) * int(args.texts_per_batch_window)
        # Load a safety margin because WikiText contains occasional duplicate
        # non-empty rows; the strict splitter below never repeats source content.
        text_pool_limit = required_texts + max(
            64, int(args.texts_per_batch_window) * 4
        )
        text_pool, args.text_source_used, args.text_source_metadata = (
            base.load_eval_texts(args, limit=text_pool_limit)
        )
        if not str(args.text_source_used).startswith("dataset:wikitext"):
            raise RuntimeError(
                f"real WikiText was required, got {args.text_source_used!r}"
            )
        split_content_disjoint_text_windows(args, text_pool)
        args.text_pool_count = len(text_pool)

    output_dir = prepare_fresh_output_dir(Path(args.output_dir))
    source_snapshot = _source_snapshot(args.model_snapshot_manifest)
    model_cfg: dict[str, object] = {
        "model": args.model,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if args.revision:
        model_cfg["revision"] = args.revision
    model, tokenizer, device = base.load_model_and_tokenizer_from_config(model_cfg)
    args.device = device
    if args.svd_device == "auto":
        args.svd_device = "cuda" if torch.cuda.is_available() else "cpu"

    modules = base.discover_target_linears(
        model,
        module_types=args.module_types,
        layer_positions=args.layer_positions,
        layers=args.layers,
        max_modules=args.max_modules,
    )
    args.selected_layers = list(modules)
    baseline_weights = base.clone_weights(modules)
    baseline_metrics: dict[str, float | int] = {}
    baseline_window_rows: list[dict[str, object]] = []
    if not args.two_stage_selection:
        baseline_metrics, baseline_window_rows = evaluate_current_model_with_windows(
            model,
            tokenizer,
            strategy="dense",
            texts=args.eval_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=device,
            eval_limit=args.eval_limit,
        )
    covariances, activation_counts = base.collect_activation_covariances(
        model,
        tokenizer,
        modules,
        texts=args.calib_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
    )
    activation_samples = base.collect_activation_samples(
        model,
        tokenizer,
        modules,
        texts=args.calib_texts,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=device,
        calib_limit=args.calib_limit,
        max_rows=args.selector_activation_sample_rows,
    )

    omitted_strategy_reasons: dict[str, str] = {}
    if args.skip_block_scale:
        omitted_strategy_reasons["Q_block_scale"] = "explicit_large_tensor_resource_guard"
    if not args.include_global_single_component_controls:
        omitted_strategy_reasons.update(
            {
                strategy: "opt_in_global_single_component_control_disabled"
                for strategy in GLOBAL_CONTROL_STRATEGIES
            }
        )
    active_strategies = tuple(
        strategy for strategy in STRATEGY_ORDER if strategy not in omitted_strategy_reasons
    )
    local_active_strategies = tuple(
        strategy
        for strategy in active_strategies
        if strategy not in GLOBAL_CONTROL_STRATEGIES
    )
    args.active_strategies = list(active_strategies)
    args.omitted_strategy_reasons = omitted_strategy_reasons
    aggregates = {
        (target, strategy): _empty_aggregate(strategy, target)
        for target in args.target_ratios
        for strategy in active_strategies
    }
    candidate_rows: list[dict[str, object]] = []
    endpoint_replacements: dict[str, dict[str, torch.Tensor]] = {
        strategy: {} for strategy in active_strategies
    }
    endpoint_candidates: dict[str, dict[str, Candidate]] = {
        strategy: {} for strategy in active_strategies
    }
    factorizer_dims: dict[str, int] = {}
    factorizer_audit: dict[str, dict[str, object]] = {}
    covariance_psd_rows: list[dict[str, object]] = []
    layer_metrics: dict[str, HessianMetric] = {}
    layer_covariances: dict[str, PreparedInputCovariance] = {}
    global_option_pools_by_strategy: dict[str, dict[str, list[Candidate]]] = {
        "Q+S+L_QL_budget": {}
    }
    if args.include_global_single_component_controls:
        global_option_pools_by_strategy.update(
            {strategy: {} for strategy in GLOBAL_CONTROL_STRATEGIES}
        )
    rate_allocator_report: dict[str, object] = {
        "mode": args.rate_allocation,
        "selection_source": "per_layer_Q+L_cap_with_alignment_guard",
        "optimality_scope": "independent_per_layer_rank_frontier",
    }

    for layer, weight_tensor in baseline_weights.items():
        weight_np = weight_tensor.detach().cpu().float().numpy()
        covariance_tensor, prepared_covariance, covariance_psd_report = prepare_metric_covariance(
            covariances[layer],
            mode=args.covariance_mode,
            damping_ratio=args.covariance_damping_ratio,
        )
        covariance_psd_rows.append({"layer": layer, **covariance_psd_report})
        quantizer_codecs = build_quantizer_candidate_codecs(
            weight_np,
            bit_widths=args.candidate_bits,
            group_sizes=args.candidate_q_group_sizes,
            quantizers=args.candidate_quantizers,
        )
        q = next(
            codec
            for codec in quantizer_codecs
            if codec.bits == int(args.bits)
            and codec.col_block_size is None
            and codec.quantizer == "symmetric_rtn"
        )
        global_q = make_global_scaled_q(
            layer,
            args.endpoint_target,
            weight_np,
            q,
            prepared_covariance,
            args.scale_bounds,
        )
        block_options = (
            []
            if args.skip_block_scale
            else make_block_scaled_q_candidates(
                layer,
                weight_np,
                q,
                prepared_covariance,
                args.repair_block_sizes,
                args.scale_bounds,
            )
        )
        factorizer = LowRankFactorizer(
            covariance_tensor,
            method=args.l_method,
            device=args.svd_device,
            whitening_floor_ratio=args.whitening_floor_ratio,
            svd_solver=args.lowrank_svd_solver,
            randomized_oversampling=args.lowrank_svd_oversampling,
            randomized_niter=args.lowrank_svd_niter,
            randomized_seed=args.seed,
            seed_namespace=layer,
        )
        metric = HessianMetric(covariance_tensor, device=args.svd_device)
        layer_metrics[layer] = metric
        layer_covariances[layer] = prepared_covariance
        factorizer_dims[layer] = int(covariance_tensor.shape[0])
        heterogeneous_families, family_rows = (
            screen_heterogeneous_candidate_families(
                layer=layer,
                weight=weight_np,
                quantizer_codecs=quantizer_codecs,
                lowrank_factor_bits=args.candidate_lowrank_factor_bits,
                default_q=q,
                metric=metric,
                support_encoding=args.support_encoding,
                target_ratio=args.endpoint_target,
                top_k=args.candidate_family_top_k,
            )
        )
        candidate_rows.extend(family_rows)
        for target in args.target_ratios:
            selected, rows, global_option_pools = build_layer_candidates(
                layer=layer,
                weight_tensor=weight_tensor,
                covariance_tensor=covariance_tensor,
                prepared_covariance=prepared_covariance,
                activation_samples=activation_samples.get(layer),
                factorizer=factorizer,
                metric=metric,
                target_ratio=target,
                q=q,
                global_q=global_q,
                block_q_options=block_options,
                lowrank_factor_bits=16,
                args=args,
            )
            candidate_rows.extend(rows)
            selected_metrics = {
                str(row["strategy"]): row
                for row in rows
                if row.get("search_family") == "selected_endpoint"
            }
            for strategy in local_active_strategies:
                metrics = selected_metrics[strategy]
                update_aggregate(aggregates[(target, strategy)], metrics)
                if abs(target - args.endpoint_target) <= 1e-12:
                    endpoint_candidates[strategy][layer] = selected[strategy]
                    final = torch.from_numpy(selected[strategy].final).to(dtype=weight_tensor.dtype)
                    endpoint_replacements[strategy][layer] = final
            if abs(target - args.endpoint_target) <= 1e-12:
                for strategy, options in global_option_pools.items():
                    if strategy in global_option_pools_by_strategy:
                        global_option_pools_by_strategy[strategy][layer] = list(options)
                for alternative_q, factor_bits in heterogeneous_families:
                    if alternative_q is q and int(factor_bits) == 16:
                        continue
                    alternative_global_q = make_global_scaled_q(
                        layer,
                        args.endpoint_target,
                        weight_np,
                        alternative_q,
                        prepared_covariance,
                        args.scale_bounds,
                    )
                    alternative_args = copy.copy(args)
                    alternative_args.skip_block_scale = True
                    (
                        _alternative_selected,
                        alternative_rows,
                        alternative_global_pools,
                    ) = build_layer_candidates(
                        layer=layer,
                        weight_tensor=weight_tensor,
                        covariance_tensor=covariance_tensor,
                        prepared_covariance=prepared_covariance,
                        activation_samples=activation_samples.get(layer),
                        factorizer=factorizer,
                        metric=metric,
                        target_ratio=target,
                        q=alternative_q,
                        global_q=alternative_global_q,
                        block_q_options=[],
                        lowrank_factor_bits=int(factor_bits),
                        args=alternative_args,
                    )
                    for row in alternative_rows:
                        row["heterogeneous_quantizer_candidate"] = True
                        row["candidate_lowrank_factor_bits"] = int(factor_bits)
                        row["eligible_for_global_allocator"] = bool(
                            str(row.get("search_family", "")).startswith(
                                "global_"
                            )
                            or row.get("strategy") in GLOBAL_CONTROL_STRATEGIES
                            or row.get("strategy") == "Q+S+L_QL_budget"
                        )
                        row["selected_within_layer_target"] = False
                    candidate_rows.extend(alternative_rows)
                    for strategy, options in alternative_global_pools.items():
                        if strategy in global_option_pools_by_strategy:
                            global_option_pools_by_strategy[strategy][layer].extend(
                                options
                            )
        factorizer_audit[layer] = dict(factorizer.diagnostics)

    allocation_validation_rows: list[dict[str, object]] = []
    allocation_validation_window_rows: list[dict[str, object]] = []
    allocation_validation_baseline: dict[str, float | int] | None = None
    allocation_selection_reports: dict[str, dict[str, object]] = {}
    if args.rate_allocation == "global_exact":
        proxy_top_k = args.selection_top_k if args.two_stage_selection else 1
        ranked_global_qsl, rate_allocator_report = rank_global_exact_qsl_allocations(
            ql_candidates=endpoint_candidates["Q+L"],
            degenerate_candidates={
                layer: [
                    endpoint_candidates["Q"][layer],
                    endpoint_candidates["Q+S_OBS"][layer],
                    endpoint_candidates["Q+L"][layer],
                    *(
                        global_option_pools_by_strategy[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ][layer]
                        if args.include_global_single_component_controls
                        else []
                    ),
                ]
                for layer in endpoint_candidates["Q+L"]
            },
            option_pools=global_option_pools_by_strategy["Q+S+L_QL_budget"],
            fallback_candidates=endpoint_candidates["Q+S+L_QL_budget"],
            metrics=layer_metrics,
            alignment=args.artifact_alignment,
            top_k=proxy_top_k,
        )
        ranked_by_strategy: dict[str, list[RankedGlobalAllocation]] = {
            "Q+S+L_QL_budget": ranked_global_qsl,
        }
        control_reports: dict[str, dict[str, object]] = {}
        ranked_controls: dict[str, list[RankedGlobalAllocation]] = {}
        if args.include_global_single_component_controls:
            control_specs = {
                "Q+L_global": {
                    "optimality_scope": (
                        "enumerated_independent_pure_lowrank_rank_grid_and_budget_band_extrema_with_"
                        "safe_header_payload_cost_pareto_and_exact_final_serialization"
                    ),
                    "candidate_pool_asymmetry": (
                        "pure-L independently fits the unchanged Q residual at each rank and reuses "
                        "the original local Q+L factor at its rank; QSL fits each rank after a "
                        "rank-conditioned sparse support, so residual targets differ even though "
                        "the factorizer settings, byte oracle and aggregate cap are shared"
                    ),
                },
                "Q+S_OBS_global": {
                    "optimality_scope": (
                        "enumerated_pure_sparse_OBS_local_and_budget_band_support_grid_with_"
                        "safe_header_payload_cost_pareto_and_exact_final_serialization"
                    ),
                    "candidate_pool_asymmetry": (
                        "pure-S enumerates OBS-refit supports at every budget-band extremum and "
                        "declared fraction; QSL enumerates rank-conditioned supports, so only the "
                        "byte oracle, cap, support scorer and OBS value refit are shared"
                    ),
                },
                GLOBAL_NONJOINT_CONTROL_STRATEGY: {
                    "optimality_scope": (
                        "enumerated_union_of_pure_sparse_OBS_and_independent_pure_lowrank_"
                        "families_with_no_same_layer_joint_state_and_safe_header_payload_cost_"
                        "pareto_and_exact_final_serialization"
                    ),
                    "candidate_pool_asymmetry": (
                        "this no-joint control uses exactly the Q, pure-S/OBS and pure-L states "
                        "nested in the QSL candidate pool; QSL differs only by additionally "
                        "admitting same-layer joint S+L states and its audited local sentinel"
                    ),
                },
            }
            for strategy in GLOBAL_CONTROL_STRATEGIES:
                ranked_control, control_report = rank_global_exact_component_allocations(
                    strategy=strategy,
                    ql_candidates=endpoint_candidates["Q+L"],
                    option_pools=global_option_pools_by_strategy[strategy],
                    metrics=layer_metrics,
                    alignment=args.artifact_alignment,
                    optimality_scope=control_specs[strategy]["optimality_scope"],
                    candidate_pool_asymmetry=control_specs[strategy]["candidate_pool_asymmetry"],
                    top_k=proxy_top_k,
                )
                ranked_controls[strategy] = ranked_control
                ranked_by_strategy[strategy] = ranked_control
                control_reports[strategy] = control_report
            nonjoint_cost = float(
                control_reports[GLOBAL_NONJOINT_CONTROL_STRATEGY][
                    "selected_hessian_cost"
                ]
            )
            best_pure_cost = min(
                float(control_reports[strategy]["selected_hessian_cost"])
                for strategy in GLOBAL_SINGLE_COMPONENT_STRATEGIES
            )
            qsl_cost = float(rate_allocator_report["selected_hessian_cost"])
            dominance_tolerance = 1e-10 * max(1.0, abs(nonjoint_cost), abs(qsl_cost))
            if nonjoint_cost > best_pure_cost + dominance_tolerance:
                raise RuntimeError(
                    "no-joint union failed to weakly dominate its nested pure controls"
                )
            if qsl_cost > nonjoint_cost + dominance_tolerance:
                raise RuntimeError(
                    "QSL exact pool failed to weakly dominate its nested no-joint control"
                )
            rate_allocator_report["joint_control_strategy"] = (
                GLOBAL_NONJOINT_CONTROL_STRATEGY
            )
            rate_allocator_report["joint_candidate_incremental_hessian_gain"] = (
                nonjoint_cost - qsl_cost
            )
            rate_allocator_report["joint_proxy_pool_weakly_dominated"] = True
            rate_allocator_report["joint_control_weakly_dominated"] = True
            rate_allocator_report[
                "nonjoint_union_weakly_dominates_pure_controls"
            ] = True
            rate_allocator_report["best_pure_control_hessian_cost"] = best_pure_cost
            rate_allocator_report[
                "nonjoint_heterogeneous_gain_over_best_pure"
            ] = best_pure_cost - nonjoint_cost
            rate_allocator_report["shared_control_cap_policy"] = (
                "all global controls and QSL use the same aggregate natural Q+L file-byte cap; "
                "each selected artifact is padded to that exact physical byte count"
            )

        if args.two_stage_selection:
            (
                selected_by_validation,
                allocation_validation_rows,
                allocation_validation_window_rows,
                allocation_validation_baseline,
                allocation_selection_reports,
            ) = rerank_global_allocations_by_validation(
                model=model,
                tokenizer=tokenizer,
                modules=modules,
                baseline_weights=baseline_weights,
                ranked_by_strategy=ranked_by_strategy,
                args=args,
            )
            qsl_selection = allocation_selection_reports["Q+S+L_QL_budget"]
            qsl_allocation = ranked_global_qsl[
                int(qsl_selection["validation_selected_proxy_rank"]) - 1
            ]
            if args.include_global_single_component_controls:
                cap_best_nojoint_report = copy.deepcopy(
                    control_reports[GLOBAL_NONJOINT_CONTROL_STRATEGY]
                )
                cap_best_nojoint_ranked = ranked_controls[
                    GLOBAL_NONJOINT_CONTROL_STRATEGY
                ]
                matched_nojoint_ranked, matched_nojoint_report = (
                    attempt_exact_natural_component_allocations(
                        strategy=GLOBAL_NONJOINT_CONTROL_STRATEGY,
                        ql_candidates=endpoint_candidates["Q+L"],
                        option_pools=global_option_pools_by_strategy[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ],
                        metrics=layer_metrics,
                        alignment=args.artifact_alignment,
                        optimality_scope=control_specs[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ]["optimality_scope"],
                        candidate_pool_asymmetry=control_specs[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ]["candidate_pool_asymmetry"],
                        top_k=proxy_top_k,
                        required_natural_file_bytes=qsl_allocation.natural_file_bytes,
                    )
                )
                rate_allocator_report["nojoint_cap_best_audit"] = (
                    cap_best_nojoint_report
                )
                if matched_nojoint_ranked:
                    (
                        matched_selected,
                        matched_rows,
                        matched_window_rows,
                        matched_baseline,
                        matched_selection_reports,
                    ) = rerank_global_allocations_by_validation(
                        model=model,
                        tokenizer=tokenizer,
                        modules=modules,
                        baseline_weights=baseline_weights,
                        ranked_by_strategy={
                            GLOBAL_NONJOINT_CONTROL_STRATEGY: matched_nojoint_ranked
                        },
                        args=args,
                        validation_baseline_metrics=allocation_validation_baseline,
                    )
                    if matched_baseline != allocation_validation_baseline:
                        raise AssertionError(
                            "matched no-joint rerank changed the validation baseline"
                        )
                    allocation_validation_rows = [
                        row
                        for row in allocation_validation_rows
                        if row.get("strategy")
                        != GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ]
                    allocation_validation_rows.extend(matched_rows)
                    allocation_validation_window_rows = [
                        row
                        for row in allocation_validation_window_rows
                        if row.get("base_strategy")
                        != GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ]
                    allocation_validation_window_rows.extend(matched_window_rows)
                    selected_by_validation.update(matched_selected)
                    allocation_selection_reports.update(
                        matched_selection_reports
                    )
                    ranked_controls[
                        GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ] = matched_nojoint_ranked
                    ranked_by_strategy[
                        GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ] = matched_nojoint_ranked
                    matched_nojoint_report["cap_best_under_shared_cap"] = {
                        "selected_hessian_cost": cap_best_nojoint_report[
                            "selected_hessian_cost"
                        ],
                        "selected_natural_file_bytes": cap_best_nojoint_report[
                            "selected_natural_file_bytes"
                        ],
                        "proxy_top_k": cap_best_nojoint_report["proxy_top_k"],
                    }
                    control_reports[
                        GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ] = matched_nojoint_report
                    rate_allocator_report[
                        "joint_control_natural_match_available"
                    ] = True
                    rate_allocator_report[
                        "joint_control_natural_match_search_status"
                    ] = "completed_match"
                    rate_allocator_report[
                        "joint_control_required_natural_file_bytes"
                    ] = qsl_allocation.natural_file_bytes
                    rate_allocator_report[
                        "joint_control_natural_match_policy"
                    ] = (
                        "after validation selects QSL, the no-joint pool is reallocated "
                        "under exact equality to the selected QSL natural artifact bytes "
                        "and independently reranked on the same validation split"
                    )
                else:
                    match_search_status = str(
                        matched_nojoint_report.get(
                            "search_status", "completed_no_match"
                        )
                    )
                    rate_allocator_report[
                        "joint_control_natural_match_available"
                    ] = False
                    rate_allocator_report[
                        "joint_control_natural_match_search_status"
                    ] = match_search_status
                    rate_allocator_report[
                        "joint_control_required_natural_file_bytes"
                    ] = qsl_allocation.natural_file_bytes
                    if match_search_status == "state_limit_exceeded":
                        match_policy = (
                            "the exact-natural no-joint search exceeded its hard state "
                            "limit; the cap-best control remains descriptive and the "
                            "joint-value claim is unavailable"
                        )
                    else:
                        match_policy = (
                            "no enumerated no-joint canonical layout exactly matched the "
                            "validation-selected QSL natural artifact bytes; the cap-best "
                            "control remains descriptive and cannot support a joint-value "
                            "claim"
                        )
                    rate_allocator_report[
                        "joint_control_natural_match_policy"
                    ] = match_policy
                    control_reports[GLOBAL_NONJOINT_CONTROL_STRATEGY][
                        "exact_natural_match_attempt"
                    ] = matched_nojoint_report

            selected_global_qsl = selected_by_validation["Q+S+L_QL_budget"]
            _apply_validation_selection_to_allocator_report(
                rate_allocator_report,
                allocation=qsl_allocation,
                selection_report=qsl_selection,
            )
            _decorate_global_qsl_selection(
                selected_global_qsl,
                ql_candidates=endpoint_candidates["Q+L"],
                report=rate_allocator_report,
                optimality_scope=str(rate_allocator_report["optimality_scope"]),
            )
            for strategy, ranked_control in ranked_controls.items():
                selected_control = selected_by_validation[strategy]
                selection_report = allocation_selection_reports[strategy]
                selected_allocation = ranked_control[
                    int(selection_report["validation_selected_proxy_rank"]) - 1
                ]
                control_report = control_reports[strategy]
                _apply_validation_selection_to_allocator_report(
                    control_report,
                    allocation=selected_allocation,
                    selection_report=selection_report,
                )
                _decorate_global_component_selection(
                    selected_control,
                    strategy=strategy,
                    ql_candidates=endpoint_candidates["Q+L"],
                    report=control_report,
                    optimality_scope=str(control_report["optimality_scope"]),
                    candidate_pool_asymmetry=str(
                        control_report["candidate_pool_asymmetry"]
                    ),
                )
                endpoint_candidates[strategy] = selected_control
            rate_allocator_report["two_stage_selection"] = {
                "enabled": True,
                "proxy_screen": "exact_file_byte_feasible_hessian_top_k",
                "proxy_top_k": proxy_top_k,
                "rerank_metric": "validation_nll",
                "selection_split": args.selection_split,
                "test_split_reserved_until_after_selection": True,
                "selection_reports": allocation_selection_reports,
            }
            if args.include_global_single_component_controls:
                qsl_validation = allocation_selection_reports[
                    "Q+S+L_QL_budget"
                ]
                nojoint_validation = allocation_selection_reports[
                    GLOBAL_NONJOINT_CONTROL_STRATEGY
                ]
                rate_allocator_report["joint_validation_nll_gain_over_nojoint"] = (
                    float(nojoint_validation["validation_selected_nll"])
                    - float(qsl_validation["validation_selected_nll"])
                )
                rate_allocator_report["joint_validation_counterfactual_scope"] = (
                    "exact_natural_matched"
                    if rate_allocator_report.get(
                        "joint_control_natural_match_available"
                    )
                    is True
                    else "cap_best_under_shared_cap_descriptive"
                )
        else:
            selected_global_qsl = ranked_global_qsl[0].candidates
            if args.include_global_single_component_controls:
                cap_best_nojoint_report = copy.deepcopy(
                    control_reports[GLOBAL_NONJOINT_CONTROL_STRATEGY]
                )
                matched_nojoint_ranked, matched_nojoint_report = (
                    attempt_exact_natural_component_allocations(
                        strategy=GLOBAL_NONJOINT_CONTROL_STRATEGY,
                        ql_candidates=endpoint_candidates["Q+L"],
                        option_pools=global_option_pools_by_strategy[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ],
                        metrics=layer_metrics,
                        alignment=args.artifact_alignment,
                        optimality_scope=control_specs[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ]["optimality_scope"],
                        candidate_pool_asymmetry=control_specs[
                            GLOBAL_NONJOINT_CONTROL_STRATEGY
                        ]["candidate_pool_asymmetry"],
                        top_k=1,
                        required_natural_file_bytes=ranked_global_qsl[
                            0
                        ].natural_file_bytes,
                    )
                )
                rate_allocator_report["nojoint_cap_best_audit"] = (
                    cap_best_nojoint_report
                )
                if matched_nojoint_ranked:
                    ranked_controls[
                        GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ] = matched_nojoint_ranked
                    matched_nojoint_report["cap_best_under_shared_cap"] = {
                        "selected_hessian_cost": cap_best_nojoint_report[
                            "selected_hessian_cost"
                        ],
                        "selected_natural_file_bytes": cap_best_nojoint_report[
                            "selected_natural_file_bytes"
                        ],
                        "proxy_top_k": cap_best_nojoint_report["proxy_top_k"],
                    }
                    control_reports[
                        GLOBAL_NONJOINT_CONTROL_STRATEGY
                    ] = matched_nojoint_report
                    rate_allocator_report[
                        "joint_control_natural_match_available"
                    ] = True
                    rate_allocator_report[
                        "joint_control_natural_match_search_status"
                    ] = "completed_match"
                else:
                    match_search_status = str(
                        matched_nojoint_report.get(
                            "search_status", "completed_no_match"
                        )
                    )
                    rate_allocator_report[
                        "joint_control_natural_match_available"
                    ] = False
                    rate_allocator_report[
                        "joint_control_natural_match_search_status"
                    ] = match_search_status
                    control_reports[GLOBAL_NONJOINT_CONTROL_STRATEGY][
                        "exact_natural_match_attempt"
                    ] = matched_nojoint_report
                rate_allocator_report[
                    "joint_control_required_natural_file_bytes"
                ] = ranked_global_qsl[0].natural_file_bytes
            _decorate_global_qsl_selection(
                selected_global_qsl,
                ql_candidates=endpoint_candidates["Q+L"],
                report=rate_allocator_report,
                optimality_scope=str(rate_allocator_report["optimality_scope"]),
            )
            for strategy, ranked_control in ranked_controls.items():
                selected_control = ranked_control[0].candidates
                control_report = control_reports[strategy]
                _decorate_global_component_selection(
                    selected_control,
                    strategy=strategy,
                    ql_candidates=endpoint_candidates["Q+L"],
                    report=control_report,
                    optimality_scope=str(control_report["optimality_scope"]),
                    candidate_pool_asymmetry=str(
                        control_report["candidate_pool_asymmetry"]
                    ),
                )
                endpoint_candidates[strategy] = selected_control
            rate_allocator_report["two_stage_selection"] = {
                "enabled": False,
                "proxy_top_k": 1,
            }

        selected_global_scaled = {
            layer: make_component_scaled_candidate(
                candidate,
                layer_covariances[layer],
                args.scale_bounds,
                strategy="Q+S+L_QL_budget_component_scale",
            )
            for layer, candidate in selected_global_qsl.items()
        }
        endpoint_candidates["Q+S+L_QL_budget"] = selected_global_qsl
        endpoint_candidates["Q+S+L_QL_budget_component_scale"] = selected_global_scaled
        selected_global_endpoints: dict[str, dict[str, Candidate]] = {
            "Q+S+L_QL_budget": selected_global_qsl,
            "Q+S+L_QL_budget_component_scale": selected_global_scaled,
        }
        for strategy in GLOBAL_CONTROL_STRATEGIES:
            if endpoint_candidates.get(strategy):
                selected_global_endpoints[strategy] = endpoint_candidates[strategy]
        if args.include_global_single_component_controls:
            rate_allocator_report["single_component_controls"] = control_reports
            rate_allocator_report["global_control_reports"] = control_reports

        for strategy, candidates in selected_global_endpoints.items():
            aggregates[(args.endpoint_target, strategy)] = _empty_aggregate(
                strategy, args.endpoint_target
            )
            for row in candidate_rows:
                if (
                    row.get("search_family") == "selected_endpoint"
                    and row.get("strategy") == strategy
                    and abs(float(row.get("target_ratio", -1.0)) - args.endpoint_target) <= 1e-12
                ):
                    row["selected_within_layer_target"] = False
                    row["superseded_by_global_allocator"] = True
            for layer, candidate in candidates.items():
                q_candidate = endpoint_candidates["Q"][layer]
                q_payload = q_candidate.payload(support_encoding=args.support_encoding)
                metrics_row = candidate_geometry(
                    candidate,
                    layer_metrics[layer],
                    activation_samples.get(layer),
                    support_encoding=args.support_encoding,
                    rho_threshold=args.rho_threshold,
                    q_reference_cost=layer_metrics[layer].cost(
                        q_candidate.final - q_candidate.weight
                    ),
                    q_reference_bits=q_payload.total_bits,
                )
                metrics_row.update(
                    {
                        "search_family": f"global_exact_Q+L_file_cap_selection:{strategy}",
                        "selected_within_layer_target": True,
                        "global_allocator_selected": True,
                    }
                )
                candidate_rows.append(metrics_row)
                update_aggregate(aggregates[(args.endpoint_target, strategy)], metrics_row)
                endpoint_replacements[strategy][layer] = torch.from_numpy(
                    candidate.final
                ).to(dtype=baseline_weights[layer].dtype)

    pre_evaluation_artifact_sizes: dict[str, int] = {}
    if args.enforce_serialized_rate_cap:
        pre_evaluation_artifact_sizes = validate_endpoint_serialized_rate_cap(
            endpoint_candidates,
            alignment=args.artifact_alignment,
        )
    if args.two_stage_selection:
        baseline_metrics, baseline_window_rows = evaluate_current_model_with_windows(
            model,
            tokenizer,
            strategy="dense",
            texts=args.eval_texts,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            device=device,
            eval_limit=args.eval_limit,
        )
        for row in baseline_window_rows:
            row["evidence_role"] = "final_test"

    endpoint_rows = [
        finalize_aggregate(aggregates[(target, strategy)], rate_tolerance=args.rate_tolerance, rho_threshold=args.rho_threshold)
        for target in args.target_ratios
        for strategy in active_strategies
        if strategy not in GLOBAL_CONTROL_STRATEGIES
        or abs(float(target) - float(args.endpoint_target)) <= 1e-12
    ]
    if allocation_selection_reports:
        for row in endpoint_rows:
            if abs(float(row["target_ratio"]) - float(args.endpoint_target)) > 1e-12:
                continue
            report = allocation_selection_reports.get(str(row["strategy"]))
            if report:
                row.update(report)
    add_parameter_efficiency(endpoint_rows)
    endpoint_window_rows = evaluate_endpoint_strategies(
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        replacements=endpoint_replacements,
        endpoint_rows=endpoint_rows,
        endpoint_target=args.endpoint_target,
        baseline_metrics=baseline_metrics,
        baseline_window_rows=baseline_window_rows,
        args=args,
    )
    comfort_rows, comfort_summary = evaluate_comfort_paths(
        model=model,
        tokenizer=tokenizer,
        modules=modules,
        baseline_weights=baseline_weights,
        replacements=endpoint_replacements,
        endpoint_rows=endpoint_rows,
        endpoint_target=args.endpoint_target,
        baseline_metrics=baseline_metrics,
        args=args,
    )
    artifact_rows: list[dict[str, object]] = []
    if args.emit_codec_artifacts:
        artifact_rows = emit_endpoint_codec_artifacts(
            output_dir,
            baseline_weights=baseline_weights,
            endpoint_candidates=endpoint_candidates,
            endpoint_rows=endpoint_rows,
            endpoint_target=args.endpoint_target,
            alignment=args.artifact_alignment,
            enforce_serialized_rate_cap=args.enforce_serialized_rate_cap,
            rate_allocation=args.rate_allocation,
        )
    joint_value_claim: dict[str, object] | None = None
    if args.include_global_single_component_controls:
        joint_value_claim = annotate_joint_value_claim(
            endpoint_rows,
            endpoint_target=args.endpoint_target,
            rate_allocator_report=rate_allocator_report,
        )

    base.write_csv(output_dir / "candidate_ablation.csv", candidate_rows)
    base.write_csv(output_dir / "strategy_endpoints.csv", endpoint_rows)
    base.write_csv(output_dir / "endpoint_window_nll.csv", baseline_window_rows + endpoint_window_rows)
    base.write_csv(
        output_dir / "allocation_validation_rerank.csv",
        allocation_validation_rows,
    )
    base.write_csv(
        output_dir / "allocation_validation_window_nll.csv",
        allocation_validation_window_rows,
    )
    base.write_csv(output_dir / "comfort_sweep.csv", comfort_rows)
    base.write_csv(output_dir / "comfort_summary.csv", comfort_summary)
    base.write_csv(output_dir / "covariance_psd_audit.csv", covariance_psd_rows)
    maximum_negative_relative = max(
        (max(0.0, -float(row["original_min_relative"])) for row in covariance_psd_rows),
        default=0.0,
    )
    maximum_shift_relative = max(
        (max(0.0, float(row["diagonal_shift_relative"])) for row in covariance_psd_rows),
        default=0.0,
    )
    run_config = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "revision": args.revision or None,
        "model_identity": _model_identity(model, tokenizer, args.revision),
        "model_snapshot": model_snapshot_evidence,
        "resource_gate": resource_gate_evidence,
        "git": _git_snapshot(),
        "source_snapshot": source_snapshot,
        "device": args.device,
        "svd_device": args.svd_device,
        "selected_layers": args.selected_layers,
        "selected_parameter_count": int(sum(weight.numel() for weight in baseline_weights.values())),
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "baseline_metrics": baseline_metrics,
        "allocation_validation_baseline_metrics": allocation_validation_baseline,
        "actual_eval_tokens": int(baseline_metrics["tokens"]),
        "payload_scope": "selected_linear_weights_only",
        "payload_codec": {
            "reference": "FP16 weights",
            "q_codes": f"packed {args.bits}-bit symmetric row RTN",
            "q_scales": "FP16 row or row-column-block scales",
            "sparse_values": "FP16",
            "sparse_support": "fixed-width CSR (storage index width inferred from columns, uint32 row pointers)",
            "lowrank_factors": "two FP16 factors",
            "folded_repairs": "no additional payload; rounded back into already-counted FP16 values/scales/factors",
            "container_headers": (
                f"measured in deterministic {int(args.artifact_alignment)}-byte-aligned research artifacts"
                if args.emit_codec_artifacts
                else "not emitted; serialized-byte claims are unavailable"
            ),
            "research_codec_only": True,
            "production_inference_backend": False,
        },
        "artifact_evidence": {
            "emitted": bool(args.emit_codec_artifacts),
            "manifest": "artifact_manifest.json" if args.emit_codec_artifacts else None,
            "payload_table": "artifact_payloads.csv" if args.emit_codec_artifacts else None,
            "strategy_count": len(artifact_rows),
            "alignment_bytes": int(args.artifact_alignment),
            "serialized_rate_cap_enforced": bool(args.enforce_serialized_rate_cap),
            "pre_evaluation_natural_file_bytes": pre_evaluation_artifact_sizes,
        },
        "hessian_proxy": (
            "input covariance C kron I_out; activation MSE proxy, not full task Hessian; "
            f"geometry mode={args.covariance_mode}, configured damping ratio="
            f"{args.covariance_damping_ratio:g}; empirical Gram collection uses a 1e-5 "
            "mean-diagonal ridge before this declared ablation; materially indefinite matrices "
            "fail; relative negative numerical PSD error <=1e-7 is repaired once for endpoint "
            "scoring and nonfactorizer repairs, with a float32 storage floor of eight machine "
            "epsilons; whitened SVD may use its separately recorded fitting floor; the spectrum "
            "is decomposed once and identity shifts are propagated algebraically"
        ),
        "covariance_psd_audit": {
            "path": "covariance_psd_audit.csv",
            "layer_count": len(covariance_psd_rows),
            "mode": args.covariance_mode,
            "configured_damping_ratio": args.covariance_damping_ratio,
            "psd_rejection_rtol": NUMERICAL_PSD_REJECTION_RTOL,
            "float32_storage_floor_rtol": FLOAT32_PSD_FLOOR_RTOL,
            "maximum_original_negative_relative": maximum_negative_relative,
            "maximum_diagonal_shift_relative": maximum_shift_relative,
            "all_endpoint_scoring_and_nonfactorizer_repairs_share_prepared_covariance": True,
            "lowrank_fit_uses_disclosed_factorizer_floor": args.l_method == "whitened_svd",
            "all_consumers_share_prepared_covariance": not any(
                bool(row.get("factorizer_regularization_applied"))
                for row in factorizer_audit.values()
            ),
        },
        "targets": args.target_ratios,
        "endpoint_target": args.endpoint_target,
        "data": {
            "requested": args.data_cfg,
            "source_used": args.text_source_used,
            "source_metadata": args.text_source_metadata,
            "fallback_allowed": False,
            "split_policy": args.text_split_policy,
            "role_splits": getattr(args, "data_role_splits", None),
            "text_pool_count": args.text_pool_count,
            "unique_text_pool_count": args.unique_text_pool_count,
            "calib_text_count": len(args.calib_texts),
            "selection_text_count": len(getattr(args, "selection_texts", [])),
            "eval_text_count": len(args.eval_texts),
            "recovery_text_count": len(args.recovery_texts),
            "eval_window_count": len(baseline_window_rows),
            "allocation_validation_window_count": len(
                [
                    row
                    for row in allocation_validation_window_rows
                    if row.get("proxy_rank") == 0
                ]
            ),
            "window_interval_semantics": "paired fixed-window mean +/- 1.96 standard errors; descriptive, not an independence-based population CI",
            "calib_digest": _text_digest(args.calib_texts),
            "selection_digest": (
                _text_digest(args.selection_texts)
                if hasattr(args, "selection_texts")
                else None
            ),
            "eval_digest": _text_digest(args.eval_texts),
            "identical_text_overlap_count": len(set(args.calib_texts).intersection(args.eval_texts)),
            "content_disjoint": not bool(set(args.calib_texts).intersection(args.eval_texts)),
            "selection_test_identical_text_overlap_count": len(
                set(getattr(args, "selection_texts", [])).intersection(args.eval_texts)
            ),
            "calibration_selection_identical_text_overlap_count": len(
                set(args.calib_texts).intersection(
                    getattr(args, "selection_texts", [])
                )
            ),
            "test_reserved_until_after_validation_selection": bool(
                args.two_stage_selection
            ),
        },
        "activation_counts": activation_counts,
        "factorizer_input_dims": factorizer_dims,
        "factorizer": {
            "method": args.l_method,
            "whitening_floor_ratio": args.whitening_floor_ratio,
            "floor_scope": "factorizer_only_endpoint_scored_with_declared_covariance",
            "per_layer_audit": factorizer_audit,
        },
        "rate_allocator": rate_allocator_report,
        "joint_value_claim": joint_value_claim,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": base.package_version("transformers"),
            "datasets": base.package_version("datasets"),
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "arguments": base.args_snapshot(args),
    }
    base.write_json(output_dir / "run_config.json", run_config)
    write_summary(
        output_dir,
        args=args,
        baseline_metrics=baseline_metrics,
        endpoint_rows=endpoint_rows,
        comfort_summary=comfort_summary,
    )
    if not args.skip_plots:
        plot_results(output_dir, endpoint_rows, comfort_rows, endpoint_target=args.endpoint_target)
    base.restore_weights(modules, baseline_weights)
    mark_output_complete(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
