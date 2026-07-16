from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "run_pretrained_hessian_repair_artifact_test",
    REPO_ROOT / "scripts" / "run_pretrained_hessian_repair.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _q(shape: tuple[int, int]) -> object:
    codes = np.zeros(shape, dtype=np.int8)
    scales = np.ones(shape[0], dtype=np.float16)
    return RUNNER.QuantCodec(codes=codes, scales=scales, bits=4)


def test_endpoint_scoring_and_nonfactorizer_repairs_share_prepared_psd_metric() -> None:
    covariance = torch.diag(torch.tensor([13.5, -1.7e-7], dtype=torch.float32))
    prepared, prepared_input, report = RUNNER.prepare_metric_covariance(covariance)
    prepared_np = prepared.double().numpy()
    assert np.linalg.eigvalsh(prepared_np).min() >= 0.0
    assert report["original_min_relative"] == pytest.approx(-1.7e-7 / 13.5, rel=1e-5)
    assert 0.0 < report["diagonal_shift_relative"] < 2e-6

    delta = np.array([[0.25, -0.5]], dtype=np.float32)
    metric = RUNNER.HessianMetric(prepared, device="cpu")
    expected = 0.5 * float(np.sum((delta @ prepared_np) * delta))
    assert metric.cost(delta) == pytest.approx(expected, rel=1e-6, abs=1e-12)
    RUNNER.obs_retained_support_correction(
        delta, np.array([[True, False]]), prepared_input
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        RUNNER.prepare_metric_covariance(
            torch.diag(torch.tensor([1.0, -1e-3], dtype=torch.float32))
        )


def test_serialized_sparse_search_finds_the_physical_boundary() -> None:
    shape = (4, 16)
    q = _q(shape)
    budget = RUNNER._allocation_artifact_file_bytes(
        layer="layer",
        q=q,
        shape=shape,
        sparse_nonzero=7,
        rank=1,
        alignment=64,
    )
    selected = RUNNER._max_sparse_under_serialized_budget(
        layer="layer",
        q=q,
        shape=shape,
        rank=1,
        logical_max_nonzero=shape[0] * shape[1],
        budget_file_bytes=budget,
        alignment=64,
    )
    assert selected >= 7
    assert RUNNER._allocation_artifact_file_bytes(
        layer="layer",
        q=q,
        shape=shape,
        sparse_nonzero=selected,
        rank=1,
        alignment=64,
    ) <= budget
    if selected < shape[0] * shape[1]:
        assert RUNNER._allocation_artifact_file_bytes(
            layer="layer",
            q=q,
            shape=shape,
            sparse_nonzero=selected + 1,
            rank=1,
            alignment=64,
        ) > budget


def test_endpoint_artifacts_roundtrip_and_equalize_ql_bytes(tmp_path: Path) -> None:
    shape = (16, 64)
    q = _q(shape)
    weight = np.zeros(shape, dtype=np.float32)
    ql = RUNNER.Candidate(
        strategy="Q+L",
        target_ratio=0.5,
        layer="layer",
        weight=weight,
        q=q,
        lowrank=RUNNER.LowRankCodec(
            np.zeros((shape[0], 8), dtype=np.float16),
            np.zeros((8, shape[1]), dtype=np.float16),
        ),
    )
    ql_budget = RUNNER.codec_artifact_natural_file_bytes(
        [RUNNER._artifact_layer(ql)], alignment=64
    )
    nnz = RUNNER._max_sparse_under_serialized_budget(
        layer="layer",
        q=q,
        shape=shape,
        rank=1,
        logical_max_nonzero=shape[0] * shape[1],
        budget_file_bytes=ql_budget,
        alignment=64,
    )
    assert nnz > 0
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[:nnz] = True
    qsl = RUNNER.Candidate(
        strategy="Q+S+L_QL_budget_component_scale",
        target_ratio=0.5,
        layer="layer",
        weight=weight,
        q=q,
        sparse=RUNNER.SparseCodec(np.zeros(shape, dtype=np.float16), mask),
        lowrank=RUNNER.LowRankCodec(
            np.zeros((shape[0], 1), dtype=np.float16),
            np.zeros((1, shape[1]), dtype=np.float16),
        ),
    )
    endpoint_rows = [
        {"strategy": "Q+L", "target_ratio": 0.5},
        {"strategy": "Q+S_OBS_global", "target_ratio": 0.5},
        {"strategy": "Q+L_global", "target_ratio": 0.5},
        {"strategy": "Q+S_OBS_or_L_global", "target_ratio": 0.5},
        {"strategy": qsl.strategy, "target_ratio": 0.5},
    ]
    candidates = {"Q+L": {"layer": ql}, qsl.strategy: {"layer": qsl}}
    candidates["Q+S+L_QL_budget"] = {"layer": RUNNER.Candidate(
        strategy="Q+S+L_QL_budget",
        target_ratio=0.5,
        layer="layer",
        weight=weight,
        q=q,
        sparse=qsl.sparse,
        lowrank=qsl.lowrank,
    )}
    candidates["Q+S_OBS_global"] = {"layer": RUNNER.Candidate(
        strategy="Q+S_OBS_global",
        target_ratio=0.5,
        layer="layer",
        weight=weight,
        q=q,
        sparse=qsl.sparse,
    )}
    candidates["Q+L_global"] = {"layer": RUNNER._candidate_with_strategy(
        ql, "Q+L_global"
    )}
    candidates["Q+S_OBS_or_L_global"] = {
        "layer": RUNNER._candidate_with_strategy(
            ql, "Q+S_OBS_or_L_global"
        )
    }
    sizes = RUNNER.validate_endpoint_serialized_rate_cap(candidates, alignment=64)
    assert sizes[qsl.strategy] <= sizes["Q+L"]
    rows = RUNNER.emit_endpoint_codec_artifacts(
        tmp_path,
        baseline_weights={"layer": torch.zeros(shape, dtype=torch.float16)},
        endpoint_candidates={
            "Q+L": {"layer": ql},
            "Q+S_OBS_global": candidates["Q+S_OBS_global"],
            "Q+L_global": candidates["Q+L_global"],
            "Q+S_OBS_or_L_global": candidates["Q+S_OBS_or_L_global"],
            qsl.strategy: {"layer": qsl},
        },
        endpoint_rows=endpoint_rows,
        endpoint_target=0.5,
        alignment=64,
        enforce_serialized_rate_cap=True,
    )
    by_strategy = {row["strategy"]: row for row in rows}
    assert by_strategy["Q+L"]["artifact_file_bytes"] == by_strategy[qsl.strategy]["artifact_file_bytes"]
    assert by_strategy[qsl.strategy]["same_physical_bytes_as_ql"] is True
    assert by_strategy["Q+S_OBS_global"]["same_physical_bytes_as_ql"] is True
    assert by_strategy["Q+L_global"]["same_physical_bytes_as_ql"] is True
    assert by_strategy["Q+S_OBS_or_L_global"]["same_physical_bytes_as_ql"] is True
    assert by_strategy[qsl.strategy]["roundtrip_exact_fp16_endpoint"] is True
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["serialized_rate_cap_enforced"] is True
    assert manifest["production_backend"] is False


def test_runner_rejects_unrepresentable_signed_one_bit_rtn() -> None:
    with pytest.raises(ValueError, match=r"\[2, 8\].*\{-1, 0, \+1\}"):
        RUNNER.validate_research_codec_bits(1)
    assert RUNNER.validate_research_codec_bits(2) == 2
    assert RUNNER.validate_research_codec_bits(8) == 8


def test_serialized_cap_rejects_missing_candidate_layers() -> None:
    shape = (2, 4)
    weight = np.zeros(shape, dtype=np.float32)

    def candidate(strategy: str, layer: str) -> object:
        return RUNNER.Candidate(strategy, 0.5, layer, weight, _q(shape))

    candidates = {
        "Q+L": {
            "layer.a": candidate("Q+L", "layer.a"),
            "layer.b": candidate("Q+L", "layer.b"),
        },
        "Q+S+L_QL_budget": {
            "layer.a": candidate("Q+S+L_QL_budget", "layer.a"),
        },
        "Q+S+L_QL_budget_component_scale": {
            "layer.a": candidate("Q+S+L_QL_budget_component_scale", "layer.a"),
            "layer.b": candidate("Q+S+L_QL_budget_component_scale", "layer.b"),
        },
    }
    with pytest.raises(AssertionError, match=r"layer-set mismatch.*missing=\['layer.b'\]"):
        RUNNER.validate_endpoint_serialized_rate_cap(candidates, alignment=64)


def test_output_directory_markers_fail_closed_on_reuse(tmp_path: Path) -> None:
    output = RUNNER.prepare_fresh_output_dir(tmp_path / "run")
    assert (output / "RUNNING").is_file()
    RUNNER.mark_output_complete(output)
    assert not (output / "RUNNING").exists()
    assert (output / "COMPLETED").is_file()
    with pytest.raises(FileExistsError, match="absent or empty"):
        RUNNER.prepare_fresh_output_dir(output)
