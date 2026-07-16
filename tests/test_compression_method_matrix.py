from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_compression_method_matrix",
    REPO_ROOT / "scripts" / "build_compression_method_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


REQUIRED_METHODS = {
    "RTN", "GPTQ", "AWQ", "SparseGPT", "Wanda", "SpQR", "SqueezeLLM",
    "LQER", "QERA", "EoRA", "QuIP#", "QTIP", "AQLM", "OmniQuant",
    "SpinQuant", "SliderQuant", "LiftQuant", "D²Quant", "ADMM-Q", "HAS-VQ",
    "SEPTQ", "HESTIA", "AAAC", "DAQ",
    "Q-Palette", "SLiM", "OBR", "EfficientQAT", "LLM-QAT", "TurboQuant",
    "SharQ", "Joint structural pruning + MPQ", "Q-VDiT", "S2Q-VDiT",
    "QuantSparse", "TeaCache", "Sparse VideoGen", "Sparse-vDiT", "CacheQuant",
    "VMonarch", "MonarchRT", "RoPeSLR",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generator_writes_complete_valid_registry(tmp_path: Path) -> None:
    paths = MATRIX.write_outputs(tmp_path)
    assert {path.name for path in paths} == {"method_matrix.csv", "summary.md"}
    rows = _read_csv(tmp_path / "method_matrix.csv")

    assert set(rows[0]) == set(MATRIX.METHOD_COLUMNS)
    assert REQUIRED_METHODS <= {row["method"] for row in rows}
    assert {row["lane"] for row in rows} == {"A", "B", "C", "D"}
    assert {row["sub_lane"] for row in rows} == {"A0", "A1", "B", "C", "D"}
    assert len({row["method_id"] for row in rows}) == len(rows)
    assert all(all(row[column].strip() for column in MATRIX.METHOD_COLUMNS) for row in rows)
    assert all(row["verified_as_of"] == MATRIX.VERIFIED_AS_OF for row in rows)


def test_liftquant_is_exactly_two_separate_protocol_rows(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    lift = [row for row in _read_csv(tmp_path / "method_matrix.csv") if row["method"] == "LiftQuant"]
    assert len(lift) == 2
    assert {(row["variant"], row["lane"], row["sub_lane"]) for row in lift} == {
        ("block correction only", "B", "B"),
        ("optional end-to-end correction", "C", "C"),
    }
    block = next(row for row in lift if row["lane"] == "B")
    e2e = next(row for row in lift if row["lane"] == "C")
    assert "4096 RedPajama samples of length 2048" in block["gradient_training_signal"]
    assert "2 epochs" in block["gradient_training_signal"]
    assert "T*=M T^-1" in block["updated_state"]
    assert "4096 samples of length 4096" in e2e["gradient_training_signal"]
    assert "1 epoch" in e2e["gradient_training_signal"]
    assert "cross-entropy" in e2e["gradient_training_signal"]
    assert "4-bit Q/S/L probe is mechanism-only" in block["comparison_conditions"]
    assert "nsamples1/nsamples2" in block["gradient_training_signal"]
    assert "compatibility_patched_layer0_smoke_passed" in block["reproduction_status"]
    assert "e2e_entrypoint_blocked_missing_datautils_block" in e2e["reproduction_status"]
    assert "never double-count" in block["payload_must_count"]
    assert "unfused M plus inverse-whitening factors" in e2e["payload_must_count"]


def test_sliderquant_and_q_palette_protocols_are_split(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    rows = _read_csv(tmp_path / "method_matrix.csv")

    sliders = [row for row in rows if row["method"] == "SliderQuant"]
    assert {(row["variant"], row["sub_lane"]) for row in sliders} == {
        ("default; fused channel scales and rank-4 LoRA", "B"),
        ("SliderQuant+ with runtime rotations", "B"),
    }
    default = next(row for row in sliders if row["method_id"] == "sliderquant")
    plus = next(row for row in sliders if row["method_id"] == "sliderquant_plus")
    assert "128 samples of length 2048" in default["gradient_training_signal"]
    assert "20 epochs" in default["gradient_training_signal"]
    assert "60 for W2A16" in default["gradient_training_signal"]
    assert "rank-4 LoRA" in default["updated_state"]
    assert "non-absorbable Hadamard" in plus["payload_must_count"]

    palettes = [row for row in rows if row["method"] == "Q-Palette"]
    assert {(row["method_id"], row["sub_lane"]) for row in palettes} == {
        ("q_palette", "A0"),
        ("q_palette_data_aware", "A1"),
    }
    aware = next(row for row in palettes if row["sub_lane"] == "A1")
    assert "proxy Hessian" in aware["gradient_training_signal"]
    assert "validation perplexity" in aware["objective"]
    assert "fusion/merge plan" in aware["payload_must_count"]


def test_scope_and_payload_boundaries_are_explicit(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    rows = _read_csv(tmp_path / "method_matrix.csv")
    by_id = {row["method_id"]: row for row in rows}

    assert by_id["turboquant"]["strict_ptq_direct_comparison"] == "no_scope_mismatch"
    assert by_id["sharq"]["lane"] == "D"
    assert by_id["bitnet_b158"]["lane"] == "D"
    assert by_id["rtn"]["reproduction_status"] == (
        "native_baseline_implemented; measured_in_existing_exploratory_runs"
    )
    assert by_id["d2quant"]["sub_lane"] == "A1"
    assert by_id["squeezellm"]["sub_lane"] == "B"
    assert "performs backward" in by_id["squeezellm"]["gradient_training_signal"]
    assert "Fisher-based sensitivity checkpoint" in by_id["squeezellm"]["gradient_training_signal"]
    assert "do not compare this main Fisher path as no-backward A1 PTQ" in by_id["squeezellm"]["comparison_conditions"]
    assert by_id["spinquant"]["sub_lane"] == "C"
    assert by_id["spinquant"]["strict_ptq_direct_comparison"] == "mechanism_only"
    assert "Trainer" in by_id["spinquant"]["gradient_training_signal"]
    assert "causal-LM loss" in by_id["spinquant"]["gradient_training_signal"]
    assert "DAC LayerNorm bias" in by_id["d2quant"]["payload_must_count"]
    assert "64 bytes" in by_id["aaac"]["payload_must_count"]
    assert "actual serialized bytes" in by_id["q_palette"]["comparison_conditions"]
    assert by_id["foem"]["official_repo"] == "https://github.com/Xingyu-Zheng/FOEM"
    assert "GPTQModel is only a secondary integration" in by_id["foem"]["evidence_note"]
    assert "without gradient learning" in by_id["yaqa"]["gradient_training_signal"]
    assert "absent at inference" in by_id["yaqa"]["payload_must_count"]
    assert by_id["srr"]["variant"] == "Structured Residual Reconstruction PTQ"
    assert "shared FP4 weights" in by_id["sharq"]["variant"]
    assert all("padding" in row["payload_must_count"].lower() or row["lane"] == "D" for row in rows)
    assert all(
        "no_official_repo_found_as_of_2026-07-13" not in row["reproduction_status"]
        for row in rows
    )
    assert all(
        "external_reproduction_pending" in row["reproduction_status"]
        for row in rows
        if "literature_only" in row["reproduction_status"]
    )


def test_2026_hessian_methods_separate_encoder_optimization_from_qat(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    by_id = {row["method_id"]: row for row in _read_csv(tmp_path / "method_matrix.csv")}

    admm = by_id["admm_q"]
    assert admm["sub_lane"] == "A1"
    assert "no model backpropagation" in admm["gradient_training_signal"]
    assert "ADMM primal/dual iterates" in admm["updated_state"]
    assert admm["official_repo"] == "not_found_in_primary_sources"

    has_vq = by_id["has_vq"]
    assert has_vq["sub_lane"] == "A1"
    assert "calibration forwards" in has_vq["gradient_training_signal"]
    assert "codebook centroid" in has_vq["payload_must_count"]
    assert "sparse residual values plus indices" in has_vq["payload_must_count"]
    assert has_vq["official_repo"] == "https://github.com/VladimerKhasia/HASVQ"

    septq = by_id["septq"]
    assert septq["sub_lane"] == "A1"
    assert "no gradient training or STE" in septq["gradient_training_signal"]
    assert "reserved higher-precision value" in septq["payload_must_count"]
    assert "support mask/indices" in septq["payload_must_count"]
    assert septq["official_repo"] == "not_found_in_primary_sources"

    hestia = by_id["hestia"]
    assert hestia["sub_lane"] == "C"
    assert "AdamW" in hestia["gradient_training_signal"]
    assert "10B Ultra-FineWeb tokens" in hestia["gradient_training_signal"]
    assert "packed ternary codes" in hestia["payload_must_count"]
    assert hestia["official_repo"] == "https://github.com/hestia2026/Hestia"

    for method_id in ("admm_q", "has_vq", "septq", "hestia"):
        assert "external_reproduction_pending" in by_id[method_id]["reproduction_status"]


def test_multimodal_rows_preserve_scope_and_release_boundaries(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    by_id = {row["method_id"]: row for row in _read_csv(tmp_path / "method_matrix.csv")}
    multimodal_ids = {
        "q_vdit", "s2q_vdit", "quantsparse", "teacache", "sparse_videogen",
        "sparse_vdit", "cachequant", "vmonarch", "monarchrt", "ropeslr",
    }

    for method_id in multimodal_ids:
        row = by_id[method_id]
        assert row["lane"] == "D"
        assert row["sub_lane"] == "D"
        assert row["strict_ptq_direct_comparison"] == "no_scope_mismatch"

    assert "code_not_released_as_of_2026-07-16" in by_id["quantsparse"]["reproduction_status"]
    assert "code_not_released_as_of_2026-07-16" in by_id["s2q_vdit"]["reproduction_status"]
    assert "official_code_available" in by_id["q_vdit"]["reproduction_status"]
    assert "runtime sparse attention" in by_id["s2q_vdit"]["comparison_conditions"]
    assert "jointly optimize temporal reuse and structural quantization" in by_id["cachequant"]["objective"]
    assert "cache-only" in by_id["teacache"]["evidence_note"]
    assert "sparse semantic spikes plus low-rank background" in by_id["ropeslr"]["variant"]
    assert "never multiply standalone speedups" in by_id["quantsparse"]["comparison_conditions"]


def test_check_detects_drift(tmp_path: Path) -> None:
    MATRIX.write_outputs(tmp_path)
    MATRIX.write_outputs(tmp_path, check=True)
    (tmp_path / "summary.md").write_text("stale\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="drift detected"):
        MATRIX.write_outputs(tmp_path, check=True)


def test_committed_outputs_match_generator() -> None:
    output_dir = REPO_ROOT / "results" / "compression_method_comparison_20260713"
    MATRIX.write_outputs(output_dir, check=True)
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Multimodal and video-system supplement" in summary
    assert "Standalone speedups must never be multiplied" in summary
    assert "SqueezeLLM 主 Fisher 路径（B）" in summary
    assert "SpinQuant optimized rotation（C）" in summary
