from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "pretrained_hessian_repair_pythia70m_serialized_20260714"
SPEC = importlib.util.spec_from_file_location(
    "summarize_serialized_hessian_result",
    REPO_ROOT / "scripts" / "summarize_serialized_hessian_result.py",
)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUMMARY
SPEC.loader.exec_module(SUMMARY)


def _raw_codec_artifacts_available() -> bool:
    manifest = json.loads(
        (RESULT_DIR / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    relative_paths = [manifest["reference"]["path"]]
    relative_paths.extend(row["artifact_path"] for row in manifest["strategies"])
    return all((RESULT_DIR / relative_path).is_file() for relative_path in relative_paths)


requires_raw_codec_artifacts = pytest.mark.skipif(
    not _raw_codec_artifacts_available(),
    reason=(
        "raw .hrc payloads are intentionally omitted from the publication clone; "
        "artifact_manifest.json retains their declared bytes and SHA-256 digests"
    ),
)


def test_exact_two_sided_sign_test_is_finite_sample_and_excludes_ties() -> None:
    four_of_sixteen = SUMMARY.exact_two_sided_sign_test([-1.0] * 4 + [1.0] * 12)
    assert four_of_sixteen == {
        "left_wins_lower_nll": 4,
        "right_wins_lower_nll": 12,
        "ties": 0,
        "non_tied_window_count": 16,
        "exact_two_sided_sign_p": pytest.approx(0.076812744140625),
    }
    all_left_with_ties = SUMMARY.exact_two_sided_sign_test([-1.0] * 16 + [0.0, 0.0])
    assert all_left_with_ties["ties"] == 2
    assert all_left_with_ties["non_tied_window_count"] == 16
    assert all_left_with_ties["exact_two_sided_sign_p"] == pytest.approx(2 / (2**16))
    assert SUMMARY.exact_two_sided_sign_test([0.0, 0.0])["exact_two_sided_sign_p"] == 1.0


@requires_raw_codec_artifacts
def test_committed_serialized_summary_validates_exact_physical_rates() -> None:
    summary = SUMMARY.build_summary(RESULT_DIR)
    execution = summary["execution_evidence"]
    assert execution["status"] == "COMPLETED"
    assert execution["exit_code"] == 0
    assert execution["elapsed_seconds"] == pytest.approx(350.30)
    assert execution["maxrss_kb"] == 2_414_092
    assert all(len(record["sha256"]) == 64 for record in execution["files"].values())
    validation = summary["physical_rate_validation"]
    assert validation["production_backend"] is False
    assert validation["endpoint_manifest_fields_consistent"] is True
    assert validation["artifact_file_sizes_match_manifest"] is True
    assert validation["artifact_sha256_digests_match_manifest"] is True
    assert validation["reference_artifact_file_bytes"] == 12_586_048
    assert validation["ql_reference"] == {
        "strategy": "Q+L",
        "artifact_file_bytes": 3_248_832,
        "artifact_natural_file_bytes": 3_248_832,
    }
    for strategy in (
        "Q+S+L_QL_budget",
        "Q+S+L_QL_budget_component_scale",
    ):
        control = validation["strict_equal_byte_controls"][strategy]
        assert control["artifact_file_bytes"] == 3_248_832
        assert control["artifact_natural_file_bytes"] == 3_233_152
        assert control["tail_padding_bytes"] == 15_680
        assert control["file_byte_difference_vs_ql"] == 0
        assert control["natural_file_byte_difference_vs_ql"] == -15_680
    assert validation["unconstrained_scaled_qsl"]["artifact_file_bytes"] == 3_253_888
    assert validation["unconstrained_scaled_qsl"]["file_byte_overage_vs_ql"] == 5_056


@requires_raw_codec_artifacts
def test_paired_comparisons_reproduce_direction_and_uncertainty_boundaries() -> None:
    rows = {
        row["comparison_id"]: row
        for row in SUMMARY.build_summary(RESULT_DIR)["comparisons"]
    }
    assert list(rows) == [spec.comparison_id for spec in SUMMARY.COMPARISONS]

    constrained_scaled = rows["constrained_scaled_qsl_vs_ql"]
    assert constrained_scaled["paired_mean_nll_difference"] == pytest.approx(0.007117924727793)
    assert constrained_scaled["normal_95_ci_low"] == pytest.approx(-0.000121426, abs=1e-9)
    assert constrained_scaled["normal_95_ci_high"] == pytest.approx(0.014357276, abs=1e-9)
    assert constrained_scaled["left_wins_lower_nll"] == 4
    assert constrained_scaled["exact_two_sided_sign_p"] == pytest.approx(0.076812744140625)
    assert constrained_scaled["artifact_file_byte_difference"] == 0
    assert constrained_scaled["endpoint_perplexity_difference"] == pytest.approx(0.55799326993433)
    assert constrained_scaled["uncertainty_label"] == "inconclusive"

    constrained_unscaled = rows["constrained_unscaled_qsl_vs_ql"]
    assert constrained_unscaled["paired_mean_nll_difference"] == pytest.approx(0.008383908609383)
    assert constrained_unscaled["normal_95_ci_low"] > 0.0
    assert constrained_unscaled["left_wins_lower_nll"] == 4
    assert constrained_unscaled["uncertainty_label"] == "inconclusive"

    unconstrained = rows["unconstrained_scaled_qsl_vs_ql"]
    assert unconstrained["paired_mean_nll_difference"] == pytest.approx(-0.003664737611305)
    assert unconstrained["normal_95_ci_low"] < 0.0 < unconstrained["normal_95_ci_high"]
    assert unconstrained["left_wins_lower_nll"] == 10
    assert unconstrained["artifact_file_byte_difference"] == 5_056
    assert unconstrained["uncertainty_label"] == "inconclusive"

    obs = rows["obs_vs_qs"]
    assert obs["paired_mean_nll_difference"] == pytest.approx(-0.02685603945274)
    assert obs["left_wins_lower_nll"] == 16
    assert obs["normal_95_ci_high"] < 0.0
    assert obs["exact_two_sided_sign_p"] == pytest.approx(2 / (2**16))
    assert obs["artifact_file_byte_difference"] == 0
    assert obs["uncertainty_label"] == "fixed_window_diagnostics_agree"

    block_scale = rows["block_scale_vs_q"]
    assert block_scale["paired_mean_nll_difference"] == pytest.approx(-0.07442798764687)
    assert block_scale["left_wins_lower_nll"] == 16
    assert block_scale["normal_95_ci_high"] < 0.0
    assert block_scale["artifact_file_byte_difference"] == 82_944
    assert block_scale["uncertainty_label"] == "fixed_window_diagnostics_agree"

    for comparison_id, row in rows.items():
        assert "non-causal" in row["interpretation"], comparison_id
        assert "not a population-level confidence claim" in row["interpretation"], comparison_id
    for comparison_id in (
        "constrained_scaled_qsl_vs_ql",
        "constrained_unscaled_qsl_vs_ql",
        "unconstrained_scaled_qsl_vs_ql",
    ):
        assert "inconclusive" in rows[comparison_id]["interpretation"]


@requires_raw_codec_artifacts
def test_generated_outputs_are_exact_and_machine_readable() -> None:
    SUMMARY.write_outputs(RESULT_DIR, check=True)
    json_text = (RESULT_DIR / "serialized_rate_summary.json").read_text(encoding="utf-8")
    assert '"schema_version": "serialized_hessian_paired_summary.v1"' in json_text
    assert '"inference_limit"' in json_text
    with (RESULT_DIR / "paired_method_comparisons.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert list(rows[0]) == list(SUMMARY.CSV_FIELDS)
    assert {row["comparison_id"] for row in rows} == {
        spec.comparison_id for spec in SUMMARY.COMPARISONS
    }
