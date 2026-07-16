from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "pretrained_hessian_repair_pythia70m_20260713"


def _csv(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pretrained_probe_uses_real_content_disjoint_evidence() -> None:
    config = json.loads((RESULT_DIR / "run_config.json").read_text(encoding="utf-8"))
    data = config["data"]
    assert data["source_used"] == "dataset:wikitext"
    assert data["fallback_allowed"] is False
    assert data["split_policy"] == "content_disjoint_sequential_text_windows"
    assert data["content_disjoint"] is True
    assert data["identical_text_overlap_count"] == 0
    assert data["calib_digest"] != data["eval_digest"]
    assert data["eval_window_count"] == 16
    assert config["actual_eval_tokens"] == 2032
    assert config["baseline_metrics"]["tokens"] == 2032


def test_equal_bit_combination_is_auditable_and_not_overclaimed() -> None:
    endpoints = {
        row["strategy"]: row
        for row in _csv("strategy_endpoints.csv")
        if float(row["target_ratio"]) == pytest.approx(0.258)
    }
    ql = endpoints["Q+L"]
    matched = endpoints["Q+S+L_QL_budget"]
    scaled = endpoints["Q+S+L_QL_budget_component_scale"]

    assert int(float(ql["payload_bits"])) == int(float(matched["payload_bits"]))
    assert int(float(ql["payload_bits"])) == int(float(scaled["payload_bits"]))
    assert int(float(scaled["comparison_budget_bits"])) == int(float(ql["payload_bits"]))
    assert scaled["rate_cap_satisfied"] == "True"
    assert float(scaled["normalized_hessian_cost"]) < float(ql["normalized_hessian_cost"])
    assert float(scaled["hessian_gain_per_added_bit"]) > float(ql["hessian_gain_per_added_bit"])
    assert float(scaled["perplexity_delta"]) < float(ql["perplexity_delta"])
    assert abs(float(scaled["rho_sl"])) <= 0.1
    assert float(scaled["cancellation_gain_qs_over_q"]) > 0.0
    assert float(scaled["cancellation_gain_ql_over_q"]) > 0.0
    assert int(float(scaled["layers_both_s_l_active"])) == 3


def test_window_rows_reproduce_endpoint_aggregates_and_intervals() -> None:
    windows = _csv("endpoint_window_nll.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in windows:
        grouped[row["strategy"]].append(row)
    assert set(grouped) >= {
        "dense",
        "Q+L",
        "Q+S+L_QL_budget_component_scale",
        "Q+S",
        "Q+S_OBS",
    }
    assert all(len(rows) == 16 for rows in grouped.values())
    assert len(windows) == 16 * len(grouped)

    endpoints = {
        row["strategy"]: row
        for row in _csv("strategy_endpoints.csv")
        if float(row["target_ratio"]) == pytest.approx(0.258)
    }
    baseline = {int(row["window_index"]): float(row["nll"]) for row in grouped["dense"]}
    for strategy, endpoint in endpoints.items():
        rows = sorted(grouped[strategy], key=lambda row: int(row["window_index"]))
        weighted_nll = sum(float(row["nll_sum"]) for row in rows) / sum(int(row["tokens"]) for row in rows)
        assert weighted_nll == pytest.approx(float(endpoint["heldout_nll"]), abs=1e-12)
        assert math.exp(float(endpoint["heldout_nll"])) == pytest.approx(
            float(endpoint["heldout_perplexity"]), rel=1e-12
        )
        deltas = [float(row["nll"]) - baseline[int(row["window_index"])] for row in rows]
        assert sum(deltas) / len(deltas) == pytest.approx(
            float(endpoint["paired_window_nll_delta_mean"]), abs=1e-12
        )
        assert math.isfinite(float(endpoint["paired_window_nll_delta_ci95_low"]))
        assert math.isfinite(float(endpoint["paired_window_nll_delta_ci95_high"]))

    def paired_interval(left: str, right: str) -> tuple[float, float, int]:
        left_rows = {int(row["window_index"]): float(row["nll"]) for row in grouped[left]}
        right_rows = {int(row["window_index"]): float(row["nll"]) for row in grouped[right]}
        deltas = [left_rows[index] - right_rows[index] for index in sorted(left_rows)]
        mean = statistics.mean(deltas)
        se = statistics.stdev(deltas) / math.sqrt(len(deltas))
        return mean - 1.96 * se, mean + 1.96 * se, sum(delta < 0.0 for delta in deltas)

    combo_low, combo_high, combo_wins = paired_interval("Q+S+L_QL_budget_component_scale", "Q+L")
    assert combo_low < 0.0 < combo_high
    assert combo_wins == 11
    obs_low, obs_high, obs_wins = paired_interval("Q+S_OBS", "Q+S")
    assert obs_low < obs_high < 0.0
    assert obs_wins == 15


def test_committed_probe_artifacts_are_complete() -> None:
    required = {
        "candidate_ablation.csv",
        "strategy_endpoints.csv",
        "endpoint_window_nll.csv",
        "comfort_sweep.csv",
        "comfort_summary.csv",
        "run_config.json",
        "summary.md",
        "formal_run.log",
        "figures/pretrained_hessian_repair_probe.png",
        "figures/pretrained_hessian_repair_probe.pdf",
    }
    for name in required:
        path = RESULT_DIR / name
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
