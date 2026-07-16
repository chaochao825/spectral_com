from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_existing_compression_results",
    REPO_ROOT / "scripts" / "audit_existing_compression_results.py",
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_committed_qwen_payload_audit_reproduces_reference_values() -> None:
    run_dir = REPO_ROOT / "results" / "residual_stack_validate_Qwen_Qwen2-7B_20260707_014041"
    rows, interactions = AUDIT.audit_qwen_run(run_dir, 0.258, 0.01)
    by_method = {row["method"]: row for row in rows}
    assert by_method["Q+L"]["entropy_lower_bound_ratio"] == pytest.approx(0.25809151785714285)
    assert by_method["Q+S"]["entropy_lower_bound_ratio"] == pytest.approx(0.262480268673021)
    assert by_method["Q+S"]["csr16_ratio"] == pytest.approx(0.2668371395188935)
    assert by_method["Q+L"]["strict_target_match_csr16"] is True
    assert by_method["Q+S"]["strict_target_match_csr16"] is False
    q_s = [row for row in interactions if row["method"] == "Q+S" and row["pair"] == "Qerr,Sres"]
    assert len(q_s) == 1
    assert q_s[0]["rho_h"] < -0.5
    assert q_s[0]["interpretation"] == "repair_cancellation"


def test_qsl_is_not_a_stable_winner_across_committed_runs() -> None:
    rows = AUDIT.stability_audit(
        REPO_ROOT / "results" / "compare_7b_dam_residual_stack_20260707" / "strategy_comparison.csv"
    )
    qsl = [row for row in rows if row["method"] == "Q+S+L"]
    assert len(qsl) == 4
    assert sum(int(row["rank_within_core"]) == 1 for row in qsl) == 2
    qwen_mlp = next(row for row in qsl if row["run"] == "Qwen2-7B attn+MLP")
    assert qwen_mlp["qsl_gain_vs_best_single"] == pytest.approx(0.004646303724946676)
