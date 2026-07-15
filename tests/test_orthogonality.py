import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from llm_spectral_dynamics.structured.orthogonality import (
    empirical_additivity_error,
    hessian_cosine,
    hessian_inner,
    rankdata,
    spearmanr,
    spectrum_summary,
)


def _load_pretrained_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_pretrained_llm_orthogonality.py"
    spec = importlib.util.spec_from_file_location("run_pretrained_llm_orthogonality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            return None
        raise
    return module


def test_hessian_cosine_uses_diagonal_curvature():
    left = np.array([1.0, 0.0])
    right = np.array([1.0, 1.0])
    hessian = np.array([4.0, 1.0])
    assert hessian_inner(left, right, hessian) == 4.0
    assert math.isclose(hessian_cosine(left, right, hessian), 4.0 / math.sqrt(4.0 * 5.0))


def test_hessian_cosine_clamps_roundoff():
    values = np.array([1.0, 2.0, 3.0])
    hessian = np.ones_like(values)
    assert hessian_cosine(values, values, hessian) == 1.0


def test_empirical_additivity_error_is_signed_and_normalized():
    value = empirical_additivity_error(10.0, 11.0, 12.0, 13.5)
    assert math.isclose(value, 0.5 / 3.0)


def test_spearmanr_handles_ties_with_average_ranks():
    assert np.allclose(rankdata([3.0, 1.0, 1.0]), [3.0, 1.5, 1.5])
    rho, n = spearmanr([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert n == 3
    assert math.isclose(rho, 1.0)


def test_spectrum_summary_reports_energy_ranks():
    matrix = np.diag([4.0, 1.0, 0.1])
    summary = spectrum_summary(matrix)
    assert summary["rank_90"] == 1
    assert summary["rank_99"] >= 2
    assert summary["top1_energy"] > 0.94


def test_spq_layer_family_and_ops_for_llama_and_pythia_names():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    assert runner.layer_family("model.layers.0.self_attn.q_proj") == "attention"
    assert runner.spq_ops_for_layer("model.layers.0.self_attn.q_proj") == ("r", "q")
    assert runner.layer_family("model.layers.0.mlp.up_proj") == "mlp"
    assert runner.spq_ops_for_layer("model.layers.0.mlp.up_proj") == ("s", "q")
    assert runner.layer_family("gpt_neox.layers.3.attention.query_key_value") == "attention"
    assert runner.layer_family("gpt_neox.layers.3.mlp.dense_h_to_4h") == "mlp"


def test_rotated_rtn_helpers_preserve_shape_and_parse_orders():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    import torch

    assert runner.next_power_of_two(513) == 1024
    matrix = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    recovered = runner.fwht_last_dim(runner.fwht_last_dim(matrix))
    assert torch.allclose(recovered, matrix, atol=1e-5)
    weight = torch.randn(3, 5, dtype=torch.float32)
    quantized = runner.rotated_rtn_quantize(weight, bits=4)
    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype
    assert runner.parse_order_candidates("qsr,rsq") == [("q", "s", "r"), ("r", "s", "q")]


def test_lossless_frontier_specs_and_summary():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    import torch

    args = SimpleNamespace(
        frontier_bits_list="8,6",
        frontier_keep_list="0.995,0.9",
        frontier_rank_list="0.995,0.5",
        frontier_q_methods="rtn",
        frontier_s_methods="wanda",
        frontier_r_methods="svd",
        frontier_orders="qsr,rqs",
        frontier_max_triple_candidates=3,
    )
    weights = {"layer": torch.zeros(4, 4)}
    specs = runner.lossless_frontier_candidate_specs(args, weights)
    families = {spec["family"] for spec in specs}
    assert {"q_only", "s_only", "r_only", "qsr_stack"} <= families
    assert sum(1 for spec in specs if spec["family"] == "qsr_stack") == 3
    qsr_spec = next(spec for spec in specs if spec["family"] == "qsr_stack")
    assert qsr_spec["frontier_triple_full_grid_count"] == 16
    assert qsr_spec["frontier_triple_evaluated_count"] == 3

    base = {
        "strategy": "lossless_frontier",
        "benchmark_drop_fraction": 0.0,
        "benchmark_drop_percent": 0.0,
        "lossless_threshold_percent": 1.0,
        "lossless_pass": True,
        "predicted_hessian_cost": 0.0,
        "perplexity": 1.0,
        "order": "",
        "q_method": "",
        "s_method": "",
        "r_method": "",
    }
    rows = [
        {**base, "family": "q_only", "nominal_bits": 8, "nominal_keep_fraction": 1.0, "nominal_rank_fraction": 1.0, "nominal_memory_ratio": 0.5},
        {**base, "family": "s_only", "nominal_bits": 16, "nominal_keep_fraction": 0.9, "nominal_rank_fraction": 1.0, "nominal_memory_ratio": 0.9},
        {**base, "family": "r_only", "nominal_bits": 16, "nominal_keep_fraction": 1.0, "nominal_rank_fraction": 0.5, "nominal_memory_ratio": 1.0},
        {**base, "family": "qsr_stack", "nominal_bits": 8, "nominal_keep_fraction": 0.9, "nominal_rank_fraction": 0.5, "nominal_memory_ratio": 0.225},
    ]
    summary = runner.summarize_lossless_frontier(rows)
    qsr = next(row for row in summary if row["family"] == "qsr_stack")
    assert qsr["beats_best_single_nominal"] is True
    assert math.isclose(qsr["best_single_nominal_memory_ratio"], 0.5)

    failed_rows = [
        {**base, "family": "q_only", "lossless_pass": False, "benchmark_drop_fraction": 0.20, "benchmark_drop_percent": 20.0, "nominal_bits": 3, "nominal_keep_fraction": 1.0, "nominal_rank_fraction": 1.0, "nominal_memory_ratio": 0.1875},
        {**base, "family": "q_only", "lossless_pass": False, "benchmark_drop_fraction": 0.05, "benchmark_drop_percent": 5.0, "nominal_bits": 8, "nominal_keep_fraction": 1.0, "nominal_rank_fraction": 1.0, "nominal_memory_ratio": 0.5},
    ]
    failed_summary = runner.summarize_lossless_frontier(failed_rows)
    assert failed_summary[0]["frontier_status"] == "no_pass_best_drop"
    assert failed_summary[0]["nominal_bits"] == 8


def test_fair_benchmark_specs_are_fixed_and_cover_families():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    specs = runner.fair_benchmark_specs()
    names = [spec["strategy"] for spec in specs]
    families = {spec["family"] for spec in specs}
    assert {"q_only", "s_only", "r_only", "qsr_stack"} <= families
    assert "q_only_rtn_4bit" in names
    assert "q_only_rotated_4bit" in names
    assert "qsr_naive_rtn_magnitude_svd" in names
    assert "qsr_rotated_wanda_whitened" in names
    assert "rqs_rotated_wanda_whitened" in names
    assert "hessian_guided_qsr_budget" in names
    assert all("benchmark_drop" not in spec for spec in specs)


def test_fair_extended_recipes_and_spq_budget_helpers():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    import torch

    args = SimpleNamespace(
        spq_lora_steps=0,
        spq_s_method="wanda",
        spq_r_method="svd",
        q_method="rtn",
        s_method="wanda",
        r_method="whitened_svd",
        svd_device="cpu",
    )
    specs = runner.fair_benchmark_extended_recipe_specs(args)
    names = {spec["strategy"] for spec in specs}
    assert {"slim_like_srq_proxy", "spq_like_rsq_no_lora", "hessian_guided_spq_no_lora"} <= names
    assert "spq_like_rsq_lora" not in names

    args.spq_lora_steps = 5
    names_with_lora = {spec["strategy"] for spec in runner.fair_benchmark_extended_recipe_specs(args)}
    assert {"spq_like_rsq_lora", "hessian_guided_spq_lora"} <= names_with_lora

    weights = {
        "gpt_neox.layers.0.attention.dense": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h": torch.arange(16, dtype=torch.float32).reshape(4, 4),
    }
    covariances = {name: torch.eye(4) for name in weights}
    spq_memory = runner.weighted_layerwise_nominal_memory_ratio(
        weights,
        {name: "".join(runner.spq_ops_for_layer(name)) for name in weights},
        bits=4,
        keep_fraction=0.8,
        rank_fraction=0.5,
    )
    assert math.isclose(spq_memory, 0.225)

    replacements, selection = runner.choose_hessian_guided_spq_budget(
        weights,
        covariances,
        q_methods=["rtn"],
        s_methods=["magnitude", "wanda"],
        r_methods=["svd", "whitened_svd"],
        args=args,
        bits=4,
        keep_fraction=0.8,
        rank_fraction=0.5,
    )
    assert set(replacements) == set(weights)
    assert len(selection) == 2
    assert all(row["selected_nominal_bits"] == 4 for row in selection)
    assert all(row["selected_nominal_keep_fraction"] == 0.8 for row in selection)
    assert all(row["selected_nominal_rank_fraction"] == 0.5 for row in selection)


def test_orthofilter_spq_residual_candidates_report_filter_and_memory():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    import torch

    args = SimpleNamespace(
        q_method="rtn",
        s_method="wanda",
        r_method="whitened_svd",
        svd_device="cpu",
        orthofilter_rho_threshold=0.25,
        orthofilter_hessian_weight=0.25,
        orthofilter_activation_weight=1.0,
        orthofilter_worst_token_weight=0.25,
        orthofilter_conflict_weight=0.5,
        orthofilter_memory_weight=0.05,
        orthofilter_zero_shot_proxy_weight=0.2,
        text_source_used="zero_shot_backup:unit_test",
    )
    weights = {
        "gpt_neox.layers.0.attention.dense": torch.randn(4, 4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h": torch.randn(4, 4),
    }
    covariances = {name: torch.eye(4) for name in weights}
    samples = {name: torch.eye(4) for name in weights}

    replacements, rows = runner.choose_orthofilter_spq_budget(
        weights,
        covariances,
        samples,
        q_methods=["rtn"],
        s_methods=["wanda"],
        r_methods=["svd"],
        args=args,
        bits=4,
        keep_fraction=0.5,
        rank_fraction=0.5,
        include_residual=True,
    )
    assert set(replacements) == set(weights)
    assert any(row["candidate_kind"] == "residual_low_rank_after_q" for row in rows)
    assert any(row["candidate_kind"] == "residual_sparse_after_q" for row in rows)
    assert sum(1 for row in rows if row["selected"]) == len(weights)
    assert all("selector_score" in row for row in rows)
    residual_rows = [row for row in rows if row["uses_residual_decomposition"]]
    assert residual_rows
    memory_by_order = {row["candidate_order"]: float(row["candidate_memory_ratio"]) for row in residual_rows}
    assert math.isclose(memory_by_order["q+r_res"], 1.25)
    assert math.isclose(memory_by_order["q+s_res"], 0.75)
    assert all(float(row["zero_shot_proxy_risk"]) >= 0.0 for row in rows)
    assert any(row["zero_shot_proxy_source"] == "choice_text_token_risk_p95" for row in rows)


def test_orthofilter_selection_respects_filter_before_score():
    runner = _load_pretrained_runner()
    if runner is None:
        return

    base = {
        "candidate_memory_ratio": 0.5,
        "candidate_q_method": "rtn",
        "candidate_s_method": "wanda",
        "candidate_r_method": "svd",
    }
    rejected_low_score = {
        **base,
        "candidate_order": "bad",
        "filter_pass": False,
        "selector_score": 0.0,
        "positive_conditional_rho": 1.0,
    }
    feasible_higher_score = {
        **base,
        "candidate_order": "good",
        "filter_pass": True,
        "selector_score": 10.0,
        "positive_conditional_rho": 0.0,
    }
    best, fallback = runner.select_orthofilter_candidate([rejected_low_score, feasible_higher_score])
    assert best["candidate_order"] == "good"
    assert fallback is False

    best, fallback = runner.select_orthofilter_candidate([{**rejected_low_score, "selector_score": 2.0}, {**rejected_low_score, "candidate_order": "less_bad", "selector_score": 1.0}])
    assert best["candidate_order"] == "less_bad"
    assert fallback is True


def test_orthofilter_zero_shot_proxy_requires_samples():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    import torch

    args = SimpleNamespace(
        orthofilter_rho_threshold=0.25,
        orthofilter_hessian_weight=0.25,
        orthofilter_activation_weight=1.0,
        orthofilter_worst_token_weight=0.25,
        orthofilter_conflict_weight=0.5,
        orthofilter_memory_weight=0.05,
        orthofilter_zero_shot_proxy_weight=0.2,
        text_source_used="zero_shot_backup:unit_test",
    )
    weight = torch.eye(2)
    final = weight * 0.5
    row = runner.make_orthofilter_candidate(
        name="gpt_neox.layers.0.attention.dense",
        weight=weight,
        cov=torch.eye(2),
        samples=torch.empty(0, 2),
        order_name="rq",
        final=final,
        first_weight=weight * 0.75,
        methods={"q": "rtn", "s": "wanda", "r": "svd"},
        args=args,
        bits=4,
        keep_fraction=0.8,
        rank_fraction=0.5,
        candidate_kind="sequential",
    )
    assert row["zero_shot_proxy_risk"] == 0.0
    assert row["zero_shot_proxy_source"] == "not_used_without_zero_shot_backup_text_or_samples"


def test_disjoint_text_windows_are_separate():
    runner = _load_pretrained_runner()
    if runner is None:
        return
    args = SimpleNamespace(
        disjoint_text_splits=True,
        calib_limit=1,
        eval_limit=1,
        texts_per_batch_window=2,
        spq_lora_train_limit=1,
        spq_lora_steps=1,
    )
    runner.split_text_windows(args, [f"text-{idx}" for idx in range(6)])
    assert args.text_split_policy == "disjoint_sequential_text_windows"
    assert args.calib_texts == ["text-0", "text-1"]
    assert args.eval_texts == ["text-2", "text-3"]
    assert args.recovery_texts == ["text-4", "text-5"]


if __name__ == "__main__":
    for _name, _func in sorted(globals().items()):
        if _name.startswith("test_"):
            _func()
            print(_name, "ok")
