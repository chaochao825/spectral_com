import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("residual_stack_validate tests require torch") from exc
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_pretrained_llm_orthogonality.py"
    spec = importlib.util.spec_from_file_location("run_pretrained_llm_orthogonality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**kwargs):
    base = {
        "bits": 4,
        "svd_device": "cpu",
        "residual_stack_rho_threshold": 0.3,
        "residual_stack_activation_weight": 1.0,
        "residual_stack_worst_token_weight": 0.5,
        "residual_stack_hessian_weight": 0.2,
        "residual_stack_include_order_gap": False,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_residual_sparse_project_keeps_exact_topk_and_allows_zero_rank():
    runner = _load_runner()
    import torch

    weight = torch.tensor([[1.0, -3.0, 2.0, 0.5]], dtype=torch.float32)
    cov = torch.eye(4)
    zero = runner.residual_sparse_project(weight, cov, 0.0, "magnitude")
    assert torch.count_nonzero(zero).item() == 0

    projected = runner.residual_sparse_project(weight, cov, 0.5, "magnitude")
    assert torch.count_nonzero(projected).item() == 2
    assert torch.allclose(projected, torch.tensor([[0.0, -3.0, 2.0, 0.0]]))


def test_lowrank_project_rank_zero_and_memory_accounting():
    runner = _load_runner()
    import torch

    weight = torch.randn(4, 6)
    cov = torch.eye(6)
    assert torch.count_nonzero(runner.lowrank_project_rank(weight, cov, 0, "svd", svd_device="cpu")).item() == 0
    assert math.isclose(runner.lowrank_memory_ratio_for_rank(weight, 2), 2 * (4 + 6) / (4 * 6))
    assert runner.rank_from_residual_memory_ratio(weight, 2 * (4 + 6) / (4 * 6)) == 2


def test_residual_stack_candidate_memory_never_exceeds_target_after_discrete_rank():
    runner = _load_runner()
    import torch

    weight = torch.randn(8, 8)
    cov = torch.eye(8)
    args = _args()
    candidates = runner.residual_stack_candidates_for_layer(
        "model.layers.0.mlp.down_proj",
        weight,
        cov,
        None,
        args,
        targets=[0.30],
        splits=[0.5],
        q_methods=["rtn"],
        s_methods=["magnitude"],
        r_methods=["svd"],
    )
    feasible = [row for row in candidates if row["budget_feasible"]]
    assert feasible
    assert all(float(row["candidate_memory_ratio"]) <= 0.30 + 1e-9 for row in feasible)
    assert {"q_only", "q_l", "q_s", "q_s_l"} <= {row["candidate_kind"] for row in candidates}


def test_residual_stack_mode_parser_accepts_new_mode():
    runner = _load_runner()
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--mode", "residual_stack_validate", "--residual-stack-memory-targets", "0.258"])
    assert args.mode == "residual_stack_validate"
    assert args.residual_stack_memory_targets == "0.258"


def test_residual_stack_greedy_does_not_downgrade_after_better_upgrade():
    runner = _load_runner()
    import torch

    final = torch.zeros(1, 1)

    def row(kind, memory, score):
        return {
            "layer": "layer0",
            "candidate_kind": kind,
            "target_memory_ratio": 0.258,
            "candidate_memory_ratio": memory,
            "selector_score": score,
            "filter_pass": True,
            "budget_feasible": True,
            "layer_parameter_count": 100,
            "q_method": "rtn",
            "s_method": "magnitude",
            "r_method": "svd",
            "_final": final,
        }

    rows = [
        row("q_only", 0.250, 10.0),
        row("q_s_l", 0.251, 5.0),
        row("q_s", 0.257, 8.0),
    ]
    _replacements, selected, summary = runner.select_residual_stack_greedy(rows, 0.258)
    assert summary["global_budget_feasible"] is True
    assert len(selected) == 1
    assert selected[0]["candidate_kind"] == "q_s_l"
    assert math.isclose(selected[0]["selector_score"], 5.0)


def test_dam_factor_quant_memory_and_shapes():
    runner = _load_runner()
    import torch

    weight = torch.randn(8, 6)
    cov = torch.eye(6)
    target = 0.30
    rank = runner.factor_quant_rank_for_memory(weight, target, bits=4)
    assert rank >= 0
    assert runner.factor_quant_memory_ratio_for_rank(weight, rank, bits=4) <= target + 1e-9

    plain, plain_info = runner.factor_quantized_svd_weight(
        weight,
        cov,
        None,
        rank=rank,
        bits=4,
        scale_mode="plain_lq",
        svd_device="cpu",
        alpha_grid=[0.0, 0.5],
    )
    dam, dam_info = runner.factor_quantized_svd_weight(
        weight,
        cov,
        None,
        rank=rank,
        bits=4,
        scale_mode="dam_activation_grid",
        svd_device="cpu",
        alpha_grid=[0.0, 0.5],
    )
    assert plain.shape == weight.shape
    assert dam.shape == weight.shape
    assert plain_info["factor_rank"] == rank
    assert dam_info["dam_alpha"] in {0.0, 0.5}
