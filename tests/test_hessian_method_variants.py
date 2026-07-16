from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_pretrained_hessian_repair_method_variant_tests",
    SCRIPTS / "run_pretrained_hessian_repair.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_covariance_modes_and_factorizer_floor_are_disclosed() -> None:
    covariance = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    full, full_prepared, full_report = RUNNER.prepare_metric_covariance(
        covariance, mode="full"
    )
    diagonal, diagonal_prepared, diagonal_report = RUNNER.prepare_metric_covariance(
        covariance, mode="diagonal", damping_ratio=0.1
    )
    identity, identity_prepared, identity_report = RUNNER.prepare_metric_covariance(
        covariance, mode="identity"
    )
    assert full_report["covariance_mode"] == "full"
    assert diagonal_report["configured_damping"] == pytest.approx(0.3)
    assert torch.count_nonzero(diagonal - torch.diag(torch.diag(diagonal))) == 0
    torch.testing.assert_close(identity, torch.eye(2) * 3.0)
    assert identity_report["collected_min_eigenvalue"] > 0.0
    assert full_prepared.matrix.flags.writeable is False
    assert diagonal_prepared.matrix.flags.writeable is False
    assert identity_prepared.matrix.flags.writeable is False

    factorizer = RUNNER.LowRankFactorizer(
        full,
        method="whitened_svd",
        device="cpu",
        whitening_floor_ratio=1e-3,
    )
    assert factorizer.diagnostics["whitening_floor_ratio"] == pytest.approx(1e-3)
    assert factorizer.diagnostics["fit_covariance_dimension"] == 2

    indefinite = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="materially indefinite"):
        RUNNER.prepare_metric_covariance(indefinite, mode="diagonal")

    randomized = RUNNER.LowRankFactorizer(
        torch.eye(8),
        method="svd",
        device="cpu",
        svd_solver="randomized",
        randomized_oversampling=2,
        randomized_niter=1,
    )
    codec = randomized.factorize(
        np.arange(128, dtype=np.float32).reshape(16, 8) / 128.0,
        rank=2,
    )
    codec_repeat = randomized.factorize(
        np.arange(128, dtype=np.float32).reshape(16, 8) / 128.0,
        rank=2,
    )
    assert codec is not None and codec.left.shape == (16, 2) and codec.right.shape == (2, 8)
    assert codec_repeat is not None
    np.testing.assert_array_equal(codec.left, codec_repeat.left)
    np.testing.assert_array_equal(codec.right, codec_repeat.right)
    assert randomized.diagnostics["last_resolved_svd_solver"] == "torch.svd_lowrank"
    assert randomized.diagnostics["svd_solver_call_counts"] == {"torch.svd_lowrank": 2}


def _candidate(
    strategy: str,
    weight: np.ndarray,
    q: object,
    *,
    layer: str = "model.layers.0.mlp.fc1",
    sparse: object | None = None,
    lowrank: object | None = None,
) -> object:
    return RUNNER.Candidate(
        strategy,
        0.5,
        layer,
        weight,
        q,
        sparse,
        lowrank,
    )


def test_global_allocator_is_nested_with_q_l_and_exactly_file_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.array([[1, -1], [0, 1]], dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[0.25], [0.125]], dtype=np.float16),
        right=np.array([[1.0, -1.0]], dtype=np.float16),
    )
    sparse_mask = np.array([[False, True], [False, False]])
    sparse = RUNNER.SparseCodec(
        values=np.where(sparse_mask, np.float16(0.125), np.float16(0.0)),
        mask=sparse_mask,
    )
    q_only = _candidate("Q", weight, q)
    ql = _candidate("Q+L", weight, q, lowrank=lowrank)
    qsl = _candidate("Q+S+L_QL_budget", weight, q, sparse=sparse, lowrank=lowrank)
    fallback = _candidate("Q+S+L_QL_budget", weight, q, lowrank=lowrank)
    metric = RUNNER.HessianMetric(torch.eye(2), device="cpu")

    serializer_calls = 0
    full_serializer = RUNNER.codec_artifact_natural_file_bytes

    def counted_serializer(*args: object, **kwargs: object) -> int:
        nonlocal serializer_calls
        serializer_calls += 1
        return full_serializer(*args, **kwargs)

    monkeypatch.setattr(RUNNER, "codec_artifact_natural_file_bytes", counted_serializer)
    selected, report = RUNNER.select_global_exact_qsl_allocation(
        ql_candidates={ql.layer: ql},
        degenerate_candidates={ql.layer: [q_only, ql]},
        option_pools={ql.layer: [qsl]},
        fallback_candidates={ql.layer: fallback},
        metrics={ql.layer: metric},
        alignment=64,
    )
    chosen = selected[ql.layer]
    assert serializer_calls == 2
    chosen_bytes = full_serializer(
        [RUNNER._artifact_layer(chosen)], alignment=64
    )
    ql_bytes = full_serializer(
        [RUNNER._artifact_layer(ql)], alignment=64
    )
    assert chosen.strategy == "Q+S+L_QL_budget"
    assert chosen_bytes <= ql_bytes
    assert metric.cost(chosen.final - weight) <= metric.cost(ql.final - weight) + 1e-12
    assert report["strict_file_byte_feasible"] is True
    assert report["q_l_cap_natural_file_bytes"] == ql_bytes


def test_nonuniform_rank_grid_reaches_true_feasible_endpoint() -> None:
    assert RUNNER.parse_allocation_rank_grid("") == []
    assert RUNNER.parse_allocation_rank_grid("8,2,8,0") == [0, 2, 8]
    assert RUNNER.resolve_allocation_rank_grid(
        24, max_dense_rank=4, configured_grid=[]
    ) == [0, 1, 2, 3, 4]
    assert RUNNER.resolve_allocation_rank_grid(
        24, max_dense_rank=4, configured_grid=[1, 2, 4, 8, 16, 99]
    ) == [0, 1, 2, 4, 8, 16, 24]
    assert RUNNER.resolve_allocation_rank_grid(
        3, max_dense_rank=4, configured_grid=[0, 2, 4, 8]
    ) == [0, 2, 3]
    with pytest.raises(ValueError, match="non-negative"):
        RUNNER.resolve_allocation_rank_grid(
            8, max_dense_rank=4, configured_grid=[-1, 2]
        )


def test_global_allocator_can_borrow_enumerated_bytes_across_layers() -> None:
    layer_a = "model.layers.0.mlp.fc1"
    layer_b = "model.layers.1.mlp.fc1"
    q_a = RUNNER.QuantCodec(
        codes=np.zeros((8, 8), dtype=np.int8),
        scales=np.ones(8, dtype=np.float16),
        bits=4,
    )
    q_b = RUNNER.QuantCodec(
        codes=np.zeros((32, 32), dtype=np.int8),
        scales=np.ones(32, dtype=np.float16),
        bits=4,
    )
    lowrank_a = RUNNER.LowRankCodec(
        left=np.zeros((8, 1), dtype=np.float16),
        right=np.zeros((1, 8), dtype=np.float16),
    )
    lowrank_b = RUNNER.LowRankCodec(
        left=np.zeros((32, 8), dtype=np.float16),
        right=np.zeros((8, 32), dtype=np.float16),
    )
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_a[0, :4] = True
    values_a = np.zeros((8, 8), dtype=np.float16)
    values_a[0, :4] = np.float16(0.25)
    sparse_a = RUNNER.SparseCodec(values=values_a, mask=mask_a)
    weight_a = sparse_a.decode()
    weight_b = np.zeros((32, 32), dtype=np.float32)

    q_only_a = _candidate("Q", weight_a, q_a, layer=layer_a)
    ql_a = _candidate("Q+L", weight_a, q_a, layer=layer_a, lowrank=lowrank_a)
    borrowed_a = _candidate(
        "Q+S+L_QL_budget",
        weight_a,
        q_a,
        layer=layer_a,
        sparse=sparse_a,
        lowrank=lowrank_a,
    )
    q_only_b = _candidate("Q", weight_b, q_b, layer=layer_b)
    ql_b = _candidate("Q+L", weight_b, q_b, layer=layer_b, lowrank=lowrank_b)
    strict_q_b = _candidate("Q+S+L_QL_budget", weight_b, q_b, layer=layer_b)
    metrics = {
        layer_a: RUNNER.HessianMetric(torch.eye(8), device="cpu"),
        layer_b: RUNNER.HessianMetric(torch.eye(32), device="cpu"),
    }

    selected, report = RUNNER.select_global_exact_qsl_allocation(
        ql_candidates={layer_a: ql_a, layer_b: ql_b},
        degenerate_candidates={
            layer_a: [q_only_a, ql_a],
            layer_b: [q_only_b, ql_b],
        },
        option_pools={layer_a: [borrowed_a], layer_b: [strict_q_b]},
        fallback_candidates={layer_a: ql_a, layer_b: ql_b},
        metrics=metrics,
        alignment=64,
    )
    local_borrowed_bytes = RUNNER.codec_artifact_natural_file_bytes(
        [RUNNER._artifact_layer(borrowed_a)], alignment=64
    )
    local_ql_bytes = RUNNER.codec_artifact_natural_file_bytes(
        [RUNNER._artifact_layer(ql_a)], alignment=64
    )
    assert local_borrowed_bytes > local_ql_bytes
    assert selected[layer_a].sparse_nnz == 4
    assert selected[layer_b].rank == 0 and selected[layer_b].sparse_nnz == 0
    assert report["strict_file_byte_feasible"] is True
    assert report["selected_qsl_natural_file_bytes"] <= report["q_l_cap_natural_file_bytes"]


def test_global_allocator_does_not_prune_on_nonadditive_one_layer_bytes() -> None:
    layer_a, layer_b = "a", "b"
    shape_a, shape_b = (3, 7), (4, 17)

    def quantizer(shape: tuple[int, int]) -> object:
        return RUNNER.QuantCodec(
            codes=np.zeros(shape, dtype=np.int8),
            scales=np.ones(shape[0], dtype=np.float16),
            bits=2,
        )

    def factor(shape: tuple[int, int], rank: int, marker: float) -> object | None:
        if rank == 0:
            return None
        left = np.zeros((shape[0], rank), dtype=np.float16)
        right = np.zeros((rank, shape[1]), dtype=np.float16)
        left[0, 0] = np.float16(marker)
        right[0, 0] = np.float16(1.0)
        return RUNNER.LowRankCodec(left=left, right=right)

    class MarkerMetric:
        def __init__(self, zero_cost: float, one_cost: float) -> None:
            self.zero_cost = zero_cost
            self.one_cost = one_cost

        def cost(self, delta: np.ndarray) -> float:
            return self.one_cost if float(delta[0, 0]) == 1.0 else self.zero_cost

    weight_a = np.zeros(shape_a, dtype=np.float32)
    weight_b = np.zeros(shape_b, dtype=np.float32)
    q_a, q_b = quantizer(shape_a), quantizer(shape_b)
    a0 = _candidate("Q+S+L_QL_budget", weight_a, q_a, layer=layer_a)
    a1 = _candidate(
        "Q+L", weight_a, q_a, layer=layer_a, lowrank=factor(shape_a, 1, 0.0)
    )
    a3 = _candidate(
        "Q+S+L_QL_budget",
        weight_a,
        q_a,
        layer=layer_a,
        lowrank=factor(shape_a, 3, 1.0),
    )
    b0 = _candidate("Q+S+L_QL_budget", weight_b, q_b, layer=layer_b)
    b1 = _candidate(
        "Q+S+L_QL_budget",
        weight_b,
        q_b,
        layer=layer_b,
        lowrank=factor(shape_b, 1, 1.0),
    )

    selected, report = RUNNER.select_global_exact_qsl_allocation(
        ql_candidates={layer_a: a1, layer_b: b0},
        degenerate_candidates={layer_a: [], layer_b: []},
        option_pools={layer_a: [a0, a3], layer_b: [b0, b1]},
        fallback_candidates={layer_a: a0, layer_b: b0},
        metrics={
            layer_a: MarkerMetric(10.0, 0.0),
            layer_b: MarkerMetric(20.0, 0.0),
        },
        alignment=64,
    )
    assert selected[layer_a].rank == 3
    assert selected[layer_b].rank == 0
    assert report["selected_hessian_cost"] == pytest.approx(20.0)
    assert report["selected_qsl_natural_file_bytes"] == 2632
    assert report["q_l_cap_natural_file_bytes"] == 2632
    assert report["frontier_coarsening_events"] == 0


@pytest.mark.parametrize("strategy", RUNNER.GLOBAL_CONTROL_STRATEGIES)
def test_global_control_allocator_uses_the_q_l_cap_without_fallback(
    strategy: str,
) -> None:
    layer = "model.layers.0.mlp.fc1"
    weight = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    mask = np.array([[True, False], [False, False]])
    sparse = RUNNER.SparseCodec(
        values=np.where(mask, np.float16(1.0), np.float16(0.0)),
        mask=mask,
    )
    ql = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    q_control = _candidate(strategy, weight, q, layer=layer)
    repaired = _candidate(
        strategy,
        weight,
        q,
        layer=layer,
        sparse=sparse if strategy == "Q+S_OBS_global" else None,
        lowrank=(
            lowrank
            if strategy
            in {"Q+L_global", RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY}
            else None
        ),
    )
    metric = RUNNER.HessianMetric(torch.eye(2), device="cpu")

    selected, report = RUNNER.select_global_exact_component_allocation(
        strategy=strategy,
        ql_candidates={layer: ql},
        option_pools={layer: [q_control, repaired]},
        metrics={layer: metric},
        alignment=64,
        optimality_scope="unit_enumerated_pool",
        candidate_pool_asymmetry="unit_disclosure",
    )

    assert selected[layer].strategy == strategy
    assert report["selection_source"] == "global_exact_canonical_layout_pareto_frontier"
    assert report["fallback_policy"] == "forbidden_fail_closed"
    assert report["candidate_pool_asymmetry"] == "unit_disclosure"
    assert report["selected_natural_file_bytes"] <= report["q_l_cap_natural_file_bytes"]
    assert report["full_serializer_cross_checks"] == 2

    repaired.diagnostics["allocation_fallback"] = "forbidden"
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        RUNNER.select_global_exact_component_allocation(
            strategy=strategy,
            ql_candidates={layer: ql},
            option_pools={layer: [repaired]},
            metrics={layer: metric},
            alignment=64,
            optimality_scope="unit_enumerated_pool",
            candidate_pool_asymmetry="unit_disclosure",
        )


def test_global_nojoint_control_rejects_same_layer_sparse_lowrank_state() -> None:
    layer = "model.layers.0.mlp.fc1"
    weight = np.eye(2, dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    mask = np.array([[True, False], [False, False]])
    joint = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
        sparse=RUNNER.SparseCodec(
            values=np.where(mask, np.float16(1.0), np.float16(0.0)),
            mask=mask,
        ),
        lowrank=lowrank,
    )
    ql = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    with pytest.raises(AssertionError, match="forbidden joint S\\+L"):
        RUNNER.select_global_exact_component_allocation(
            strategy=RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
            ql_candidates={layer: ql},
            option_pools={layer: [joint]},
            metrics={layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
            alignment=64,
            optimality_scope="unit_enumerated_pool",
            candidate_pool_asymmetry="unit_disclosure",
        )


def test_global_allocator_exposes_exact_proxy_top_k() -> None:
    layer = "model.layers.0.mlp.fc1"
    weight = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    q_only = _candidate("Q", weight, q, layer=layer)
    ql = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    ranked, report = RUNNER.rank_global_exact_qsl_allocations(
        ql_candidates={layer: ql},
        degenerate_candidates={layer: [q_only, ql]},
        option_pools={layer: []},
        fallback_candidates={layer: ql},
        metrics={layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
        alignment=64,
        top_k=2,
    )

    assert len(ranked) == 2
    assert ranked[0].hessian_cost <= ranked[1].hessian_cost
    assert ranked[0].allocation_digest != ranked[1].allocation_digest
    assert report["proxy_top_k_requested"] == 2
    assert report["proxy_top_k_returned"] == 2
    assert report["full_serializer_cross_checks"] == 3


def test_validation_nll_rerank_can_select_non_proxy_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = "model.layers.0.mlp.fc1"
    module = torch.nn.Linear(2, 2, bias=False)
    baseline = {layer: module.weight.detach().clone()}
    weight = baseline[layer].float().numpy()
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    first = _candidate("Q+S+L_QL_budget", weight, q, layer=layer)
    second = _candidate("Q+S+L_QL_budget", weight, q, layer=layer)
    ranked = [
        RUNNER.RankedGlobalAllocation(
            candidates={layer: first},
            natural_file_bytes=100,
            hessian_cost=1.0,
            choices=(0,),
            allocation_digest="first",
        ),
        RUNNER.RankedGlobalAllocation(
            candidates={layer: second},
            natural_file_bytes=100,
            hessian_cost=2.0,
            choices=(1,),
            allocation_digest="second",
        ),
    ]

    def fake_evaluate(
        _model: object,
        _tokenizer: object,
        *,
        strategy: str,
        **_kwargs: object,
    ) -> tuple[dict[str, float | int], list[dict[str, object]]]:
        nll = 0.5
        if strategy.endswith("_1"):
            nll = 1.5
        elif strategy.endswith("_2"):
            nll = 1.0
        return (
            {"nll": nll, "perplexity": float(np.exp(nll)), "tokens": 1},
            [
                {
                    "strategy": strategy,
                    "window_index": 0,
                    "batch_index": 0,
                    "sequence_index": 0,
                    "tokens": 1,
                    "nll_sum": nll,
                    "nll": nll,
                    "perplexity": float(np.exp(nll)),
                }
            ],
        )

    monkeypatch.setattr(RUNNER, "evaluate_current_model_with_windows", fake_evaluate)
    monkeypatch.setattr(RUNNER.base, "restore_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(RUNNER.base, "apply_replacements", lambda *_args, **_kwargs: None)
    args = SimpleNamespace(
        two_stage_selection=True,
        selection_texts=["validation"],
        sequence_length=2,
        batch_size=1,
        device="cpu",
        selection_limit=1,
    )
    selected, rows, _windows, baseline_metrics, reports = (
        RUNNER.rerank_global_allocations_by_validation(
            model=object(),
            tokenizer=object(),
            modules={layer: module},
            baseline_weights=baseline,
            ranked_by_strategy={"Q+S+L_QL_budget": ranked},
            args=args,
        )
    )

    assert selected["Q+S+L_QL_budget"] is ranked[1].candidates
    assert reports["Q+S+L_QL_budget"]["validation_selected_proxy_rank"] == 2
    assert baseline_metrics["nll"] == pytest.approx(0.5)
    assert [row["selected_by_validation"] for row in rows if row["proxy_rank"] > 0] == [
        False,
        True,
    ]


def test_two_stage_text_roles_are_split_and_content_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools = {
        "train": ["shared", "train-a", "train-b", "train-c"],
        "validation": ["shared", "train-a", "validation-a", "validation-b", "validation-c"],
        "test": ["shared", "validation-a", "test-a", "test-b", "test-c"],
    }

    def fake_load(role_args: object, *, limit: int) -> tuple[list[str], str, list[dict[str, object]]]:
        split = role_args.data_cfg["split"]
        return pools[split][:limit], "dataset:wikitext", [{"split": split}]

    monkeypatch.setattr(RUNNER.base, "load_eval_texts", fake_load)
    args = SimpleNamespace(
        data_cfg={"dataset": "wikitext", "split": "validation"},
        texts_per_batch_window=1,
        calib_limit=2,
        selection_limit=2,
        eval_limit=2,
        skip_comfort=True,
        calibration_split="train",
        selection_split="validation",
        test_split="test",
    )
    RUNNER.load_two_stage_text_windows(args)

    assert set(args.calib_texts).isdisjoint(args.selection_texts)
    assert set(args.calib_texts).isdisjoint(args.eval_texts)
    assert set(args.selection_texts).isdisjoint(args.eval_texts)
    assert args.data_role_splits == {
        "calibration": "train",
        "selection": "validation",
        "test": "test",
    }


def test_heterogeneous_quantizer_grid_covers_bits_groups_and_rules() -> None:
    weight = np.array(
        [[-1.0, -0.25, 0.25, 1.0], [0.0, 0.1, 0.2, 2.0]],
        dtype=np.float32,
    )
    codecs = RUNNER.build_quantizer_candidate_codecs(
        weight,
        bit_widths=[3, 4],
        group_sizes=[0, 2],
        quantizers=["symmetric_rtn", "symmetric_mse_clip"],
    )
    assert len(codecs) == 8
    keys = {
        (
            codec.bits,
            0 if codec.col_block_size is None else codec.col_block_size,
            codec.quantizer,
        )
        for codec in codecs
    }
    assert len(keys) == 8
    grouped = next(
        codec
        for codec in codecs
        if codec.bits == 4
        and codec.col_block_size == 2
        and codec.quantizer == "symmetric_rtn"
    )
    assert grouped.scales.shape == (2, 2)
    rtn = next(
        codec
        for codec in codecs
        if codec.bits == 4
        and codec.col_block_size is None
        and codec.quantizer == "symmetric_rtn"
    )
    mse = next(
        codec
        for codec in codecs
        if codec.bits == 4
        and codec.col_block_size is None
        and codec.quantizer == "symmetric_mse_clip"
    )
    assert np.mean((mse.decode() - weight) ** 2) <= np.mean(
        (rtn.decode() - weight) ** 2
    )


def test_quantized_lowrank_factors_reduce_logical_payload() -> None:
    source = RUNNER.LowRankCodec(
        left=np.array([[0.5, -0.25], [0.125, 0.75]], dtype=np.float16),
        right=np.array(
            [[0.5, -0.5, 0.25], [0.125, 0.25, -0.75]],
            dtype=np.float16,
        ),
    )
    quantized = RUNNER.quantize_lowrank_factors(source, bits=4)
    assert quantized is not None
    assert quantized.factor_bits == 4
    assert quantized.left_scales is not None
    assert quantized.right_scales is not None
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 3), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    weight = np.zeros((2, 3), dtype=np.float32)
    fp16_candidate = _candidate("Q+L", weight, q, lowrank=source)
    quantized_candidate = _candidate("Q+L", weight, q, lowrank=quantized)
    assert (
        quantized_candidate.payload(support_encoding="csr_fixed").total_bits
        < fp16_candidate.payload(support_encoding="csr_fixed").total_bits
    )


def test_obs_joint_endpoint_keeps_selected_lowrank_factor_width() -> None:
    covariance_tensor = torch.eye(8, dtype=torch.float32)
    covariance, prepared, _report = RUNNER.prepare_metric_covariance(
        covariance_tensor,
        mode="full",
    )
    factorizer = RUNNER.LowRankFactorizer(
        covariance,
        method="svd",
        device="cpu",
        svd_solver="full",
    )
    weight_tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64.0
    q = RUNNER.QuantCodec(
        codes=np.zeros((8, 8), dtype=np.int8),
        scales=np.ones(8, dtype=np.float16),
        bits=4,
    )
    global_q = RUNNER.Candidate(
        "Q_global_scale",
        1.0,
        "model.layers.0.mlp.gate_proj",
        weight_tensor.numpy(),
        q,
    )
    args = SimpleNamespace(
        support_encoding="csr_fixed",
        skip_block_scale=True,
        s_method="magnitude",
        obs_rcond=1e-10,
        max_allocation_ranks=1,
        allocation_rank_grid=[0, 1],
        residual_order="s_then_l",
        enforce_serialized_rate_cap=False,
        artifact_alignment=64,
        rate_allocation="per_layer",
        strict_sparse_refit="obs",
        global_frontier_top_ranks=0,
        global_frontier_support_fractions=[],
        global_frontier_budget_multipliers=[],
        include_global_single_component_controls=False,
        endpoint_target=1.0,
        scale_bounds=(0.0, 2.0),
        rho_threshold=0.1,
    )

    selected, _rows, _global_pools = RUNNER.build_layer_candidates(
        layer="model.layers.0.mlp.gate_proj",
        weight_tensor=weight_tensor,
        covariance_tensor=covariance_tensor,
        prepared_covariance=prepared,
        activation_samples=None,
        factorizer=factorizer,
        metric=RUNNER.HessianMetric(covariance, device="cpu"),
        target_ratio=1.0,
        q=q,
        global_q=global_q,
        block_q_options=[],
        lowrank_factor_bits=4,
        args=args,
    )

    obs_joint = selected["Q+S_OBS+L"]
    assert obs_joint.lowrank is not None
    assert obs_joint.lowrank.factor_bits == 4


def test_validation_rerank_preserves_canonical_allocator_selection_source() -> None:
    report = {
        "selection_source": "global_exact_canonical_layout_pareto_frontier",
        "selected_hessian_cost": 1.0,
        "selected_natural_file_bytes": 900,
        "unused_natural_bytes_before_tail_padding": 100,
        "q_l_cap_natural_file_bytes": 1000,
        "endpoint_label": "Q+S+L_QL_budget",
    }
    allocation = RUNNER.RankedGlobalAllocation(
        candidates={},
        natural_file_bytes=920,
        hessian_cost=1.1,
        choices=(),
        allocation_digest="validation-winner",
    )
    selection_report = {
        "selection_source": "validation_nll_rerank_of_exact_proxy_top_k",
        "validation_selected_proxy_rank": 2,
    }

    RUNNER._apply_validation_selection_to_allocator_report(
        report,
        allocation=allocation,
        selection_report=selection_report,
    )

    assert (
        report["selection_source"]
        == "global_exact_canonical_layout_pareto_frontier"
    )
    assert (
        report["final_selection_source"]
        == "validation_nll_rerank_of_exact_proxy_top_k"
    )
    assert report["selected_natural_file_bytes"] == 920
    assert report["selected_qsl_natural_file_bytes"] == 920


def test_proxy_top_k_retains_distinct_values_with_the_same_layout() -> None:
    layer = "same.layout"
    weight = np.ones((2, 2), dtype=np.float32)
    q_a = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
        quantizer="symmetric_rtn",
    )
    q_b = RUNNER.QuantCodec(
        codes=np.ones((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
        quantizer="symmetric_mse_clip",
    )
    candidate_a = _candidate("Q+S+L_QL_budget", weight, q_a, layer=layer)
    candidate_b = _candidate("Q+S+L_QL_budget", weight, q_b, layer=layer)
    cap = _candidate("Q+L", weight, q_a, layer=layer)
    ranked, report = RUNNER._rank_global_exact_pareto_allocations(
        cap_candidates={layer: cap},
        option_pools={layer: [candidate_a, candidate_b]},
        metrics={layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
        alignment=64,
        endpoint_label="unit",
        optimality_scope="unit",
        top_k=2,
    )
    assert len(ranked) == 2
    assert report["proxy_top_k_returned"] == 2
    assert {item.candidates[layer].q.quantizer for item in ranked} == {
        "symmetric_rtn",
        "symmetric_mse_clip",
    }


def test_global_allocator_can_require_an_exact_natural_file_size() -> None:
    layer = "exact.match"
    weight = np.eye(2, dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    cap = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    small = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
    )
    exact = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
        lowrank=lowrank,
    )
    required = RUNNER.codec_artifact_allocations_natural_file_bytes(
        [RUNNER._artifact_allocation(exact)],
        alignment=64,
    )
    ranked, report = RUNNER.rank_global_exact_component_allocations(
        strategy=RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        ql_candidates={layer: cap},
        option_pools={layer: [small, exact]},
        metrics={layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
        alignment=64,
        optimality_scope="unit",
        candidate_pool_asymmetry="unit",
        top_k=2,
        required_natural_file_bytes=required,
    )

    assert ranked
    assert all(item.natural_file_bytes == required for item in ranked)
    assert ranked[0].candidates[layer].rank == 1
    assert report["exact_natural_file_byte_match_required"] is True
    assert report["exact_natural_file_byte_match_available"] is True
    assert report["exact_natural_file_byte_match"] is True


def test_global_allocator_reports_unavailable_exact_natural_match() -> None:
    layer = "exact.unavailable"
    weight = np.eye(2, dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    cap = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    small = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
    )
    cap_bytes = RUNNER.codec_artifact_allocations_natural_file_bytes(
        [RUNNER._artifact_allocation(cap)],
        alignment=64,
    )
    ranked, report = RUNNER.rank_global_exact_component_allocations(
        strategy=RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        ql_candidates={layer: cap},
        option_pools={layer: [small]},
        metrics={layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
        alignment=64,
        optimality_scope="unit",
        candidate_pool_asymmetry="unit",
        top_k=2,
        required_natural_file_bytes=cap_bytes,
    )

    assert ranked == []
    assert report["exact_natural_file_byte_match_required"] is True
    assert report["exact_natural_file_byte_match_available"] is False
    assert report["selected_natural_file_bytes"] is None


def test_exact_natural_match_state_limit_disables_claim_without_approximating(
    monkeypatch,
) -> None:
    layer = "exact.state-limit"
    weight = np.eye(2, dtype=np.float32)
    q = RUNNER.QuantCodec(
        codes=np.zeros((2, 2), dtype=np.int8),
        scales=np.ones(2, dtype=np.float16),
        bits=4,
    )
    lowrank = RUNNER.LowRankCodec(
        left=np.array([[1.0], [0.0]], dtype=np.float16),
        right=np.array([[1.0, 0.0]], dtype=np.float16),
    )
    cap = _candidate("Q+L", weight, q, layer=layer, lowrank=lowrank)
    small = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
    )
    exact = _candidate(
        RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        weight,
        q,
        layer=layer,
        lowrank=lowrank,
    )
    required = RUNNER.codec_artifact_allocations_natural_file_bytes(
        [RUNNER._artifact_allocation(exact)],
        alignment=64,
    )
    kwargs = {
        "strategy": RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
        "ql_candidates": {layer: cap},
        "option_pools": {layer: [small, exact]},
        "metrics": {layer: RUNNER.HessianMetric(torch.eye(2), device="cpu")},
        "alignment": 64,
        "optimality_scope": "unit",
        "candidate_pool_asymmetry": "unit",
        "top_k": 2,
    }
    monkeypatch.setattr(RUNNER, "GLOBAL_ALLOCATOR_EXPANDED_STATE_LIMIT", 1)

    ranked, report = RUNNER.attempt_exact_natural_component_allocations(
        **kwargs,
        required_natural_file_bytes=required,
    )

    assert ranked == []
    assert report["search_status"] == "state_limit_exceeded"
    assert report["search_completed"] is False
    assert report["exact_natural_file_byte_match_available"] is False
    assert report["required_natural_file_bytes"] == required
    assert report["fallback_policy"] == (
        "retain_cap_best_nojoint_for_description_disable_joint_claim"
    )
    with pytest.raises(RuntimeError, match="expansion exceeded"):
        RUNNER.rank_global_exact_component_allocations(**kwargs)


def test_joint_claim_requires_a_completed_exact_natural_match_search() -> None:
    endpoint_rows = [
        {
            "strategy": "Q+S+L_QL_budget",
            "target_ratio": 0.3,
            "artifact_natural_file_bytes": 100,
            "artifact_file_bytes": 128,
            "heldout_evaluated": True,
            "heldout_nll": 1.0,
        },
        {
            "strategy": RUNNER.GLOBAL_NONJOINT_CONTROL_STRATEGY,
            "target_ratio": 0.3,
            "artifact_natural_file_bytes": 100,
            "artifact_file_bytes": 128,
            "heldout_evaluated": True,
            "heldout_nll": 2.0,
        },
    ]
    report = {"joint_control_natural_match_available": False}

    claim = RUNNER.annotate_joint_value_claim(
        endpoint_rows,
        endpoint_target=0.3,
        rate_allocator_report=report,
    )

    assert claim["same_natural_file_bytes"] is True
    assert claim["qsl_test_nll_gain_over_nojoint"] == pytest.approx(1.0)
    assert claim["exact_natural_match_search_succeeded"] is False
    assert claim["supported"] is False
    assert "did not complete" in claim["reason"]


def test_heterogeneous_family_prescreen_keeps_default_and_each_factor_width() -> None:
    weight = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)
    codecs = RUNNER.build_quantizer_candidate_codecs(
        weight,
        bit_widths=[3, 4],
        group_sizes=[0],
        quantizers=["symmetric_rtn", "symmetric_mse_clip"],
    )
    default_q = next(
        codec
        for codec in codecs
        if codec.bits == 4 and codec.quantizer == "symmetric_rtn"
    )
    selected, rows = RUNNER.screen_heterogeneous_candidate_families(
        layer="layer",
        weight=weight,
        quantizer_codecs=codecs,
        lowrank_factor_bits=[4, 16],
        default_q=default_q,
        metric=RUNNER.HessianMetric(torch.eye(2), device="cpu"),
        support_encoding="csr_fixed",
        target_ratio=0.5,
        top_k=1,
    )
    assert any(q is default_q and factor_bits == 16 for q, factor_bits in selected)
    assert {factor_bits for _q, factor_bits in selected} == {4, 16}
    assert sum(bool(row["selected_for_expensive_family_expansion"]) for row in rows) >= 2


def test_endpoint_aggregate_records_selected_codec_assignment() -> None:
    aggregate = RUNNER._empty_aggregate("Q+S+L_QL_budget", 0.5)
    base_metrics = {
        "s_active": False,
        "l_active": True,
        "reference_bits": 100,
        "payload_bits": 50,
        "hessian_cost": 1.0,
        "baseline_hessian_energy": 4.0,
        "hessian_self_q": 1.0,
        "hessian_self_s": 0.0,
        "hessian_self_l": 0.0,
        "hessian_cross_qs": 0.0,
        "hessian_cross_ql": 0.0,
        "hessian_cross_sl": 0.0,
        "sparse_nnz": 0,
        "q_scale_count": 2,
        "folded_repair_dof": 0,
        "lowrank_rank": 1,
        "activation_reconstruction_error": 0.5,
        "worst_token_risk": 0.5,
        "token_risk_p95": 0.5,
        "comparison_budget_bits": 50,
        "rate_cap_strategy": "global",
        "q_bits": 4,
        "q_quantizer": "symmetric_rtn",
        "q_col_block_size": 0,
        "lowrank_factor_bits": 16,
    }
    RUNNER.update_aggregate(
        aggregate,
        {**base_metrics, "layer": "layer.0"},
    )
    RUNNER.update_aggregate(
        aggregate,
        {
            **base_metrics,
            "layer": "layer.1",
            "q_bits": 3,
            "q_quantizer": "symmetric_mse_clip",
            "q_col_block_size": 64,
            "lowrank_factor_bits": 4,
        },
    )
    row = RUNNER.finalize_aggregate(
        aggregate,
        rate_tolerance=0.05,
        rho_threshold=0.2,
    )

    assert json.loads(row["q_bits_by_layer"]) == {
        "layer.0": 4,
        "layer.1": 3,
    }
    assert json.loads(row["q_quantizers_by_layer"]) == {
        "layer.0": "symmetric_rtn",
        "layer.1": "symmetric_mse_clip",
    }
    assert json.loads(row["q_group_sizes_by_layer"]) == {
        "layer.0": 0,
        "layer.1": 64,
    }
    assert json.loads(row["lowrank_factor_bits_by_layer"]) == {
        "layer.0": 16,
        "layer.1": 4,
    }
    assert json.loads(row["q_bits_distribution"]) == {"3": 1, "4": 1}
