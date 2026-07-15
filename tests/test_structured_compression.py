import math
from types import SimpleNamespace

import numpy as np
import pytest

import llm_spectral_dynamics.structured.evaluation as evaluation_module
from llm_spectral_dynamics.structured.approximations import (
    approximate_weight,
    batched_svd_factors,
    block_circulant_approximation,
    low_rank_approximation,
    low_rank_approximation_from_factors,
    monarch_like_approximation,
    monarch_param_count,
    svd_factors,
)
from llm_spectral_dynamics.structured.adapters import AdapterWrappedLinear, adapter_spec
from llm_spectral_dynamics.structured.metrics import singular_values, weight_spectrum_metrics
from llm_spectral_dynamics.structured.quantization import structured_quantization_rows, symmetric_quantize
from llm_spectral_dynamics.structured.replacement import StructuredLinear
from llm_spectral_dynamics.structured.residuals import build_residual, residual_analysis_rows, residual_budget_params
from llm_spectral_dynamics.structured.phase3 import _append_replaced_modules, _best_methods_from_phase1, _method_mapping, _stage_modules
from llm_spectral_dynamics.structured.phase2 import _linear_output
from llm_spectral_dynamics.structured.evaluation import ZERO_SHOT_BACKUP_NAMES, _choice_continuation, _conditional_nll, _validate_zero_shot_dataset
from llm_spectral_dynamics.structured.rotation import hadamard_rotate_columns
from llm_spectral_dynamics.structured.utils import set_global_seed


def test_svd_metrics_recover_synthetic_rank():
    rng = np.random.default_rng(0)
    u, _ = np.linalg.qr(rng.normal(size=(16, 4)))
    v, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    s = np.array([4.0, 2.0, 1.0, 0.5])
    weight = (u * s) @ v.T
    values = singular_values(weight)
    metrics = weight_spectrum_metrics(weight, values)
    assert metrics["rank_99"] <= 4
    assert metrics["effective_rank"] > 1.0
    assert math.isclose(values[0], 4.0, rel_tol=1e-5)


def test_low_rank_error_matches_singular_tail_energy():
    rng = np.random.default_rng(1)
    weight = rng.normal(size=(10, 8)).astype(np.float32)
    result = low_rank_approximation(weight, rank=3)
    s = np.linalg.svd(weight, compute_uv=False)
    expected = np.sqrt(np.sum(s[3:] ** 2)) / np.linalg.norm(weight)
    observed = np.linalg.norm(weight - result.matrix) / np.linalg.norm(weight)
    assert math.isclose(float(observed), float(expected), rel_tol=1e-5)
    assert result.params == 3 * (10 + 8)


def test_low_rank_cached_factors_match_direct_approximation():
    rng = np.random.default_rng(10)
    weight = rng.normal(size=(14, 9)).astype(np.float32)
    direct = low_rank_approximation(weight, rank=4)
    cached = low_rank_approximation_from_factors(weight.shape, svd_factors(weight), rank=4)
    assert cached.params == direct.params
    assert cached.rank == direct.rank
    assert np.allclose(cached.matrix, direct.matrix)


def test_batched_svd_matches_individual_svd():
    rng = np.random.default_rng(12)
    blocks = rng.normal(size=(5, 6, 4)).astype(np.float32)
    u, s, vh = batched_svd_factors(blocks)
    reconstructed = (u * s[:, None, :]) @ vh
    assert np.allclose(reconstructed, blocks, atol=1e-5)
    assert np.allclose(s, np.stack([np.linalg.svd(block, compute_uv=False) for block in blocks]), atol=1e-5)


def test_block_circulant_projection_preserves_shape_and_budget():
    rng = np.random.default_rng(2)
    weight = rng.normal(size=(13, 9)).astype(np.float32)
    result = block_circulant_approximation(weight, budget=100, block_sizes=[4, 8])
    assert result.matrix.shape == weight.shape
    assert result.params <= 100
    assert result.block_size == 4


def test_vectorized_block_circulant_matches_blockwise_projection():
    rng = np.random.default_rng(13)
    weight = rng.normal(size=(13, 9)).astype(np.float32)
    result = block_circulant_approximation(weight, budget=100, block_sizes=[4, 8])
    expected = np.zeros((16, 12), dtype=np.float32)
    padded = np.zeros_like(expected)
    padded[:13, :9] = weight
    for row in range(0, 16, 4):
        for col in range(0, 12, 4):
            block = padded[row : row + 4, col : col + 4]
            coeff = np.array([np.mean([block[i, (i + shift) % 4] for i in range(4)]) for shift in range(4)])
            expected[row : row + 4, col : col + 4] = coeff[(np.arange(4)[None, :] - np.arange(4)[:, None]) % 4]
    assert np.allclose(result.matrix, expected[:13, :9])


def test_monarch_like_is_deterministic_and_counts_params():
    rng = np.random.default_rng(3)
    weight = rng.normal(size=(16, 16)).astype(np.float32)
    a = monarch_like_approximation(weight, budget=128, block_size=8, terms=2)
    b = monarch_like_approximation(weight, budget=128, block_size=8, terms=2)
    assert np.allclose(a.matrix, b.matrix)
    assert a.params == monarch_param_count(weight.shape, 8, a.rank, 2)
    assert a.matrix.shape == weight.shape


def test_monarch_like_full_coverage_reconstructs_blocks_and_counts_executed_factors():
    rng = np.random.default_rng(11)
    weight = rng.normal(size=(8, 8)).astype(np.float32)
    result = monarch_like_approximation(weight, budget=10_000, block_size=4, terms=2)
    assert result.rank == 4
    assert result.terms == 2
    assert result.params == 2 * 2 * 4 * 2 * 4
    assert np.allclose(result.matrix, weight, atol=1e-5)


def test_monarch_param_count_uses_executed_row_blocks_and_caps_rank():
    assert monarch_param_count((24, 80), block_size=8, rank_per_block=20, terms=2) == 2 * 3 * 8 * 2 * 8


def test_residual_budget_and_variants_have_expected_shapes():
    rng = np.random.default_rng(4)
    residual = rng.normal(size=(12, 8)).astype(np.float32)
    budget = residual_budget_params(residual, 0.25)
    assert budget == 24
    for kind in ["none", "low_rank", "sparse", "channel"]:
        result = build_residual(residual, residual_type=kind, residual_fraction=0.25)
        assert result.matrix.shape == residual.shape
        assert result.params <= max(budget, residual.size)


def test_residual_analysis_accepts_precomputed_factors():
    rng = np.random.default_rng(14)
    weight = rng.normal(size=(10, 8)).astype(np.float32)
    structured = weight * 0.75
    residual = weight - structured
    factors = svd_factors(residual)
    direct = residual_analysis_rows(
        weight,
        structured,
        compression_ratio=4,
        residual_fractions=[0.0, 0.1],
        residual_types=["low_rank"],
    )
    cached = residual_analysis_rows(
        weight,
        structured,
        compression_ratio=4,
        residual_fractions=[0.0, 0.1],
        residual_types=["low_rank"],
        residual_factors=factors,
    )
    assert direct == cached


def test_approximate_weight_dispatches_all_methods():
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(12, 12)).astype(np.float32)
    for method in ["low_rank", "block_circulant", "monarch_like"]:
        result = approximate_weight(weight, method=method, compression_ratio=4, block_sizes=[4, 8], monarch_block_size=4)
        assert result.matrix.shape == weight.shape
        assert result.params > 0


def test_structured_linear_output_shape():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    weight = rng.normal(size=(5, 7)).astype(np.float32)
    bias = rng.normal(size=(5,)).astype(np.float32)
    module = StructuredLinear(weight, bias).to_module()
    x = torch.randn(3, 7)
    y = module(x)
    assert tuple(y.shape) == (3, 5)


def test_structured_linear_preserves_requested_device_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuda is not available")
    weight = np.ones((2, 3), dtype=np.float32)
    module = StructuredLinear(weight).to_module(dtype=torch.float16, device="cuda:0")
    x = torch.ones(4, 3, device="cuda:0", dtype=torch.float16)
    y = module(x)
    assert y.device.type == "cuda"
    assert y.dtype == torch.float16
    assert tuple(y.shape) == (4, 2)


def test_phase3_stage_labels_follow_actual_target_subset():
    class Record:
        def __init__(self, module_type):
            self.module_type = module_type

    stage = {"name": "up_gate_proj", "modules": ["up_proj", "gate_proj"]}
    assert _stage_modules(stage, ["down_proj"]) == []
    assert _stage_modules(stage, ["gate_proj", "down_proj"]) == ["gate_proj"]
    assert _append_replaced_modules(["down_proj"], [Record("down_proj"), Record("gate_proj")]) == ["down_proj", "gate_proj"]


def test_phase3_best_method_uses_median_across_layers(tmp_path):
    metrics_dir = tmp_path / "phase1" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "approximation_errors.csv").write_text(
        "module_type,method,compression_ratio_target,relative_weight_error,name,layer\n"
        "q_proj,low_rank,4,0.01,layer0.q_proj,0\n"
        "q_proj,low_rank,4,0.90,layer1.q_proj,1\n"
        "q_proj,low_rank,4,0.90,layer2.q_proj,2\n"
        "q_proj,block_circulant,4,0.50,layer0.q_proj,0\n"
        "q_proj,block_circulant,4,0.50,layer1.q_proj,1\n"
        "q_proj,block_circulant,4,0.50,layer2.q_proj,2\n"
        "q_proj,monarch_like,4,0.70,layer0.q_proj,0\n"
        "q_proj,monarch_like,4,0.70,layer1.q_proj,1\n"
        "q_proj,monarch_like,4,0.70,layer2.q_proj,2\n",
        encoding="utf-8",
    )
    assert _best_methods_from_phase1(tmp_path, 4) == {"q_proj": "block_circulant"}


@pytest.mark.parametrize("missing_row", ["method", "layer"])
def test_phase3_best_method_rejects_incomplete_method_or_layer_coverage(tmp_path, missing_row):
    metrics_dir = tmp_path / "phase1" / "metrics"
    metrics_dir.mkdir(parents=True)
    rows = [
        "q_proj,low_rank,4,0.4,layer0.q_proj,0",
        "q_proj,low_rank,4,0.4,layer1.q_proj,1",
        "q_proj,block_circulant,4,0.5,layer0.q_proj,0",
        "q_proj,block_circulant,4,0.5,layer1.q_proj,1",
        "q_proj,monarch_like,4,0.6,layer0.q_proj,0",
        "q_proj,monarch_like,4,0.6,layer1.q_proj,1",
    ]
    if missing_row == "method":
        rows = [row for row in rows if ",monarch_like," not in row]
    else:
        rows.remove("q_proj,monarch_like,4,0.6,layer1.q_proj,1")
    (metrics_dir / "approximation_errors.csv").write_text(
        "module_type,method,compression_ratio_target,relative_weight_error,name,layer\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing methods|inconsistent layer coverage"):
        _best_methods_from_phase1(tmp_path, 4)


@pytest.mark.parametrize(
    "last_row",
    [
        "q_proj,monarch_like,4,0.7,layer0.q_proj,0",
        "q_proj,monarch_like,4,0.7,layer1.q_proj,9",
    ],
)
def test_phase3_best_method_rejects_duplicate_or_layer_mismatch(tmp_path, last_row):
    metrics_dir = tmp_path / "phase1" / "metrics"
    metrics_dir.mkdir(parents=True)
    rows = [
        "q_proj,low_rank,4,0.4,layer0.q_proj,0",
        "q_proj,low_rank,4,0.4,layer1.q_proj,1",
        "q_proj,block_circulant,4,0.5,layer0.q_proj,0",
        "q_proj,block_circulant,4,0.5,layer1.q_proj,1",
        "q_proj,monarch_like,4,0.6,layer0.q_proj,0",
        "q_proj,monarch_like,4,0.6,layer1.q_proj,1",
        last_row,
    ]
    (metrics_dir / "approximation_errors.csv").write_text(
        "module_type,method,compression_ratio_target,relative_weight_error,name,layer\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate rows|inconsistent layer coverage"):
        _best_methods_from_phase1(tmp_path, 4)


def test_phase3_best_method_fails_closed_when_phase1_selection_is_incomplete(tmp_path):
    cfg = {
        "output_dir": str(tmp_path),
        "target_modules": ["q_proj", "k_proj"],
        "phase3": {"structure": "best_weight_error"},
    }
    with pytest.raises(ValueError, match="q_proj.*k_proj"):
        _method_mapping(cfg, 4)


def test_phase2_linear_output_includes_bias():
    x = np.array([[1.0, 2.0]], dtype=np.float32)
    weight = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    bias = np.array([7.0, 8.0], dtype=np.float32)
    assert np.allclose(_linear_output(x, weight, bias), np.array([[18.0, 25.0]], dtype=np.float32))


def test_zero_shot_backup_names_cover_formal_tasks():
    assert ZERO_SHOT_BACKUP_NAMES == {
        "piqa": "piqa",
        "arc_easy": "ai2_arc_easy",
        "hellaswag": "hellaswag",
    }


def test_zero_shot_dataset_validation_rejects_empty_and_missing_columns():
    class FakeDataset:
        def __init__(self, rows, columns):
            self.rows = rows
            self.column_names = columns

        def __len__(self):
            return self.rows

    with pytest.raises(ValueError, match="empty"):
        _validate_zero_shot_dataset("piqa", FakeDataset(0, ["goal", "sol1", "sol2", "label"]))
    with pytest.raises(ValueError, match="missing columns"):
        _validate_zero_shot_dataset("piqa", FakeDataset(1, ["goal", "label"]))


def test_conditional_nll_uses_combined_tokenization_and_offsets():
    torch = pytest.importorskip("torch")

    class BoundaryTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "offset_mapping": torch.tensor([[[0, 1], [1, 3], [3, 5]]]),
            }

        def encode(self, *_args, **_kwargs):
            raise AssertionError("separate tokenization must not be used when offsets are available")

    class UniformModel:
        def __call__(self, *, input_ids):
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 8), device=input_ids.device))

    tokenizer = BoundaryTokenizer()
    score = _conditional_nll(UniformModel(), tokenizer, "a", " bcde", device="cpu")
    assert tokenizer.calls[0][0] == "a bcde"
    assert math.isclose(score, math.log(8), rel_tol=1e-6)
    assert _choice_continuation("answer") == " answer"
    assert _choice_continuation(" answer") == " answer"


def test_conditional_nll_strict_mode_rejects_offset_fallback(monkeypatch):
    class NoOffsetsTokenizer:
        def __call__(self, *_args, **_kwargs):
            raise NotImplementedError("offsets unavailable")

        def encode(self, *_args, **_kwargs):
            raise AssertionError("strict mode must not use separate tokenization")

    monkeypatch.setenv("LLM_SC_ZERO_SHOT_STRICT", "1")
    with pytest.raises(RuntimeError, match="requires combined tokenization"):
        _conditional_nll(object(), NoOffsetsTokenizer(), "prompt", " answer", device="cpu")


def test_conditional_nll_strict_mode_rejects_empty_continuation_positions(monkeypatch):
    torch = pytest.importorskip("torch")

    class EmptyContinuationTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "offset_mapping": torch.tensor([[[0, 3], [3, 6]]]),
            }

    monkeypatch.setenv("LLM_SC_ZERO_SHOT_STRICT", "1")
    with pytest.raises(RuntimeError, match="no continuation token positions"):
        _conditional_nll(object(), EmptyContinuationTokenizer(), "long prompt", " answer", device="cpu")


def test_evaluate_zero_shot_strict_mode_rejects_non_finite_scores(monkeypatch):
    monkeypatch.setenv("LLM_SC_ZERO_SHOT_STRICT", "1")
    monkeypatch.setattr(evaluation_module, "_load_zero_shot_examples", lambda _task, _limit: [("prompt", [" a", " b"], 0)])
    monkeypatch.setattr(evaluation_module, "_conditional_nll", lambda *_args, **_kwargs: float("inf"))
    with pytest.raises(RuntimeError, match="zero-shot task piqa failed"):
        evaluation_module.evaluate_zero_shot(object(), object(), tasks=["piqa"], limit=1, device="cpu")


def test_set_global_seed_repeats_adapter_initialization():
    torch = pytest.importorskip("torch")
    set_global_seed(123)
    first = AdapterWrappedLinear(torch.nn.Linear(6, 4, bias=False), method="lora", budget=32, rank=2).to_module()
    first_params = [param.detach().clone() for param in first.update_modules.parameters()]
    set_global_seed(123)
    second = AdapterWrappedLinear(torch.nn.Linear(6, 4, bias=False), method="lora", budget=32, rank=2).to_module()
    second_params = [param.detach().clone() for param in second.update_modules.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(first_params, second_params))


def test_adapter_wrapper_registers_trainable_parameters():
    torch = pytest.importorskip("torch")
    base = torch.nn.Linear(6, 4, bias=False)
    wrapped = AdapterWrappedLinear(base, method="lora", budget=32, rank=2).to_module()
    trainable = [param for param in wrapped.parameters() if param.requires_grad]
    assert sum(param.numel() for param in trainable) == 2 * (6 + 4)
    y = wrapped(torch.randn(3, 6))
    assert tuple(y.shape) == (3, 4)


def test_adapter_specs_cover_phase4_methods():
    for method in ["structured", "structured_lora", "lora", "mora", "fourierft", "bca"]:
        spec = adapter_spec(method, 8, 6, budget=64, rank=2, block_size=4)
        assert spec.params > 0
        assert spec.params <= 64
        assert spec.within_budget
        assert spec.method == method


def test_adapter_wrapper_caps_rank_to_budget():
    torch = pytest.importorskip("torch")
    base = torch.nn.Linear(20, 10, bias=False)
    wrapped = AdapterWrappedLinear(base, method="lora", budget=64, rank=8).to_module()
    trainable = [param for param in wrapped.parameters() if param.requires_grad]
    assert sum(param.numel() for param in trainable) <= 64
    assert wrapped.update.weight_matrix().shape == base.weight.shape


def test_hadamard_rotation_and_quantization_preserve_shape():
    rng = np.random.default_rng(7)
    weight = rng.normal(size=(5, 6)).astype(np.float32)
    rotated = hadamard_rotate_columns(weight)
    quant = symmetric_quantize(rotated, bits=3)
    assert rotated.shape == weight.shape
    assert quant.shape == weight.shape
    assert math.isclose(float(np.linalg.norm(rotated)), float(np.linalg.norm(weight)), rel_tol=1e-6)


def test_hadamard_rotation_preserves_norm_for_qwen_like_width():
    rng = np.random.default_rng(9)
    weight = rng.normal(size=(3, 8960)).astype(np.float32)
    rotated = hadamard_rotate_columns(weight)
    assert rotated.shape == weight.shape
    assert math.isclose(float(np.linalg.norm(rotated)), float(np.linalg.norm(weight)), rel_tol=1e-6)


def test_structured_quantization_rows_include_all_bits():
    rng = np.random.default_rng(8)
    weight = rng.normal(size=(8, 8)).astype(np.float32)
    rows = structured_quantization_rows(
        weight,
        compression_ratio=4,
        method="low_rank",
        bit_widths=[4, 2],
        residual_fraction=0.1,
        residual_type="low_rank",
        block_sizes=[4],
        monarch_block_size=4,
        monarch_terms=2,
        residual_precision="float16",
    )
    assert [row["bit_width"] for row in rows] == [4, 2]
    assert all(row["relative_structured_quantized_error"] >= 0 for row in rows)
    assert all(row["residual_precision"] == "float16" for row in rows)
