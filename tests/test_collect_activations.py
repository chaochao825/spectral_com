import numpy as np

from llm_spectral_dynamics.collect_activations import (
    activation_to_matrix,
    collect_synthetic_statistics,
    hook_sites_for_requested_sites,
    _eigenspectrum_from_accumulator,
    validate_sites,
)
from llm_spectral_dynamics.streaming_covariance import RunningCovariance


def test_activation_to_matrix_excludes_first_tokens():
    arr = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    matrix = activation_to_matrix(arr, exclude_first_tokens=1)
    assert matrix.shape == (6, 3)
    np.testing.assert_array_equal(matrix[0], arr[0, 1])
    empty = activation_to_matrix(arr, exclude_first_tokens=4)
    assert empty.shape == (0, 3)


def test_synthetic_collection_respects_exclude_first_tokens():
    result = collect_synthetic_statistics(
        model_name="synthetic",
        variant="pretrained",
        sites=["resid_post"],
        layers=[0],
        num_sequences=3,
        seq_len=5,
        seed=0,
        sample_limit=8,
        powerlaw_rank_min=2,
        powerlaw_rank_max=8,
        bootstrap_samples=8,
        dynamic_enabled=True,
        dynamic_site="resid_post",
        dynamic_layer=0,
        dynamic_pca_rank=4,
        dynamic_max_sequences=3,
        dynamic_lags=[1],
        exclude_first_tokens=2,
    )
    assert result.metric_rows[0]["n_samples"] == 9
    assert result.dynamic_rows[0]["tau"] == 1


def test_validate_sites_fails_fast_for_planned_unimplemented_site():
    try:
        validate_sites(["resid_post", "ffn_intermediate"])
    except ValueError as exc:
        assert "planned but not implemented" in str(exc)
    else:
        raise AssertionError("validate_sites should reject unimplemented planned sites")


def test_resid_mid_adds_attn_out_hook_dependency():
    hooks = hook_sites_for_requested_sites(["resid_mid"])
    assert hooks == ["attn_out"]


def test_resid_post_uses_layer_hook_dependency():
    hooks = hook_sites_for_requested_sites(["resid_post", "attn_out", "mlp_out"])
    assert hooks == ["attn_out", "mlp_out", "resid_post"]


def test_small_sample_large_dim_uses_sample_space_spectrum():
    rng = np.random.default_rng(0)
    acc = RunningCovariance(sample_limit=4, seed=0)
    acc.update(rng.normal(size=(4, 16)))
    eig, method, samples = _eigenspectrum_from_accumulator(acc, output_zscore=False)
    assert method == "sample_space_reservoir"
    assert samples.shape == (4, 16)
    assert eig.shape[0] == 4
