import numpy as np

from llm_spectral_dynamics.streaming_covariance import RunningCovariance, sample_space_covariance_eigenvalues


def test_running_covariance_matches_numpy_cov():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(200, 5))
    acc = RunningCovariance(sample_limit=16, seed=0)
    acc.update(values[:50])
    acc.update(values[50:123])
    acc.update(values[123:])
    np.testing.assert_allclose(acc.mean, values.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(acc.covariance(), np.cov(values, rowvar=False), atol=1e-12)
    assert acc.sample_array().shape == (16, 5)


def test_sample_space_eigenvalues_match_covariance_rank():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(20, 50))
    eig = sample_space_covariance_eigenvalues(values)
    assert eig.shape[0] == 20
    assert np.all(eig[:-1] >= eig[1:])
    assert np.all(eig >= -1e-12)


def test_large_dim_reservoir_skips_full_covariance_storage():
    rng = np.random.default_rng(7)
    acc = RunningCovariance(sample_limit=4, seed=0)
    acc.update(rng.normal(size=(8, 16)))
    assert acc.store_full_covariance is False
    assert acc.m2 is None
    assert acc.sample_array().shape == (4, 16)
    try:
        acc.covariance()
    except ValueError as exc:
        assert "full covariance was not stored" in str(exc)
    else:
        raise AssertionError("covariance should fail when full covariance storage is disabled")
