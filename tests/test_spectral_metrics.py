import numpy as np

from llm_spectral_dynamics.spectral_metrics import (
    condition_number,
    covariance_eigenspectrum,
    effective_rank,
    explained_variance_at_k,
    participation_ratio,
    spectral_entropy,
)


def test_spectral_metrics_known_diagonal_covariance():
    cov = np.diag([4.0, 2.0, 1.0])
    eig = covariance_eigenspectrum(cov)
    np.testing.assert_allclose(eig, [4.0, 2.0, 1.0])
    assert explained_variance_at_k(eig, [1])["top_1_explained_variance"] == 4.0 / 7.0
    assert participation_ratio(eig) == (7.0 * 7.0) / 21.0
    assert effective_rank(eig) > 1.0
    assert 0.0 < spectral_entropy(eig) <= 1.0
    assert condition_number(eig) == 4.0

