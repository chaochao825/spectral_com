import numpy as np

from llm_spectral_dynamics.fit_powerlaw import bootstrap_powerlaw_ci, fit_powerlaw


def test_powerlaw_fit_recovers_exponent():
    ranks = np.arange(1, 200, dtype=np.float64)
    eig = 10.0 * ranks**-1.5
    fit = fit_powerlaw(eig, rank_min=2, rank_max=120)
    assert abs(fit.alpha - 1.5) < 1e-3
    assert fit.r2 > 0.999


def test_powerlaw_bootstrap_ci_is_ordered():
    ranks = np.arange(1, 80, dtype=np.float64)
    eig = ranks**-1.2
    fit = bootstrap_powerlaw_ci(eig, rank_min=2, rank_max=60, n_boot=32, seed=1)
    assert fit.ci_low is not None
    assert fit.ci_high is not None
    assert fit.ci_low <= fit.alpha <= fit.ci_high

