import numpy as np

from llm_spectral_dynamics.dmd import exact_dmd, summarize_dmd_eigenvalues


def test_dmd_recovers_linear_transition_eigenvalues():
    a = np.diag([0.9, 0.5])
    x = np.array([1.0, 2.0])
    seq = []
    for _ in range(40):
        seq.append(x.copy())
        x = a @ x
    result = exact_dmd(np.asarray(seq), rank=2)
    eig = np.sort(np.real(result.eigenvalues))
    np.testing.assert_allclose(eig, [0.5, 0.9], atol=1e-6)
    summary = summarize_dmd_eigenvalues(result.eigenvalues)
    assert summary["dmd_real_ratio"] == 1.0

