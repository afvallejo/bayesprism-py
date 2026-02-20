from __future__ import annotations

import numpy as np

from bayesprism.gibbs import get_gibbs_idx, sample_z_theta_n


def test_sample_z_theta_n_invariants() -> None:
    x_n = np.array([10, 5, 3])
    phi = np.array([[0.6, 0.2, 0.2], [0.4, 0.8, 0.8]])
    gibbs_idx = get_gibbs_idx({"chain_length": 100, "burn_in": 50, "thinning": 2})

    out = sample_z_theta_n(x_n, phi, alpha=1.0, gibbs_idx=gibbs_idx)
    assert out["Z_n"].shape == (3, 2)
    assert out["theta_n"].shape == (2,)
    np.testing.assert_allclose(np.sum(out["theta_n"]), 1.0, atol=1e-12)
