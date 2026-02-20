from __future__ import annotations

import numpy as np
import pandas as pd

from bayesprism.models import RefPhi
from bayesprism.reference_update import transform_phi_t, update_reference


def test_transform_phi_t_is_normalized() -> None:
    phi_t = np.array([0.2, 0.3, 0.5])
    gamma_t = np.array([0.1, -0.2, 0.3])
    out = transform_phi_t(phi_t, gamma_t)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)


def test_update_reference_refphi_mle_shapes() -> None:
    phi = RefPhi(
        phi=pd.DataFrame(
            [[0.6, 0.4], [0.3, 0.7]],
            index=["A", "B"],
            columns=["g1", "g2"],
        ),
        pseudo_min=1e-8,
    )
    z = np.array(
        [
            [[10.0, 3.0], [2.0, 6.0]],
            [[8.0, 4.0], [1.0, 7.0]],
        ]
    )

    out = update_reference(
        Z=z,
        phi_prime=phi,
        map={"A": ["A"], "B": ["B"]},
        key=None,
        opt_control={"optimizer": "MLE", "maxit": 2000},
    )
    assert out.phi.shape == phi.phi.shape
    np.testing.assert_allclose(out.phi.sum(axis=1).to_numpy(), np.ones(2), atol=1e-12)
