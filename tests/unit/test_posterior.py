from __future__ import annotations

import numpy as np

from bayesprism.posterior import merge_k, new_joint_post


def test_new_joint_post_and_merge_k_shapes() -> None:
    gibbs_list = [
        {
            "Z_n": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "theta_n": np.array([0.4, 0.6]),
            "theta_cv_n": np.array([0.1, 0.2]),
            "gibbs_constant": 0.0,
        }
    ]
    jp = new_joint_post(
        bulk_id=["b1"],
        gene_id=["g1", "g2"],
        cell_type=["s1", "s2"],
        gibbs_list=gibbs_list,
    )
    merged = merge_k(jp, {"t1": ["s1", "s2"]})
    assert merged.Z.shape == (1, 2, 1)
    np.testing.assert_allclose(merged.theta.to_numpy(), np.array([[1.0]]))
