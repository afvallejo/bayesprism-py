from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesprism.preprocess import collapse, filter_bulk_outlier, norm_to_one, validate_input


def test_norm_to_one_row_sums_to_one() -> None:
    ref = pd.DataFrame(
        [[10.0, 0.0, 5.0], [4.0, 2.0, 1.0]],
        index=["A", "B"],
        columns=["g1", "g2", "g3"],
    )
    phi = norm_to_one(ref, pseudo_min=1e-8)
    np.testing.assert_allclose(phi.sum(axis=1).to_numpy(), np.ones(2), atol=1e-12)


def test_collapse_preserves_first_label_order() -> None:
    ref = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=["c1", "c2", "c3"],
        columns=["g1", "g2"],
    )
    out = collapse(ref, ["x", "y", "x"])
    assert list(out.index) == ["x", "y"]
    np.testing.assert_allclose(out.loc["x"].to_numpy(), np.array([6, 8]))


def test_validate_input_rejects_negative() -> None:
    bad = pd.DataFrame([[-1.0, 2.0]], columns=["g1", "g2"])
    with pytest.raises(ValueError, match="negative"):
        validate_input(bad)


def test_filter_bulk_outlier_filters_expected_gene() -> None:
    mixture = pd.DataFrame(
        [[100, 1, 1], [90, 1, 1]],
        index=["m1", "m2"],
        columns=["g_outlier", "g2", "g3"],
    )
    filtered = filter_bulk_outlier(mixture, outlier_cut=0.8, outlier_fraction=0.5)
    assert "g_outlier" not in filtered.columns
