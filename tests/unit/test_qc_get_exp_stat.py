from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesprism.qc import _bh_adjust, get_exp_stat


def _parallel_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(11)
    genes = [f"g{i}" for i in range(1, 7)]

    labels_state = (
        ["A_s1"] * 3 +
        ["A_s2"] * 3 +
        ["B_s1"] * 3 +
        ["B_s2"] * 3
    )
    labels_type = ["A"] * 6 + ["B"] * 6

    means = {
        "A_s1": np.array([30, 25, 12, 3, 2, 1], dtype=float),
        "A_s2": np.array([15, 28, 26, 4, 2, 1], dtype=float),
        "B_s1": np.array([2, 3, 5, 30, 25, 8], dtype=float),
        "B_s2": np.array([2, 2, 4, 20, 27, 24], dtype=float),
    }

    rows = []
    for state in labels_state:
        rows.append(rng.poisson(lam=means[state]))

    sc_dat = pd.DataFrame(rows, columns=genes)
    return sc_dat, labels_type, labels_state


def test_get_exp_stat_validation_errors() -> None:
    sc_dat = pd.DataFrame([[1, 2], [3, 4]], columns=["g1", "g2"])

    with pytest.raises(TypeError):
        get_exp_stat(
            sc_dat=np.array([[1, 2], [3, 4]]),
            cell_type_labels=["A", "B"],
            cell_state_labels=["A1", "B1"],
        )

    with pytest.raises(ValueError, match="length of cell_type_labels"):
        get_exp_stat(
            sc_dat=sc_dat,
            cell_type_labels=["A"],
            cell_state_labels=["A1", "B1"],
        )

    with pytest.raises(ValueError, match="pseudo_count"):
        get_exp_stat(
            sc_dat=sc_dat,
            cell_type_labels=["A", "B"],
            cell_state_labels=["A1", "B1"],
            pseudo_count=0,
        )

    with pytest.raises(ValueError, match="n_cores"):
        get_exp_stat(
            sc_dat=sc_dat,
            cell_type_labels=["A", "B"],
            cell_state_labels=["A1", "B1"],
            n_cores=0,
        )


def test_get_exp_stat_parallel_matches_single_core() -> None:
    sc_dat, labels_type, labels_state = _parallel_dataset()

    one = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=1,
        n_cores=1,
    )
    four = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=1,
        n_cores=4,
    )

    assert list(one.keys()) == list(four.keys())
    for key in one:
        np.testing.assert_allclose(
            one[key].to_numpy(),
            four[key].to_numpy(),
            atol=1e-12,
            rtol=0,
            equal_nan=True,
        )


def test_get_exp_stat_excludes_same_cell_type_pairs() -> None:
    sc_dat = pd.DataFrame(
        [
            [40, 5, 2],
            [39, 6, 1],
            [11, 30, 4],
            [10, 31, 3],
        ],
        columns=["g1", "g2", "g3"],
    )
    labels_type = ["A", "A", "A", "A"]
    labels_state = ["A_s1", "A_s1", "A_s2", "A_s2"]

    out = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=1,
        n_cores=1,
    )

    assert out["A"]["pval.up.min"].isna().all()
    assert out["A"]["min.lfc"].isna().all()


def test_get_exp_stat_library_size_scaling_invariant() -> None:
    sc_dat = pd.DataFrame(
        [
            [100.0, 200.0, 300.0],
            [50.0, 100.0, 150.0],
            [80.0, 160.0, 240.0],
            [40.0, 80.0, 120.0],
        ],
        columns=["g1", "g2", "g3"],
    )
    labels_type = ["A", "A", "B", "B"]
    labels_state = ["A_s1", "A_s1", "B_s1", "B_s1"]

    out = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=1,
        n_cores=1,
    )

    assert np.all(np.abs(out["A"]["min.lfc"].to_numpy(dtype=float)) < 1e-12)


def test_get_exp_stat_low_count_second_state_filter() -> None:
    sc_dat = pd.DataFrame(
        [
            [40, 5, 2],
            [38, 6, 1],
            [41, 4, 2],
            [3, 31, 2],
            [4, 28, 2],
        ],
        columns=["g1", "g2", "g3"],
    )
    labels_type = ["A", "A", "A", "B", "B"]
    labels_state = ["A_s1", "A_s1", "A_s1", "B_low", "B_low"]

    cutoff3 = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=3,
        n_cores=1,
    )
    cutoff1 = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=labels_type,
        cell_state_labels=labels_state,
        pseudo_count=0.1,
        cell_count_cutoff=1,
        n_cores=1,
    )

    assert cutoff3["A"]["pval.up.min"].isna().all()
    assert cutoff1["A"]["pval.up.min"].notna().any()


def test_bh_adjust_known_values() -> None:
    p_values = np.array([0.01, 0.02, np.nan, 0.03, 1.0], dtype=float)
    adjusted = _bh_adjust(p_values)
    expected = np.array([0.04, 0.04, np.nan, 0.04, 1.0], dtype=float)
    np.testing.assert_allclose(adjusted, expected, atol=1e-12, rtol=0, equal_nan=True)
