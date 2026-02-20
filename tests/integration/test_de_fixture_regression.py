from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayesprism.qc import get_exp_stat, select_marker

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "de_small"


def _load_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    npz = np.load(FIXTURE_DIR / "de_fixture.npz", allow_pickle=True)
    metadata = json.loads(
        (FIXTURE_DIR / "de_fixture_metadata.json").read_text(encoding="utf-8")
    )
    return {k: npz[k] for k in npz.files}, metadata


def _run_python_de(
    data: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sc_dat = pd.DataFrame(
        data["sc_dat"],
        index=data["sc_dat_rows"].astype(str),
        columns=data["sc_dat_cols"].astype(str),
    )

    params = metadata["params"]
    stat = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=[str(x) for x in data["labels_cell_type"]],
        cell_state_labels=[str(x) for x in data["labels_cell_state"]],
        pseudo_count=float(params["pseudo_count"]),
        cell_count_cutoff=int(params["cell_count_cutoff"]),
        n_cores=2,
    )

    marker = select_marker(
        sc_dat=sc_dat,
        stat=stat,
        pval_max=float(params["pval_max"]),
        lfc_min=float(params["lfc_min"]),
    )
    return stat, marker


def test_get_exp_stat_matches_de_fixture() -> None:
    data, metadata = _load_fixture()
    stat, _ = _run_python_de(data, metadata)

    ct_names = [str(x) for x in data["ct_names"]]
    for ct in ct_names:
        expected = pd.DataFrame(
            data[f"stat_{ct}"],
            index=data[f"stat_rows_{ct}"].astype(str),
            columns=data[f"stat_cols_{ct}"].astype(str),
        )
        got = stat[ct].loc[expected.index, expected.columns]
        np.testing.assert_allclose(
            got.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            atol=1e-10,
            rtol=0,
            equal_nan=True,
        )


def test_select_marker_matches_de_fixture() -> None:
    data, metadata = _load_fixture()
    _, marker = _run_python_de(data, metadata)

    expected = pd.DataFrame(
        data["marker_matrix"],
        index=data["marker_rows"].astype(str),
        columns=data["marker_cols"].astype(str),
    )

    assert list(marker.columns.astype(str)) == list(expected.columns.astype(str))
    np.testing.assert_allclose(
        marker.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    )
