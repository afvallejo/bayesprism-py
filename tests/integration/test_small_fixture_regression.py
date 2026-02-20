from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayesprism.api import new_prism, run_prism

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "small"


def _load_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    npz = np.load(FIXTURE_DIR / "small_fixture.npz", allow_pickle=True)
    metadata = json.loads((FIXTURE_DIR / "small_fixture_metadata.json").read_text(encoding="utf-8"))
    return {k: npz[k] for k in npz.files}, metadata


def _map_from_arrays(cell_types: np.ndarray, cell_states: np.ndarray) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ct, cs in zip(cell_types.tolist(), cell_states.tolist(), strict=True):
        out.setdefault(str(ct), []).append(str(cs))
    return out


def test_new_prism_matches_small_fixture_deterministic() -> None:
    data, metadata = _load_fixture()

    reference = pd.DataFrame(
        data["reference"],
        index=data["reference_rows"].astype(str),
        columns=data["genes"].astype(str),
    )
    mixture = pd.DataFrame(
        data["mixture"],
        index=data["mixture_rows"].astype(str),
        columns=data["genes"].astype(str),
    )
    labels = metadata["labels"]

    prism = new_prism(
        reference=reference,
        input_type="count.matrix",
        cell_type_labels=[str(x) for x in labels["cell_type_label"]],
        cell_state_labels=[str(x) for x in labels["cell_state_label"]],
        key=None,
        mixture=mixture,
        outlier_cut=1,
        outlier_fraction=1,
        pseudo_min=1e-8,
    )

    np.testing.assert_allclose(
        prism.phi_cell_state.phi.to_numpy(),
        data["phi_cell_state"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        prism.phi_cell_type.phi.to_numpy(),
        data["phi_cell_type"],
        atol=1e-12,
    )

    expected_map = _map_from_arrays(data["map_cell_type"], data["map_cell_state"])
    assert prism.map == expected_map


def test_run_prism_matches_small_fixture_regression() -> None:
    data, metadata = _load_fixture()

    reference = pd.DataFrame(
        data["reference"],
        index=data["reference_rows"].astype(str),
        columns=data["genes"].astype(str),
    )
    mixture = pd.DataFrame(
        data["mixture"],
        index=data["mixture_rows"].astype(str),
        columns=data["genes"].astype(str),
    )
    labels = metadata["labels"]
    controls = metadata["controls"]

    prism = new_prism(
        reference=reference,
        input_type="count.matrix",
        cell_type_labels=[str(x) for x in labels["cell_type_label"]],
        cell_state_labels=[str(x) for x in labels["cell_state_label"]],
        key=None,
        mixture=mixture,
        outlier_cut=1,
        outlier_fraction=1,
        pseudo_min=1e-8,
    )

    bp = run_prism(
        prism=prism,
        n_cores=1,
        update_gibbs=True,
        gibbs_control={
            "chain_length": int(controls["chain_length"]),
            "burn_in": int(controls["burn_in"]),
            "thinning": int(controls["thinning"]),
            "seed": int(controls["seed"]),
            "alpha": float(controls["alpha"]),
        },
        opt_control={
            "optimizer": str(controls["optimizer"]),
            "maxit": int(controls["maxit"]),
            "n_cores": 1,
        },
    )

    np.testing.assert_allclose(
        bp.posterior_initial_cell_type.theta.to_numpy(),
        data["theta_first_type"],
        atol=5e-3,
        rtol=5e-2,
    )
    np.testing.assert_allclose(
        bp.posterior_theta_f.theta.to_numpy(),
        data["theta_final_type"],
        atol=5e-3,
        rtol=5e-2,
    )
