from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bayesprism.api import new_prism, run_prism  # noqa: E402


def _build_inputs(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(seed)

    genes = [f"g{i}" for i in range(1, 11)]
    cell_states = ["A_s1", "A_s2", "B_s1", "B_s2", "C_s1", "C_s2"]
    cell_types = ["A", "A", "B", "B", "C", "C"]

    reference_base = np.array(
        [
            [40, 32, 18, 6, 5, 3, 3, 2, 1, 1],
            [25, 38, 21, 7, 5, 3, 2, 2, 1, 1],
            [3, 4, 8, 40, 33, 20, 9, 3, 1, 1],
            [3, 4, 8, 23, 38, 25, 11, 4, 1, 1],
            [2, 2, 3, 7, 10, 20, 35, 25, 7, 4],
            [2, 2, 3, 6, 9, 16, 26, 37, 11, 6],
        ],
        dtype=float,
    )
    reference_counts = reference_base + rng.poisson(lam=2.0, size=reference_base.shape)

    reference = pd.DataFrame(reference_counts, index=cell_states, columns=genes)

    weights = np.array(
        [
            [0.50, 0.10, 0.15, 0.05, 0.15, 0.05],
            [0.10, 0.45, 0.20, 0.10, 0.10, 0.05],
            [0.05, 0.05, 0.45, 0.20, 0.15, 0.10],
            [0.08, 0.07, 0.15, 0.20, 0.20, 0.30],
        ],
        dtype=float,
    )
    library = np.array([1800, 1700, 1900, 2000], dtype=float)

    phi = reference.to_numpy(dtype=float)
    phi = phi / phi.sum(axis=1, keepdims=True)
    expected = weights @ phi
    mixture_counts = rng.poisson(lam=expected * library[:, None]).astype(float)
    mixture = pd.DataFrame(mixture_counts, index=[f"s{i}" for i in range(1, 5)], columns=genes)

    return reference, mixture, cell_types, cell_states


def main() -> None:
    seed = 20260220
    out_dir = ROOT / "tests" / "data" / "fixtures" / "small"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference, mixture, cell_type_labels, cell_state_labels = _build_inputs(seed=seed)

    controls = {
        "chain_length": 80,
        "burn_in": 20,
        "thinning": 4,
        "seed": 77,
        "alpha": 1.0,
        "optimizer": "MAP",
        "maxit": 200,
    }

    prism = new_prism(
        reference=reference,
        input_type="count.matrix",
        cell_type_labels=cell_type_labels,
        cell_state_labels=cell_state_labels,
        key=None,
        mixture=mixture,
        outlier_cut=1.0,
        outlier_fraction=1.0,
        pseudo_min=1e-8,
    )

    bp = run_prism(
        prism=prism,
        n_cores=1,
        update_gibbs=True,
        gibbs_control={
            "chain_length": controls["chain_length"],
            "burn_in": controls["burn_in"],
            "thinning": controls["thinning"],
            "seed": controls["seed"],
            "alpha": controls["alpha"],
        },
        opt_control={
            "optimizer": controls["optimizer"],
            "maxit": controls["maxit"],
            "n_cores": 1,
        },
    )

    map_cell_type: list[str] = []
    map_cell_state: list[str] = []
    for ct, states in prism.map.items():
        map_cell_type.extend([ct] * len(states))
        map_cell_state.extend(states)

    np.savez(
        out_dir / "small_fixture.npz",
        reference=reference.to_numpy(dtype=float),
        reference_rows=reference.index.to_numpy(dtype=str),
        mixture=mixture.to_numpy(dtype=float),
        mixture_rows=mixture.index.to_numpy(dtype=str),
        genes=reference.columns.to_numpy(dtype=str),
        phi_cell_state=prism.phi_cell_state.phi.to_numpy(dtype=float),
        phi_cell_type=prism.phi_cell_type.phi.to_numpy(dtype=float),
        map_cell_type=np.asarray(map_cell_type, dtype=str),
        map_cell_state=np.asarray(map_cell_state, dtype=str),
        theta_first_type=bp.posterior_initial_cell_type.theta.to_numpy(dtype=float),
        theta_final_type=bp.posterior_theta_f.theta.to_numpy(dtype=float),
    )

    metadata = {
        "seed": seed,
        "labels": {
            "cell_type_label": cell_type_labels,
            "cell_state_label": cell_state_labels,
        },
        "controls": controls,
    }
    (out_dir / "small_fixture_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote fixture files to {out_dir}")


if __name__ == "__main__":
    main()
