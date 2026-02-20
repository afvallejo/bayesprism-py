from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bayesprism.qc import get_exp_stat, select_marker  # noqa: E402


def _build_de_inputs(seed: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    genes = [f"g{i}" for i in range(1, 9)]

    labels_state = ["A_s1"] * 4 + ["A_s2"] * 4 + ["B_s1"] * 4 + ["B_s2"] * 4 + ["C_low"] * 2
    labels_type = ["A"] * 8 + ["B"] * 8 + ["C"] * 2

    means = {
        "A_s1": np.array([30, 24, 18, 4, 2, 2, 1, 1], dtype=float),
        "A_s2": np.array([18, 31, 20, 4, 2, 2, 1, 1], dtype=float),
        "B_s1": np.array([2, 3, 6, 29, 22, 11, 6, 2], dtype=float),
        "B_s2": np.array([2, 3, 5, 18, 29, 18, 8, 3], dtype=float),
        "C_low": np.array([5, 5, 5, 5, 5, 5, 20, 20], dtype=float),
    }

    rows = [rng.poisson(lam=means[state]).astype(float) for state in labels_state]
    index = [f"cell_{i}" for i in range(1, len(rows) + 1)]
    sc_dat = pd.DataFrame(rows, index=index, columns=genes)
    return sc_dat, labels_type, labels_state


def main() -> None:
    seed = 20260221
    out_dir = ROOT / "tests" / "data" / "fixtures" / "de_small"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc_dat, cell_type_labels, cell_state_labels = _build_de_inputs(seed=seed)

    params = {
        "pseudo_count": 0.1,
        "cell_count_cutoff": 3,
        "pval_max": 0.05,
        "lfc_min": 0.2,
    }

    stat = get_exp_stat(
        sc_dat=sc_dat,
        cell_type_labels=cell_type_labels,
        cell_state_labels=cell_state_labels,
        pseudo_count=params["pseudo_count"],
        cell_count_cutoff=params["cell_count_cutoff"],
        n_cores=2,
    )
    marker = select_marker(
        sc_dat=sc_dat,
        stat=stat,
        pval_max=params["pval_max"],
        lfc_min=params["lfc_min"],
    )

    npz_payload: dict[str, np.ndarray] = {
        "sc_dat": sc_dat.to_numpy(dtype=float),
        "sc_dat_rows": sc_dat.index.to_numpy(dtype=str),
        "sc_dat_cols": sc_dat.columns.to_numpy(dtype=str),
        "labels_cell_type": np.asarray(cell_type_labels, dtype=str),
        "labels_cell_state": np.asarray(cell_state_labels, dtype=str),
        "ct_names": np.asarray(list(stat.keys()), dtype=str),
        "marker_matrix": marker.to_numpy(dtype=float),
        "marker_rows": marker.index.to_numpy(dtype=str),
        "marker_cols": marker.columns.to_numpy(dtype=str),
    }

    for ct, table in stat.items():
        npz_payload[f"stat_{ct}"] = table.to_numpy(dtype=float)
        npz_payload[f"stat_rows_{ct}"] = table.index.to_numpy(dtype=str)
        npz_payload[f"stat_cols_{ct}"] = table.columns.to_numpy(dtype=str)

    np.savez(out_dir / "de_fixture.npz", **npz_payload)

    metadata = {"seed": seed, "params": params}
    (out_dir / "de_fixture_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote fixture files to {out_dir}")


if __name__ == "__main__":
    main()
