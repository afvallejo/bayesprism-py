#!/usr/bin/env python3
"""Convert R-generated CSV golden files into a compact NPZ + JSON bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing CSV fixtures",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ file path")
    parser.add_argument("--metadata", type=Path, required=True, help="Output metadata JSON path")
    args = parser.parse_args()

    fixture_dir = args.input

    reference = _read_matrix(fixture_dir / "reference_input.csv")
    mixture = _read_matrix(fixture_dir / "mixture_input.csv")
    phi_cell_state = _read_matrix(fixture_dir / "phi_cell_state.csv")
    phi_cell_type = _read_matrix(fixture_dir / "phi_cell_type.csv")
    theta_first_state = _read_matrix(fixture_dir / "theta_first_state.csv")
    theta_first_type = _read_matrix(fixture_dir / "theta_first_type.csv")
    theta_final_type = _read_matrix(fixture_dir / "theta_final_type.csv")
    phi_update = _read_matrix(fixture_dir / "phi_update.csv")
    transform_phi_t = _read_matrix(fixture_dir / "transform_phi_t.csv")
    sample_one_z = _read_matrix(fixture_dir / "sample_one_Z.csv")
    sample_one_theta = _read_matrix(fixture_dir / "sample_one_theta.csv")

    map_df = pd.read_csv(fixture_dir / "map.csv")
    labels_df = pd.read_csv(fixture_dir / "labels.csv")
    controls_df = pd.read_csv(fixture_dir / "controls.csv")

    np.savez_compressed(
        args.output,
        reference=reference.to_numpy(dtype=float),
        mixture=mixture.to_numpy(dtype=float),
        phi_cell_state=phi_cell_state.to_numpy(dtype=float),
        phi_cell_type=phi_cell_type.to_numpy(dtype=float),
        theta_first_state=theta_first_state.to_numpy(dtype=float),
        theta_first_type=theta_first_type.to_numpy(dtype=float),
        theta_final_type=theta_final_type.to_numpy(dtype=float),
        phi_update=phi_update.to_numpy(dtype=float),
        transform_phi_t=transform_phi_t.to_numpy(dtype=float),
        sample_one_z=sample_one_z.to_numpy(dtype=float),
        sample_one_theta=sample_one_theta.to_numpy(dtype=float),
        reference_rows=np.array(reference.index.astype(str)),
        mixture_rows=np.array(mixture.index.astype(str)),
        genes=np.array(reference.columns.astype(str)),
        cell_states=np.array(phi_cell_state.index.astype(str)),
        cell_types=np.array(phi_cell_type.index.astype(str)),
        map_cell_type=np.array(map_df["cell_type"].astype(str)),
        map_cell_state=np.array(map_df["cell_state"].astype(str)),
    )

    metadata = {
        "labels": labels_df.to_dict(orient="list"),
        "controls": controls_df.iloc[0].to_dict(),
        "files": sorted(path.name for path in fixture_dir.glob("*.csv")),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
