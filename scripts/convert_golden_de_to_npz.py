#!/usr/bin/env python3
"""Convert DE golden CSV files to NPZ + JSON metadata."""

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

    sc_dat = _read_matrix(fixture_dir / "sc_dat.csv")
    labels = pd.read_csv(fixture_dir / "labels.csv")
    params = pd.read_csv(fixture_dir / "params.csv")
    marker_matrix = _read_matrix(fixture_dir / "marker_matrix.csv")
    marker_genes = pd.read_csv(fixture_dir / "marker_genes.csv")

    stat_files = sorted(fixture_dir.glob("stat_*.csv"))
    stat_names = [path.stem.replace("stat_", "", 1) for path in stat_files]
    stat_frames = {
        name: _read_matrix(path)
        for name, path in zip(stat_names, stat_files, strict=True)
    }

    payload: dict[str, np.ndarray] = {
        "sc_dat": sc_dat.to_numpy(dtype=float),
        "sc_dat_rows": np.array(sc_dat.index.astype(str)),
        "sc_dat_cols": np.array(sc_dat.columns.astype(str)),
        "labels_cell_id": np.array(labels["cell_id"].astype(str)),
        "labels_cell_type": np.array(labels["cell_type"].astype(str)),
        "labels_cell_state": np.array(labels["cell_state"].astype(str)),
        "marker_matrix": marker_matrix.to_numpy(dtype=float),
        "marker_rows": np.array(marker_matrix.index.astype(str)),
        "marker_cols": np.array(marker_matrix.columns.astype(str)),
        "marker_genes": np.array(marker_genes["gene"].astype(str)),
        "ct_names": np.array(stat_names, dtype=str),
    }

    for ct_name, frame in stat_frames.items():
        payload[f"stat_{ct_name}"] = frame.to_numpy(dtype=float)
        payload[f"stat_rows_{ct_name}"] = np.array(frame.index.astype(str))
        payload[f"stat_cols_{ct_name}"] = np.array(frame.columns.astype(str))

    np.savez_compressed(args.output, **payload)

    metadata = {
        "params": params.iloc[0].to_dict(),
        "stat_names": stat_names,
        "files": sorted(path.name for path in fixture_dir.glob("*.csv")),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
