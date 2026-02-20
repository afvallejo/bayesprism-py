"""Input preprocessing and normalization helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def ensure_dataframe(
    matrix: pd.DataFrame | pd.Series | np.ndarray | sparse.spmatrix,
    row_prefix: str,
) -> pd.DataFrame:
    """Convert array-like inputs to a labeled DataFrame."""
    if isinstance(matrix, pd.DataFrame):
        return matrix.copy()
    if isinstance(matrix, pd.Series):
        if matrix.name is None:
            matrix.name = f"{row_prefix}-1"
        return pd.DataFrame(
            [matrix.to_numpy(dtype=float)],
            index=[matrix.name],
            columns=matrix.index,
        )
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    if isinstance(matrix, np.ndarray):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("input matrix must be 1D or 2D")
        n_rows, n_cols = matrix.shape
        row_names = [f"{row_prefix}-{i + 1}" for i in range(n_rows)]
        col_names = [f"gene-{j + 1}" for j in range(n_cols)]
        warnings.warn(
            "Input converted from ndarray; generated synthetic gene names.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(matrix, index=row_names, columns=col_names)
    raise TypeError("input must be a pandas DataFrame/Series, ndarray, or sparse matrix")


def validate_input(input_matrix: pd.DataFrame | sparse.spmatrix) -> None:
    """Validate numeric constraints and matrix type for BayesPrism inputs."""
    if not isinstance(input_matrix, pd.DataFrame) and not sparse.issparse(input_matrix):
        raise TypeError("the type of mixture and reference needs to be matrix-like")

    if sparse.issparse(input_matrix):
        values = input_matrix.toarray().astype(float)
        columns = None
    else:
        values = input_matrix.to_numpy(dtype=float)
        columns = input_matrix.columns

    if values.size == 0:
        raise ValueError("input is empty")

    v_max = float(np.max(values))
    if v_max <= 1:
        warnings.warn("input seems normalized", RuntimeWarning, stacklevel=2)
    elif v_max < 20:
        warnings.warn(
            "input may be log-transformed; this should be avoided",
            RuntimeWarning,
            stacklevel=2,
        )

    if float(np.min(values)) < 0:
        raise ValueError("input contains negative values")
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("input contains NaN or non-finite values")

    if columns is None or len(columns) == 0:
        raise ValueError("please specify gene identifiers as column names")
    if pd.Index(columns).isnull().any():
        raise ValueError("gene identifiers contain null values")


def norm_to_one(ref: pd.DataFrame, pseudo_min: float) -> pd.DataFrame:
    """Row-normalize reference with optional pseudo-min treatment for zero counts."""
    if pseudo_min < 0:
        raise ValueError("pseudo_min must be non-negative")
    if ref.shape[1] == 0:
        raise ValueError("reference has zero genes")

    ref_values = ref.to_numpy(dtype=float)
    row_sums = ref_values.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("reference rows must have positive sums")

    g = ref.shape[1]
    phi = ref.div(row_sums, axis=0) * (1 - pseudo_min * g) + pseudo_min

    min_value = ref.min(axis=1)
    which_row = min_value > 0
    if which_row.any():
        phi.loc[which_row] = ref.loc[which_row].div(ref.loc[which_row].sum(axis=1), axis=0)

    return phi


def collapse(ref: pd.DataFrame, labels: Sequence[str | None]) -> pd.DataFrame:
    """Collapse rows of a matrix by labels while preserving first-observed label order."""
    if ref.shape[0] != len(labels):
        raise ValueError("nrow(ref) must equal length(labels)")

    labels_series = pd.Series(labels)
    non_na_idx = ~labels_series.isna()
    labels_series = labels_series[non_na_idx].astype(str)
    ref_non_na = ref.loc[non_na_idx.to_numpy()]

    unique_labels = list(pd.unique(labels_series))
    collapsed_rows = []
    for label in unique_labels:
        mask = labels_series == label
        collapsed_rows.append(ref_non_na.loc[mask.to_numpy()].sum(axis=0))

    collapsed = pd.DataFrame(collapsed_rows, index=unique_labels)
    collapsed.columns = ref.columns
    return collapsed


def filter_bulk_outlier(
    mixture: pd.DataFrame,
    outlier_cut: float,
    outlier_fraction: float,
) -> pd.DataFrame:
    """Filter outlier genes in bulk mixture profiles."""
    if mixture.shape[0] == 0:
        raise ValueError("mixture has zero samples")

    row_sums = mixture.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("mixture rows must have positive sums")

    mixture_norm = mixture.div(row_sums, axis=0)
    outlier_idx = (
        (mixture_norm > outlier_cut).sum(axis=0) / mixture_norm.shape[0]
    ) > outlier_fraction
    return mixture.loc[:, ~outlier_idx]
