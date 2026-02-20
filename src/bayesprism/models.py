"""Core data models mirroring BayesPrism S4 class semantics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when model constraints are violated."""


def _validate_numeric_frame(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas.DataFrame")
    if df.columns.isnull().any():
        raise ValidationError(f"{name} contains null gene names")
    values = df.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValidationError(f"{name} contains NaN or non-finite values")


@dataclass(slots=True)
class RefPhi:
    """Reference expression profile for cell state/type (rows) by genes (columns)."""

    phi: pd.DataFrame
    pseudo_min: float | None = None

    def __post_init__(self) -> None:
        _validate_numeric_frame(self.phi, "phi")
        phi_values = self.phi.to_numpy(dtype=float)
        if (phi_values < 0).any():
            raise ValidationError("reference contains negative values")
        if self.pseudo_min is not None:
            if not np.isscalar(self.pseudo_min):
                raise ValidationError("invalid pseudo_min")
            phi_min = float(np.min(phi_values))
            if not np.isclose(phi_min, float(self.pseudo_min), atol=0.0, rtol=0.0):
                warnings.warn("pseudo_min does not match min(phi)", RuntimeWarning, stacklevel=2)


@dataclass(slots=True)
class RefTumor:
    """Updated reference profiles for malignant and non-malignant cells."""

    psi_mal: pd.DataFrame
    psi_env: pd.DataFrame
    key: str
    pseudo_min: float | None = None

    def __post_init__(self) -> None:
        _validate_numeric_frame(self.psi_mal, "psi_mal")
        _validate_numeric_frame(self.psi_env, "psi_env")
        if not isinstance(self.key, str) or not self.key:
            raise ValidationError("invalid key")
        if list(self.psi_mal.columns) != list(self.psi_env.columns):
            raise ValidationError("gene names of psi_mal and psi_env do not match")
        values = np.concatenate(
            [self.psi_mal.to_numpy(dtype=float).ravel(), self.psi_env.to_numpy(dtype=float).ravel()]
        )
        if (values < 0).any():
            raise ValidationError("reference contains negative values")
        if self.pseudo_min is not None:
            if not np.isscalar(self.pseudo_min):
                raise ValidationError("invalid pseudo_min")
            phi_min = float(np.min(values))
            if not np.isclose(phi_min, float(self.pseudo_min), atol=0.0, rtol=0.0):
                warnings.warn("pseudo_min does not match min(phi)", RuntimeWarning, stacklevel=2)


@dataclass(slots=True)
class Prism:
    """Input bundle for deconvolution workflows."""

    phi_cell_state: RefPhi
    phi_cell_type: RefPhi
    map: dict[str, list[str]]
    key: str | None
    mixture: pd.DataFrame

    def __post_init__(self) -> None:
        _validate_numeric_frame(self.mixture, "mixture")
        if self.key is not None and self.key not in self.map:
            raise ValidationError("invalid key")

        cs_genes = list(self.phi_cell_state.phi.columns)
        ct_genes = list(self.phi_cell_type.phi.columns)
        bk_genes = list(self.mixture.columns)
        if cs_genes != ct_genes or cs_genes != bk_genes:
            raise ValidationError("gene names do not match")

        if list(self.phi_cell_type.phi.index) != list(self.map.keys()):
            raise ValidationError("cell types between map and phi_cell_type do not match")

        map_states = [state for states in self.map.values() for state in states]
        if set(self.phi_cell_state.phi.index) != set(map_states):
            raise ValidationError("cell states between map and phi_cell_state do not match")


@dataclass(slots=True)
class ThetaPost:
    """Posterior summaries for theta."""

    theta: pd.DataFrame
    theta_cv: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        _validate_numeric_frame(self.theta, "theta")
        if not self.theta_cv.empty:
            _validate_numeric_frame(self.theta_cv, "theta_cv")
            has_same_index = list(self.theta.index) == list(self.theta_cv.index)
            has_same_columns = list(self.theta.columns) == list(self.theta_cv.columns)
            if not has_same_index or not has_same_columns:
                raise ValidationError("dimnames of theta and theta_cv do not match")


@dataclass(slots=True)
class JointPost:
    """Joint posterior summaries for Z and theta."""

    Z: np.ndarray
    bulk_ids: list[str]
    gene_ids: list[str]
    cell_types: list[str]
    theta: pd.DataFrame
    theta_cv: pd.DataFrame = field(default_factory=pd.DataFrame)
    constant: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.Z, np.ndarray) or self.Z.ndim != 3:
            raise ValidationError("Z must be a 3D numpy array")
        n, g, k = self.Z.shape
        if len(self.bulk_ids) != n or len(self.gene_ids) != g or len(self.cell_types) != k:
            raise ValidationError("Z dimensions do not match provided labels")
        _validate_numeric_frame(self.theta, "theta")
        if list(self.theta.index) != self.bulk_ids:
            raise ValidationError("sample IDs of Z and theta do not match")
        if list(self.theta.columns) != self.cell_types:
            raise ValidationError("cell types of Z and theta do not match")
        if not self.theta_cv.empty:
            _validate_numeric_frame(self.theta_cv, "theta_cv")
            has_same_index = list(self.theta_cv.index) == self.bulk_ids
            has_same_columns = list(self.theta_cv.columns) == self.cell_types
            if not has_same_index or not has_same_columns:
                raise ValidationError("theta_cv dimnames do not match")


@dataclass(slots=True)
class GibbsSampler:
    """Input structure for Gibbs routines."""

    reference: RefPhi | RefTumor
    X: pd.DataFrame
    gibbs_control: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_numeric_frame(self.X, "X")
        if isinstance(self.reference, RefPhi):
            ref_genes = list(self.reference.phi.columns)
        else:
            ref_genes = list(self.reference.psi_mal.columns)
        if ref_genes != list(self.X.columns):
            raise ValidationError("gene names do not match between reference and X")


@dataclass(slots=True)
class BayesPrismResult:
    """Output object for run_prism."""

    prism: Prism
    posterior_initial_cell_state: JointPost
    posterior_initial_cell_type: JointPost
    reference_update: RefPhi | RefTumor | None = None
    posterior_theta_f: ThetaPost | None = None
    control_param: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BayesPrismSTResult:
    """Output object for run_prism_st."""

    prism: Prism
    posterior_cell_state: JointPost
    posterior_cell_type: JointPost
    reference_update: RefPhi | RefTumor | None = None
    control_param: dict[str, Any] = field(default_factory=dict)
