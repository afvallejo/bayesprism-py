"""Top-level BayesPrism API functions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from .gibbs import run_gibbs
from .models import (
    BayesPrismResult,
    BayesPrismSTResult,
    GibbsSampler,
    JointPost,
    Prism,
    RefPhi,
)
from .posterior import merge_k
from .preprocess import collapse, ensure_dataframe, filter_bulk_outlier, norm_to_one, validate_input
from .reference_update import update_reference

_GIBBS_ALIASES = {
    "chain.length": "chain_length",
    "burn.in": "burn_in",
    "n.cores": "n_cores",
}

_OPT_ALIASES = {
    "n.cores": "n_cores",
}


def _normalize_keys(control: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in control.items():
        normalized[aliases.get(key, key)] = value
    return normalized


def valid_opt_control(control: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and fill optimization control defaults."""
    ctrl = {
        "maxit": 100000,
        "maximize": False,
        "trace": 0,
        "eps": 1e-7,
        "dowarn": True,
        "tol": 0,
        "maxNA": 500,
        "n_cores": 1,
        "optimizer": "MAP",
        "sigma": 2.0,
    }
    control = {} if control is None else _normalize_keys(control, _OPT_ALIASES)
    unknown = [k for k in control if k not in ctrl]
    if unknown:
        raise ValueError(f"unknown names in opt_control: {unknown}")
    ctrl.update(control)

    if ctrl["optimizer"] not in {"MAP", "MLE"}:
        raise ValueError(f"unknown optimizer: {ctrl['optimizer']}")
    if ctrl["optimizer"] == "MAP":
        sigma = float(ctrl["sigma"])
        if sigma < 0:
            raise ValueError("sigma needs to be positive")
        ctrl["sigma"] = sigma

    return ctrl


def valid_gibbs_control(control: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and fill Gibbs control defaults."""
    ctrl = {
        "chain_length": 1000,
        "burn_in": 500,
        "thinning": 2,
        "n_cores": 1,
        "seed": 123,
        "alpha": 1.0,
    }
    control = {} if control is None else _normalize_keys(control, _GIBBS_ALIASES)
    unknown = [k for k in control if k not in ctrl]
    if unknown:
        raise ValueError(f"unknown names in gibbs_control: {unknown}")
    ctrl.update(control)

    if float(ctrl["alpha"]) < 0:
        raise ValueError("alpha needs to be positive")

    for key in ["chain_length", "burn_in", "thinning", "n_cores"]:
        ctrl[key] = int(ctrl[key])
    ctrl["alpha"] = float(ctrl["alpha"])

    return ctrl


def new_prism(
    reference: pd.DataFrame | pd.Series | np.ndarray,
    input_type: str,
    cell_type_labels: list[str],
    cell_state_labels: list[str] | None,
    key: str | None,
    mixture: pd.DataFrame | pd.Series | np.ndarray,
    outlier_cut: float = 0.01,
    outlier_fraction: float = 0.1,
    pseudo_min: float = 1e-8,
) -> Prism:
    """Construct a prism object from user-provided reference and mixture matrices."""
    if input_type not in {"count.matrix", "GEP"}:
        raise ValueError("input_type must be 'count.matrix' or 'GEP'")

    reference_df = ensure_dataframe(reference, row_prefix="reference")
    mixture_df = ensure_dataframe(mixture, row_prefix="mixture")

    if mixture_df.index.isnull().any():
        mixture_df.index = [f"mixture-{i + 1}" for i in range(mixture_df.shape[0])]

    if cell_state_labels is None:
        cell_state_labels = list(cell_type_labels)

    if len(cell_type_labels) != len(cell_state_labels):
        raise ValueError("length of cell_type_labels and cell_state_labels do not match")
    if len(cell_type_labels) != reference_df.shape[0]:
        raise ValueError("length of cell_type_labels and nrow(reference) do not match")

    cell_type_labels = [str(x) for x in cell_type_labels]
    cell_state_labels = [str(x) for x in cell_state_labels]

    validate_input(reference_df)
    validate_input(mixture_df)

    mixture_df = filter_bulk_outlier(
        mixture=mixture_df,
        outlier_cut=outlier_cut,
        outlier_fraction=outlier_fraction,
    )

    reference_df = reference_df.loc[:, reference_df.sum(axis=0) > 0]

    gene_shared = [gene for gene in reference_df.columns if gene in set(mixture_df.columns)]
    if len(gene_shared) == 0:
        raise ValueError("gene names of reference and mixture do not match")
    if len(gene_shared) < 100:
        warnings.warn(
            "very few genes from reference and mixture match; please verify gene names",
            RuntimeWarning,
            stacklevel=2,
        )

    ref_cs = collapse(ref=reference_df, labels=cell_state_labels)
    ref_ct = collapse(ref=reference_df, labels=cell_type_labels)

    ref_cs = ref_cs.loc[:, gene_shared]
    ref_ct = ref_ct.loc[:, gene_shared]
    mixture_df = mixture_df.loc[:, gene_shared]

    ref_cs = norm_to_one(ref=ref_cs, pseudo_min=pseudo_min)
    ref_ct = norm_to_one(ref=ref_ct, pseudo_min=pseudo_min)

    cell_map: dict[str, list[str]] = {}
    for cell_type in ref_ct.index.astype(str).tolist():
        states = [
            state
            for i, state in enumerate(cell_state_labels)
            if cell_type_labels[i] == cell_type and state not in cell_map.get(cell_type, [])
        ]
        cell_map[cell_type] = states

    return Prism(
        phi_cell_state=RefPhi(phi=ref_cs, pseudo_min=pseudo_min),
        phi_cell_type=RefPhi(phi=ref_ct, pseudo_min=pseudo_min),
        map=cell_map,
        key=key,
        mixture=mixture_df,
    )


def run_prism(
    prism: Prism,
    n_cores: int = 1,
    update_gibbs: bool = True,
    gibbs_control: dict[str, Any] | None = None,
    opt_control: dict[str, Any] | None = None,
) -> BayesPrismResult:
    """Main deconvolution workflow."""
    gibbs_control = dict(gibbs_control or {})
    opt_control = dict(opt_control or {})

    gibbs_control.setdefault("n_cores", n_cores)
    opt_control.setdefault("n_cores", n_cores)

    opt_control = valid_opt_control(opt_control)
    gibbs_control = valid_gibbs_control(gibbs_control)

    if prism.phi_cell_state.pseudo_min == 0:
        gibbs_control["alpha"] = max(1.0, float(gibbs_control["alpha"]))

    gibbs_sampler_ini_cs = GibbsSampler(
        reference=prism.phi_cell_state,
        X=prism.mixture,
        gibbs_control=gibbs_control,
    )
    joint_post_ini_cs = run_gibbs(gibbs_sampler_ini_cs, final=False)
    if not isinstance(joint_post_ini_cs, JointPost):
        raise RuntimeError("Initial Gibbs over RefPhi must return JointPost")

    joint_post_ini_ct = merge_k(joint_post_obj=joint_post_ini_cs, map=prism.map)

    if not update_gibbs:
        return BayesPrismResult(
            prism=prism,
            posterior_initial_cell_state=joint_post_ini_cs,
            posterior_initial_cell_type=joint_post_ini_ct,
            control_param={
                "gibbs_control": gibbs_control,
                "opt_control": opt_control,
                "update_gibbs": update_gibbs,
            },
        )

    psi = update_reference(
        Z=joint_post_ini_ct.Z,
        phi_prime=prism.phi_cell_type,
        map=prism.map,
        key=prism.key,
        opt_control=opt_control,
    )

    gibbs_sampler_update = GibbsSampler(
        reference=psi,
        X=prism.mixture,
        gibbs_control=gibbs_control,
    )
    theta_f = run_gibbs(gibbs_sampler_update, final=True)

    return BayesPrismResult(
        prism=prism,
        posterior_initial_cell_state=joint_post_ini_cs,
        posterior_initial_cell_type=joint_post_ini_ct,
        reference_update=psi,
        posterior_theta_f=theta_f,
        control_param={
            "gibbs_control": gibbs_control,
            "opt_control": opt_control,
            "update_gibbs": update_gibbs,
        },
    )


def update_theta(
    bp: BayesPrismResult,
    gibbs_control: dict[str, Any] | None = None,
    opt_control: dict[str, Any] | None = None,
) -> BayesPrismResult:
    """Run update Gibbs sampling using existing BayesPrism initial posterior."""
    if bp.posterior_initial_cell_type is None:
        raise ValueError("bp must contain initial cell type posterior")

    gibbs_control_bp = dict(bp.control_param.get("gibbs_control", {}))
    opt_control_bp = dict(bp.control_param.get("opt_control", {}))

    gibbs_control_bp.update(_normalize_keys(gibbs_control or {}, _GIBBS_ALIASES))
    opt_control_bp.update(_normalize_keys(opt_control or {}, _OPT_ALIASES))

    gibbs_control_valid = valid_gibbs_control(gibbs_control_bp)
    opt_control_valid = valid_opt_control(opt_control_bp)

    psi = update_reference(
        Z=bp.posterior_initial_cell_type.Z,
        phi_prime=bp.prism.phi_cell_type,
        map=bp.prism.map,
        key=bp.prism.key,
        opt_control=opt_control_valid,
    )
    gibbs_sampler_update = GibbsSampler(
        reference=psi,
        X=bp.prism.mixture,
        gibbs_control=gibbs_control_valid,
    )
    theta_f = run_gibbs(gibbs_sampler_update, final=True)

    return BayesPrismResult(
        prism=bp.prism,
        posterior_initial_cell_state=bp.posterior_initial_cell_state,
        posterior_initial_cell_type=bp.posterior_initial_cell_type,
        reference_update=psi,
        posterior_theta_f=theta_f,
        control_param={
            "gibbs_control": gibbs_control_valid,
            "opt_control": opt_control_valid,
            "update_gibbs": True,
        },
    )


def run_prism_st(
    prism: Prism,
    n_cores: int = 1,
    gibbs_control: dict[str, Any] | None = None,
    opt_control: dict[str, Any] | None = None,
) -> BayesPrismSTResult:
    """Spatial-oriented workflow mirroring run.prism.st semantics."""
    gibbs_control = dict(gibbs_control or {})
    opt_control = dict(opt_control or {"optimizer": "MLE"})

    gibbs_control.setdefault("n_cores", n_cores)
    opt_control.setdefault("n_cores", n_cores)

    gibbs_control_valid = valid_gibbs_control(gibbs_control)
    opt_control_valid = valid_opt_control(opt_control)

    gibbs_sampler_ini = GibbsSampler(
        reference=prism.phi_cell_state,
        X=prism.mixture,
        gibbs_control=gibbs_control_valid,
    )
    joint_post_ini = run_gibbs(gibbs_sampler_ini, final=False)
    if not isinstance(joint_post_ini, JointPost):
        raise RuntimeError("Initial Gibbs over RefPhi must return JointPost")

    psi = update_reference(
        Z=joint_post_ini.Z,
        phi_prime=prism.phi_cell_state,
        map=prism.map,
        key=prism.key,
        opt_control=opt_control_valid,
    )

    gibbs_sampler_update = GibbsSampler(
        reference=psi,
        X=prism.mixture,
        gibbs_control=gibbs_control_valid,
    )
    joint_post_update = run_gibbs(gibbs_sampler_update, final=False)
    if not isinstance(joint_post_update, JointPost):
        raise NotImplementedError(
            "run_prism_st tumor mode returns ThetaPost in upstream; "
            "staged handling in parity checklist"
        )

    joint_post_update_ct = merge_k(joint_post_obj=joint_post_update, map=prism.map)

    return BayesPrismSTResult(
        prism=prism,
        posterior_cell_state=joint_post_update,
        posterior_cell_type=joint_post_update_ct,
        reference_update=psi,
        control_param={"gibbs_control": gibbs_control_valid, "opt_control": opt_control_valid},
    )


def get_fraction(
    bp: BayesPrismResult,
    which_theta: str,
    state_or_type: str,
) -> pd.DataFrame:
    """Extract posterior mean cell fractions."""
    if which_theta not in {"first", "final"}:
        raise ValueError("which_theta must be 'first' or 'final'")
    if state_or_type not in {"state", "type"}:
        raise ValueError("state_or_type must be 'state' or 'type'")

    if which_theta == "first" and state_or_type == "state":
        return bp.posterior_initial_cell_state.theta
    if which_theta == "first" and state_or_type == "type":
        return bp.posterior_initial_cell_type.theta

    if bp.posterior_theta_f is None:
        raise ValueError("final theta is not available")
    if state_or_type == "state":
        warnings.warn(
            "only cell type is available for updated Gibbs; returning cell type info",
            RuntimeWarning,
            stacklevel=2,
        )
    return bp.posterior_theta_f.theta


def get_exp(bp: BayesPrismResult, state_or_type: str, cell_name: str) -> pd.DataFrame:
    """Extract posterior mean sample-specific expression for one state/type."""
    if state_or_type not in {"state", "type"}:
        raise ValueError("state_or_type must be 'state' or 'type'")

    post = (
        bp.posterior_initial_cell_state
        if state_or_type == "state"
        else bp.posterior_initial_cell_type
    )
    if cell_name not in post.cell_types:
        raise KeyError(f"unknown cell name: {cell_name}")
    cell_idx = post.cell_types.index(cell_name)
    return pd.DataFrame(post.Z[:, :, cell_idx], index=post.bulk_ids, columns=post.gene_ids)
