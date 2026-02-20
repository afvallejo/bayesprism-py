"""Embedding-learning API surface (staged for parity completion)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import BayesPrismResult, JointPost


def compute_elbo(opt_value: float, psi_env: np.ndarray, joint_post: JointPost) -> float:
    """Compute ELBO term used by the embedding module."""
    env_indices = list(range(len(joint_post.cell_types)))
    z_gk_env = joint_post.Z[:, :, env_indices].sum(axis=0)
    elbo_env = -float(np.nansum(np.log(psi_env) * z_gk_env.T))
    return float(opt_value + elbo_env - joint_post.constant)


def learn_embedding(
    bp: BayesPrismResult,
    eta_prior: np.ndarray | None = None,
    cycle: int = 50,
    gibbs_control: dict[str, Any] | None = None,
    opt_control: dict[str, Any] | None = None,
    em_res: dict[str, Any] | None = None,
    compute_elbo_value: bool | int = False,
) -> dict[str, Any]:
    """Staged placeholder for EM-based embedding learning."""
    del bp, eta_prior, cycle, gibbs_control, opt_control, em_res, compute_elbo_value
    raise NotImplementedError(
        "Embedding EM loop is staged. See docs/parity_checklist.md for current port status."
    )


def learn_embedding_nmf(
    bp: BayesPrismResult,
    k: int,
    cycle: int = 50,
    gibbs_control: dict[str, Any] | None = None,
    opt_control: dict[str, Any] | None = None,
    nmf_control: dict[str, Any] | None = None,
    em_res: dict[str, Any] | None = None,
    compute_elbo_value: bool | int = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Staged placeholder for NMF-initialized embedding learning."""
    del (
        bp,
        k,
        cycle,
        gibbs_control,
        opt_control,
        nmf_control,
        em_res,
        compute_elbo_value,
        kwargs,
    )
    raise NotImplementedError(
        "NMF-initialized embedding learning is staged. "
        "See docs/parity_checklist.md for current port status."
    )
