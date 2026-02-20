"""Reference update and optimization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp as scipy_logsumexp

from .models import RefPhi, RefTumor
from .preprocess import norm_to_one


def logsumexp(x: np.ndarray) -> float:
    """Stable log-sum-exp."""
    return float(scipy_logsumexp(np.asarray(x, dtype=float)))


def transform_phi_t(phi_t: np.ndarray, gamma_t: np.ndarray) -> np.ndarray:
    """Transform one reference row with log fold-change vector."""
    gamma_t = np.asarray(gamma_t, dtype=float)
    phi_t = np.asarray(phi_t, dtype=float)
    gamma_stab = gamma_t - np.max(gamma_t)
    psi_t = phi_t * np.exp(gamma_stab)
    return psi_t / np.sum(psi_t)


def transform_phi(phi: pd.DataFrame | np.ndarray, gamma: np.ndarray) -> pd.DataFrame:
    """Transform full reference matrix with per-row log fold-change matrix."""
    if isinstance(phi, pd.DataFrame):
        phi_values = phi.to_numpy(dtype=float)
        index = phi.index
        columns = phi.columns
    else:
        phi_values = np.asarray(phi, dtype=float)
        index = pd.RangeIndex(phi_values.shape[0])
        columns = pd.RangeIndex(phi_values.shape[1])

    psi = np.zeros_like(phi_values)
    for t in range(phi_values.shape[0]):
        psi[t, :] = transform_phi_t(phi_values[t, :], gamma[t, :])

    return pd.DataFrame(psi, index=index, columns=columns)


def log_posterior_gamma(
    gamma_t: np.ndarray,
    phi_t: np.ndarray,
    phi_t_log: np.ndarray,
    z_gt_t: np.ndarray,
    z_t_t: float,
    prior_num: float,
) -> float:
    """Negative log posterior for MAP optimization of one cell type."""
    x = phi_t_log + gamma_t
    psi_t_log = x - scipy_logsumexp(x)
    log_likelihood = float(np.sum(z_gt_t * psi_t_log))
    log_prior = float(np.sum(prior_num * gamma_t**2))
    return -(log_likelihood + log_prior)


def log_posterior_gamma_grad(
    gamma_t: np.ndarray,
    phi_t: np.ndarray,
    phi_t_log: np.ndarray,
    z_gt_t: np.ndarray,
    z_t_t: float,
    prior_num: float,
) -> np.ndarray:
    """Gradient of negative log posterior for MAP optimization."""
    del phi_t_log
    psi_t = transform_phi_t(phi_t=phi_t, gamma_t=gamma_t)
    log_likelihood_grad = z_gt_t - (z_t_t * psi_t)
    log_prior_grad = 2 * prior_num * gamma_t
    return -(log_likelihood_grad + log_prior_grad)


def transform_phi_transpose(phi_transpose: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Apply one shared gamma vector using transposed shape semantics."""
    # phi_transpose shape is G x T. Output is T x G.
    n_genes, n_types = phi_transpose.shape
    psi = np.zeros((n_types, n_genes), dtype=float)
    for t in range(n_types):
        psi[t, :] = transform_phi_t(phi_transpose[:, t], gamma)
    return psi


def log_mle_gamma(
    gamma: np.ndarray,
    phi_transpose: np.ndarray,
    phi_log_transpose: np.ndarray,
    z_tg: np.ndarray,
    z_t: np.ndarray,
) -> float:
    """Negative log likelihood for shared-gamma MLE optimizer."""
    del phi_transpose, z_t
    x = phi_log_transpose + gamma[:, None]
    psi_log = x.T - scipy_logsumexp(x, axis=0)[:, None]
    log_likelihood = float(np.sum(z_tg * psi_log))
    return -log_likelihood


def log_mle_gamma_grad(
    gamma: np.ndarray,
    phi_transpose: np.ndarray,
    phi_log_transpose: np.ndarray,
    z_tg: np.ndarray,
    z_t: np.ndarray,
) -> np.ndarray:
    """Gradient of negative log likelihood for shared-gamma MLE optimizer."""
    del phi_log_transpose
    psi = transform_phi_transpose(phi_transpose=phi_transpose, gamma=gamma)
    log_likelihood_grad = np.sum(z_tg - (z_t[:, None] * psi), axis=0)
    return -log_likelihood_grad


def optimize_psi(
    phi: pd.DataFrame,
    z_gt: np.ndarray,
    prior_num: float,
    opt_control: dict[str, Any],
) -> dict[str, Any]:
    """MAP update over per-cell-type gamma vectors."""
    phi_values = phi.to_numpy(dtype=float)
    z_t = np.sum(z_gt, axis=0)

    maxiter = int(opt_control.get("maxit", 100000))
    opt_gamma = np.zeros_like(phi_values)
    value = 0.0

    for t in range(phi_values.shape[0]):
        res = minimize(
            fun=log_posterior_gamma,
            x0=np.zeros(phi_values.shape[1], dtype=float),
            jac=log_posterior_gamma_grad,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            args=(phi_values[t, :], np.log(phi_values[t, :]), z_gt[:, t], float(z_t[t]), prior_num),
        )
        opt_gamma[t, :] = res.x
        value += float(res.fun)

    opt_gamma[np.max(np.abs(opt_gamma), axis=1) > 20, :] = 0.0

    psi = transform_phi(phi, opt_gamma)
    return {"psi": psi, "value": value, "gamma": opt_gamma}


def optimize_psi_one_gamma(
    phi: pd.DataFrame,
    z_gt: np.ndarray,
    opt_control: dict[str, Any],
) -> dict[str, Any]:
    """MLE update using a single gamma shared across cell types."""
    phi_values = phi.to_numpy(dtype=float)
    phi_t = phi_values.T
    phi_log_t = np.log(phi_values).T

    z_tg = z_gt.T
    z_t = np.sum(z_gt, axis=0)

    maxiter = int(opt_control.get("maxit", 100000))

    res = minimize(
        fun=log_mle_gamma,
        x0=np.zeros(phi_values.shape[1], dtype=float),
        jac=log_mle_gamma_grad,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
        args=(phi_t, phi_log_t, z_tg, z_t),
    )

    opt_gamma = res.x
    gamma_matrix = np.vstack([opt_gamma for _ in range(phi_values.shape[0])])
    psi = transform_phi(phi, gamma_matrix)

    return {"psi": psi, "value": float(res.fun), "gamma": opt_gamma}


def get_mle_psi_mal(z_ng_mal: pd.DataFrame, pseudo_min: float) -> pd.DataFrame:
    """Generate malignant reference profile by row-wise MLE and pseudo-min adjustment."""
    row_sums = z_ng_mal.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError("z_ng_mal rows must have positive sums")

    mle_psi_mal = z_ng_mal.div(row_sums, axis=0)
    return norm_to_one(mle_psi_mal, pseudo_min=pseudo_min)


def update_reference(
    Z: np.ndarray,
    phi_prime: RefPhi,
    map: dict[str, list[str]],
    key: str | None,
    opt_control: dict[str, Any],
) -> RefPhi | RefTumor:
    """Update reference profile using initial Gibbs posterior summaries."""
    sigma = float(opt_control.get("sigma", 2.0))
    optimizer = str(opt_control.get("optimizer", "MAP"))

    opt_local = dict(opt_control)
    opt_local.pop("sigma", None)
    opt_local.pop("optimizer", None)

    cell_types = list(phi_prime.phi.index)
    gene_ids = list(phi_prime.phi.columns)

    if key is None:
        z_gt = Z.sum(axis=0)  # G x T
        if optimizer == "MAP":
            psi_res = optimize_psi(
                phi=phi_prime.phi,
                z_gt=z_gt,
                prior_num=-1 / (2 * sigma**2),
                opt_control=opt_local,
            )
            psi = psi_res["psi"]
        elif optimizer == "MLE":
            psi = optimize_psi_one_gamma(phi=phi_prime.phi, z_gt=z_gt, opt_control=opt_local)["psi"]
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")
        return RefPhi(phi=psi, pseudo_min=phi_prime.pseudo_min)

    if key not in map:
        raise ValueError("key must be present in map")

    key_idx = cell_types.index(key)
    z_ng_mal = pd.DataFrame(Z[:, :, key_idx], columns=gene_ids)
    psi_mal = get_mle_psi_mal(z_ng_mal=z_ng_mal, pseudo_min=float(phi_prime.pseudo_min or 0.0))

    cell_type_env = [ct for ct in map.keys() if ct != key]
    env_indices = [cell_types.index(ct) for ct in cell_type_env]
    z_gt_env = Z[:, :, env_indices].sum(axis=0)  # G x (T-1)
    phi_env = phi_prime.phi.loc[cell_type_env, :]

    if optimizer == "MAP":
        psi_env_res = optimize_psi(
            phi=phi_env,
            z_gt=z_gt_env,
            prior_num=-1 / (2 * sigma**2),
            opt_control=opt_local,
        )
        psi_env = psi_env_res["psi"]
    elif optimizer == "MLE":
        psi_env = optimize_psi_one_gamma(phi=phi_env, z_gt=z_gt_env, opt_control=opt_local)["psi"]
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    psi_mal.index = [f"mixture-{i + 1}" for i in range(psi_mal.shape[0])]
    return RefTumor(
        psi_mal=psi_mal,
        psi_env=psi_env,
        key=key,
        pseudo_min=phi_prime.pseudo_min,
    )
