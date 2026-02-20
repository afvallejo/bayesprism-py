"""Gibbs sampling routines for deconvolution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import gammaln

from .models import GibbsSampler, JointPost, RefPhi, RefTumor, ThetaPost
from .posterior import new_joint_post, new_theta_post


def get_gibbs_idx(gibbs_control: dict[str, Any]) -> np.ndarray:
    """Return 1-based retained MCMC indices after burn-in and thinning."""
    chain_length = int(gibbs_control["chain_length"])
    burn_in = int(gibbs_control["burn_in"])
    thinning = int(gibbs_control["thinning"])
    all_idx = np.arange(1, chain_length + 1)
    burned_idx = all_idx[burn_in:]
    return burned_idx[::thinning]


def _safe_prob(prob: np.ndarray) -> np.ndarray:
    total = float(prob.sum())
    if total <= 0:
        return np.ones_like(prob) / prob.shape[0]
    return prob / total


def rdirichlet(alpha: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    """Generate one sample from Dirichlet distribution."""
    alpha_arr = np.asarray(alpha, dtype=float)
    x = rng.gamma(shape=alpha_arr, scale=1.0)
    return _safe_prob(x)


def sample_z_theta_n(
    X_n: np.ndarray,
    phi: np.ndarray,
    alpha: float,
    gibbs_idx: np.ndarray,
    compute_elbo: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray | float]:
    """Run Gibbs sampling for Z and theta for a single bulk sample."""
    if rng is None:
        rng = np.random.default_rng()

    g = phi.shape[1]
    k = phi.shape[0]

    theta_n_i = np.full(k, 1.0 / k)
    z_n_i = np.zeros((g, k), dtype=float)

    z_n_sum = np.zeros((g, k), dtype=float)
    theta_n_sum = np.zeros(k, dtype=float)
    theta_n2_sum = np.zeros(k, dtype=float)

    retain = set(int(i) for i in gibbs_idx.tolist())
    max_iter = int(np.max(gibbs_idx))

    multinom_coef = 0.0

    for i in range(1, max_iter + 1):
        prob_mat = phi * theta_n_i[:, None]
        for gene in range(g):
            z_n_i[gene, :] = rng.multinomial(int(X_n[gene]), _safe_prob(prob_mat[:, gene]))

        z_nk_i = z_n_i.sum(axis=0)
        theta_n_i = rdirichlet(alpha=z_nk_i + alpha, rng=rng)

        if i in retain:
            z_n_sum += z_n_i
            theta_n_sum += theta_n_i
            theta_n2_sum += theta_n_i**2

            if compute_elbo:
                multinom_coef += float(np.sum(gammaln(z_nk_i + 1)) - np.sum(gammaln(z_n_i + 1)))

    sample_size = len(retain)
    theta_n = theta_n_sum / sample_size
    theta_var = np.maximum(theta_n2_sum / sample_size - theta_n**2, 0.0)
    theta_cv_n = np.divide(
        np.sqrt(theta_var),
        theta_n,
        out=np.full_like(theta_n, np.inf),
        where=theta_n != 0,
    )

    return {
        "Z_n": z_n_sum / sample_size,
        "theta_n": theta_n,
        "theta_cv_n": theta_cv_n,
        "gibbs_constant": multinom_coef / sample_size,
    }


def sample_theta_n(
    X_n: np.ndarray,
    phi: np.ndarray,
    alpha: float,
    gibbs_idx: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Run Gibbs sampling for theta only for a single bulk sample."""
    if rng is None:
        rng = np.random.default_rng()

    g = phi.shape[1]
    k = phi.shape[0]

    theta_n_i = np.full(k, 1.0 / k)
    z_n_i = np.zeros((g, k), dtype=float)

    theta_n_sum = np.zeros(k, dtype=float)
    theta_n2_sum = np.zeros(k, dtype=float)

    retain = set(int(i) for i in gibbs_idx.tolist())
    max_iter = int(np.max(gibbs_idx))

    for i in range(1, max_iter + 1):
        prob_mat = phi * theta_n_i[:, None]
        for gene in range(g):
            z_n_i[gene, :] = rng.multinomial(int(X_n[gene]), _safe_prob(prob_mat[:, gene]))

        theta_n_i = rdirichlet(alpha=z_n_i.sum(axis=0) + alpha, rng=rng)

        if i in retain:
            theta_n_sum += theta_n_i
            theta_n2_sum += theta_n_i**2

    sample_size = len(retain)
    theta_n = theta_n_sum / sample_size
    theta_var = np.maximum(theta_n2_sum / sample_size - theta_n**2, 0.0)
    theta_cv_n = np.divide(
        np.sqrt(theta_var),
        theta_n,
        out=np.full_like(theta_n, np.inf),
        where=theta_n != 0,
    )

    return {"theta_n": theta_n, "theta_cv_n": theta_cv_n}


def run_gibbs_ref_phi_ini(gibbs_sampler_obj: GibbsSampler, compute_elbo: bool = False) -> JointPost:
    """Initial Gibbs sampling for RefPhi reference."""
    phi = gibbs_sampler_obj.reference.phi.to_numpy(dtype=float)
    x = gibbs_sampler_obj.X
    gibbs_control = gibbs_sampler_obj.gibbs_control

    gibbs_idx = get_gibbs_idx(gibbs_control)
    seed = gibbs_control["seed"]
    alpha = float(gibbs_control["alpha"])

    gibbs_list = []
    for n in range(x.shape[0]):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        gibbs_list.append(
            sample_z_theta_n(
                X_n=x.iloc[n].to_numpy(dtype=float),
                phi=phi,
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                compute_elbo=compute_elbo,
                rng=rng,
            )
        )

    return new_joint_post(
        bulk_id=list(x.index.astype(str)),
        gene_id=list(x.columns.astype(str)),
        cell_type=list(gibbs_sampler_obj.reference.phi.index.astype(str)),
        gibbs_list=gibbs_list,
    )


def run_gibbs_ref_phi_final(gibbs_sampler_obj: GibbsSampler) -> ThetaPost:
    """Final Gibbs sampling for RefPhi reference (theta only)."""
    phi = gibbs_sampler_obj.reference.phi.to_numpy(dtype=float)
    x = gibbs_sampler_obj.X
    gibbs_control = gibbs_sampler_obj.gibbs_control

    gibbs_idx = get_gibbs_idx(gibbs_control)
    seed = gibbs_control["seed"]
    alpha = float(gibbs_control["alpha"])

    gibbs_list = []
    for n in range(x.shape[0]):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        gibbs_list.append(
            sample_theta_n(
                X_n=x.iloc[n].to_numpy(dtype=float),
                phi=phi,
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                rng=rng,
            )
        )

    return new_theta_post(
        bulk_id=list(x.index.astype(str)),
        cell_type=list(gibbs_sampler_obj.reference.phi.index.astype(str)),
        gibbs_list=gibbs_list,
    )


def run_gibbs_ref_tumor(gibbs_sampler_obj: GibbsSampler) -> ThetaPost:
    """Final Gibbs sampling for RefTumor reference."""
    reference = gibbs_sampler_obj.reference
    x = gibbs_sampler_obj.X
    gibbs_control = gibbs_sampler_obj.gibbs_control

    gibbs_idx = get_gibbs_idx(gibbs_control)
    seed = gibbs_control["seed"]
    alpha = float(gibbs_control["alpha"])

    gibbs_list = []
    for n in range(x.shape[0]):
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        psi_mal_n = reference.psi_mal.iloc[n].to_numpy(dtype=float)
        phi_n = np.vstack([psi_mal_n, reference.psi_env.to_numpy(dtype=float)])
        nonzero_idx = np.max(phi_n, axis=0) > 0

        gibbs_list.append(
            sample_theta_n(
                X_n=x.iloc[n, nonzero_idx].to_numpy(dtype=float),
                phi=phi_n[:, nonzero_idx],
                alpha=alpha,
                gibbs_idx=gibbs_idx,
                rng=rng,
            )
        )

    return new_theta_post(
        bulk_id=list(x.index.astype(str)),
        cell_type=[reference.key] + list(reference.psi_env.index.astype(str)),
        gibbs_list=gibbs_list,
    )


def run_gibbs(
    gibbs_sampler_obj: GibbsSampler,
    final: bool,
    if_estimate: bool = True,
    compute_elbo: bool = False,
) -> JointPost | ThetaPost:
    """Run Gibbs sampling dispatcher."""
    del if_estimate  # staged: runtime estimation is not yet ported

    if isinstance(gibbs_sampler_obj.reference, RefPhi):
        if not final:
            return run_gibbs_ref_phi_ini(
                gibbs_sampler_obj=gibbs_sampler_obj,
                compute_elbo=compute_elbo,
            )
        return run_gibbs_ref_phi_final(gibbs_sampler_obj=gibbs_sampler_obj)

    if isinstance(gibbs_sampler_obj.reference, RefTumor):
        return run_gibbs_ref_tumor(gibbs_sampler_obj=gibbs_sampler_obj)

    raise TypeError("Unsupported reference type")
