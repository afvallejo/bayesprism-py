"""Posterior container constructors and merge helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import JointPost, ThetaPost


def new_joint_post(
    bulk_id: list[str],
    gene_id: list[str],
    cell_type: list[str],
    gibbs_list: list[dict[str, Any]],
) -> JointPost:
    """Construct a JointPost from per-sample Gibbs summaries."""
    n = len(bulk_id)
    g = len(gene_id)
    k = len(cell_type)
    if len(gibbs_list) != n:
        raise ValueError("length(gibbs_list) must equal number of bulk samples")

    z = np.zeros((n, g, k), dtype=float)
    theta = np.zeros((n, k), dtype=float)
    theta_cv = np.zeros((n, k), dtype=float)
    has_theta_cv = gibbs_list[0].get("theta_cv_n") is not None

    for i, sample in enumerate(gibbs_list):
        z[i, :, :] = np.asarray(sample["Z_n"], dtype=float)
        theta[i, :] = np.asarray(sample["theta_n"], dtype=float)
        if has_theta_cv:
            theta_cv[i, :] = np.asarray(sample["theta_cv_n"], dtype=float)

    theta_df = pd.DataFrame(theta, index=bulk_id, columns=cell_type)
    theta_cv_df = (
        pd.DataFrame(theta_cv, index=bulk_id, columns=cell_type)
        if has_theta_cv
        else pd.DataFrame()
    )

    constant = float(sum(float(sample.get("gibbs_constant", 0.0)) for sample in gibbs_list))
    return JointPost(
        Z=z,
        bulk_ids=bulk_id,
        gene_ids=gene_id,
        cell_types=cell_type,
        theta=theta_df,
        theta_cv=theta_cv_df,
        constant=constant,
    )


def new_theta_post(
    bulk_id: list[str],
    cell_type: list[str],
    gibbs_list: list[dict[str, Any]],
) -> ThetaPost:
    """Construct a ThetaPost from per-sample Gibbs summaries."""
    n = len(bulk_id)
    k = len(cell_type)
    if len(gibbs_list) != n:
        raise ValueError("length(gibbs_list) must equal number of bulk samples")

    theta = np.zeros((n, k), dtype=float)
    theta_cv = np.zeros((n, k), dtype=float)

    for i, sample in enumerate(gibbs_list):
        theta[i, :] = np.asarray(sample["theta_n"], dtype=float)
        theta_cv[i, :] = np.asarray(sample["theta_cv_n"], dtype=float)

    return ThetaPost(
        theta=pd.DataFrame(theta, index=bulk_id, columns=cell_type),
        theta_cv=pd.DataFrame(theta_cv, index=bulk_id, columns=cell_type),
    )


def merge_k(joint_post_obj: JointPost, map: dict[str, list[str]]) -> JointPost:
    """Marginalize cell states into cell types according to map."""
    bulk_id = joint_post_obj.bulk_ids
    gene_id = joint_post_obj.gene_ids
    cell_type_merged = list(map.keys())

    z_merged = np.zeros((len(bulk_id), len(gene_id), len(cell_type_merged)), dtype=float)
    theta_merged = np.zeros((len(bulk_id), len(cell_type_merged)), dtype=float)

    for k_idx, merged_name in enumerate(cell_type_merged):
        member_states = map[merged_name]
        member_idx = [joint_post_obj.cell_types.index(state) for state in member_states]
        if len(member_idx) == 1:
            z_merged[:, :, k_idx] = joint_post_obj.Z[:, :, member_idx[0]]
            theta_merged[:, k_idx] = joint_post_obj.theta.iloc[:, member_idx[0]].to_numpy()
        else:
            z_merged[:, :, k_idx] = joint_post_obj.Z[:, :, member_idx].sum(axis=2)
            theta_merged[:, k_idx] = joint_post_obj.theta.iloc[:, member_idx].sum(axis=1).to_numpy()

    return JointPost(
        Z=z_merged,
        bulk_ids=bulk_id,
        gene_ids=gene_id,
        cell_types=cell_type_merged,
        theta=pd.DataFrame(theta_merged, index=bulk_id, columns=cell_type_merged),
        constant=joint_post_obj.constant,
    )
