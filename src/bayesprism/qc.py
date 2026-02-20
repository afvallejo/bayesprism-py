"""QC and marker-selection helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from importlib import resources

import pandas as pd

from .preprocess import norm_to_one

_ALLOWED_GENE_GROUPS = {
    "other_Rb",
    "chrM",
    "chrX",
    "chrY",
    "Rb",
    "Mrp",
    "act",
    "hb",
    "MALAT1",
}


def _data_path(filename: str) -> str:
    return str(resources.files("bayesprism.data").joinpath(filename))


def assign_category(input_genes: Sequence[str], species: str = "hs") -> pd.DataFrame:
    """Map genes to watchlist categories for cleanup filtering."""
    if species not in {"hs", "mm"}:
        raise ValueError("species must be 'hs' or 'mm'")

    gene_file = "genelist.hs.new.txt" if species == "hs" else "genelist.mm.new.txt"
    gene_list = pd.read_csv(_data_path(gene_file), sep="\t", header=None)

    input_genes = [str(g) for g in input_genes]
    ens_ratio = sum(g.startswith("ENS") for g in input_genes) / max(len(input_genes), 1)

    if ens_ratio > 0.8:
        input_genes_short = [g.split(".")[0] for g in input_genes]
        gene_df = gene_list[[0, 1]].copy()
    else:
        warnings.warn(
            "Gene symbols detected. ENSEMBL IDs are recommended for unique mapping.",
            RuntimeWarning,
            stacklevel=2,
        )
        input_genes_short = input_genes
        gene_df = gene_list[[0, 2]].copy()

    gene_df.columns = ["group", "gene"]
    groups = list(gene_df["group"].unique())

    matrix = pd.DataFrame(False, index=input_genes, columns=groups)
    for group in groups:
        group_genes = set(gene_df.loc[gene_df["group"] == group, "gene"])
        matrix[group] = [gene in group_genes for gene in input_genes_short]

    return matrix


def compute_specificity(input_matrix: pd.DataFrame, pseudo_min: float = 1e-8) -> pd.Series:
    """Compute max gene specificity across cell types/states."""
    ref_ct = norm_to_one(ref=input_matrix, pseudo_min=pseudo_min)
    exp_spec = ref_ct.T.div(ref_ct.sum(axis=0), axis=0)
    return exp_spec.max(axis=1)


def cleanup_genes(
    input: pd.DataFrame,
    input_type: str,
    species: str,
    gene_group: Sequence[str],
    exp_cells: int = 1,
) -> pd.DataFrame:
    """Filter genes by watchlist categories and low-expression threshold."""
    if species not in {"hs", "mm"}:
        raise ValueError("species must be 'hs' or 'mm'")
    if input_type not in {"GEP", "count.matrix"}:
        raise ValueError("please specify the correct input_type")
    if not set(gene_group).issubset(_ALLOWED_GENE_GROUPS):
        raise ValueError("gene_group contains unsupported category")

    if input_type == "GEP":
        exp_cells = min(exp_cells, 1)

    category_matrix = assign_category(input_genes=list(input.columns), species=species)
    category_matrix = category_matrix.loc[:, list(gene_group)]

    exclude_idx = category_matrix.sum(axis=1) > 0
    filtered = input.loc[:, ~exclude_idx.to_numpy()]

    if exp_cells > 0:
        keep = (filtered > 0).sum(axis=0) >= exp_cells
        filtered = filtered.loc[:, keep.to_numpy()]

    return filtered


def select_gene_type(input: pd.DataFrame, gene_type: Sequence[str]) -> pd.DataFrame:
    """Retain genes by broad GENCODE category (human only helper)."""
    allowed = {"protein_coding", "pseudogene", "lincRNA"}
    if not set(gene_type).issubset(allowed):
        raise ValueError("gene_type contains unsupported category")

    gene_list = pd.read_csv(_data_path("gencode.v22.broad.category.txt"), sep="\t", header=None)

    input_genes = [str(g) for g in input.columns]
    ens_ratio = sum(g.startswith("ENS") for g in input_genes) / max(len(input_genes), 1)

    if ens_ratio > 0.8:
        stripped = [g.split(".")[0] for g in input_genes]
        gene_df = pd.DataFrame({"gene": stripped}).merge(
            gene_list[[7, 8]].rename(columns={7: "gene", 8: "category"}),
            on="gene",
            how="left",
        )
    else:
        gene_df = pd.DataFrame({"gene": input_genes}).merge(
            gene_list[[4, 8]].rename(columns={4: "gene", 8: "category"}),
            on="gene",
            how="left",
        )

    selected = gene_df["category"].isin(set(gene_type)).to_numpy()
    return input.loc[:, selected]


def get_exp_stat(*args: object, **kwargs: object) -> dict[str, pd.DataFrame]:
    """Staged placeholder for differential-expression statistics."""
    del args, kwargs
    raise NotImplementedError(
        "get_exp_stat depends on scran/BiocParallel behavior and is staged for a parity port."
    )


def select_marker(
    sc_dat: pd.DataFrame,
    stat: dict[str, pd.DataFrame],
    pval_max: float = 0.01,
    lfc_min: float = 0.1,
) -> pd.DataFrame:
    """Select marker genes from pre-computed DE statistics."""
    markers: set[str] = set()
    for stat_df in stat.values():
        pval_col = "pval.up.min" if "pval.up.min" in stat_df.columns else "pval_up_min"
        lfc_col = "min.lfc" if "min.lfc" in stat_df.columns else "min_lfc"
        selected = stat_df[(stat_df[pval_col] < pval_max) & (stat_df[lfc_col] > lfc_min)]
        markers.update(selected.index.astype(str).tolist())

    keep = [gene for gene in sc_dat.columns.astype(str) if gene in markers]
    return sc_dat.loc[:, keep]
