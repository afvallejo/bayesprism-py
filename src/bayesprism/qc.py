"""QC and marker-selection helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import resources

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

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


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment with NA propagation."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    if not np.any(valid):
        return adjusted

    p_valid = p[valid]
    m = p_valid.size

    order = np.argsort(p_valid, kind="mergesort")
    p_sorted = p_valid[order]

    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty_like(p_valid)
    q_valid[order] = q_sorted
    adjusted[valid] = q_valid
    return adjusted


def _berger_combine(p_values: list[np.ndarray]) -> np.ndarray:
    """Berger combination used by scran for pval.type='all'."""
    if not p_values:
        return np.array([], dtype=float)

    n_genes = p_values[0].shape[0]
    combined = np.full(n_genes, np.nan, dtype=float)

    for g in range(n_genes):
        vals = np.array([p[g] for p in p_values], dtype=float)
        finite = np.isfinite(vals)
        if np.any(finite):
            combined[g] = np.max(vals[finite])

    return combined


def _choose_effect_size_all(
    p_values: list[np.ndarray],
    effects: list[np.ndarray],
) -> np.ndarray:
    """Choose effect of the comparison with worst p-value, ties by first occurrence."""
    if not p_values:
        return np.array([], dtype=float)

    n_genes = p_values[0].shape[0]
    chosen = np.full(n_genes, np.nan, dtype=float)
    worst = np.full(n_genes, -np.inf, dtype=float)

    for p, effect in zip(p_values, effects, strict=True):
        valid = np.isfinite(p)
        update = valid & (p > worst)
        chosen[update] = effect[update]
        worst[update] = p[update]

    return chosen


def _unique_ct_to_state(
    cell_type_labels: list[str],
    cell_state_labels: list[str],
) -> pd.DataFrame:
    """R-like unique(cbind(cell.type, cell.state)) preserving first occurrence."""
    out = pd.DataFrame(
        {
            "cell_type": cell_type_labels,
            "cell_state": cell_state_labels,
        }
    )
    return out.drop_duplicates(keep="first")


def _validate_get_exp_stat_inputs(
    sc_dat: pd.DataFrame,
    cell_type_labels: Sequence[str],
    cell_state_labels: Sequence[str],
    pseudo_count: float,
    cell_count_cutoff: int,
    n_cores: int,
) -> None:
    if not isinstance(sc_dat, pd.DataFrame):
        raise TypeError("sc_dat must be a pandas.DataFrame")

    n_cells = sc_dat.shape[0]
    if len(cell_type_labels) != n_cells:
        raise ValueError("length of cell_type_labels must match nrow(sc_dat)")
    if len(cell_state_labels) != n_cells:
        raise ValueError("length of cell_state_labels must match nrow(sc_dat)")

    if sc_dat.shape[1] == 0:
        raise ValueError("sc_dat must contain at least one gene")
    if pd.Index(sc_dat.columns).isnull().any():
        raise ValueError("sc_dat contains null gene names")

    values = sc_dat.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.isfinite(values).all():
        raise ValueError("sc_dat contains NaN or non-finite values")

    if any(pd.isna(x) for x in cell_type_labels):
        raise ValueError("cell_type_labels contains missing values")
    if any(pd.isna(x) for x in cell_state_labels):
        raise ValueError("cell_state_labels contains missing values")

    if pseudo_count <= 0:
        raise ValueError("pseudo_count must be positive")
    if cell_count_cutoff < 0:
        raise ValueError("cell_count_cutoff must be non-negative")
    if n_cores < 1:
        raise ValueError("n_cores must be >= 1")

    if len(set(str(x) for x in cell_state_labels)) < 2:
        raise ValueError("need at least two unique cell states")


def _pairwise_welch_up(
    dat_tmp_values: np.ndarray,
    state_to_idx: dict[str, np.ndarray],
    pair: tuple[str, str],
    gene_names: list[str],
) -> tuple[str, str, pd.DataFrame]:
    """One-sided Welch t-test in the 'up' direction for one ordered pair."""
    first, second = pair
    first_idx = state_to_idx[first]
    second_idx = state_to_idx[second]

    n1 = int(first_idx.size)
    n2 = int(second_idx.size)

    n_genes = dat_tmp_values.shape[1]
    p_value = np.full(n_genes, np.nan, dtype=float)
    log_fc = np.full(n_genes, np.nan, dtype=float)

    if n1 > 1 and n2 > 1:
        mat1 = dat_tmp_values[first_idx, :]
        mat2 = dat_tmp_values[second_idx, :]

        mean1 = np.mean(mat1, axis=0)
        mean2 = np.mean(mat2, axis=0)
        log_fc = mean1 - mean2

        var1 = np.maximum(np.var(mat1, axis=0, ddof=1), 1e-8)
        var2 = np.maximum(np.var(mat2, axis=0, ddof=1), 1e-8)

        err = var1 / n1 + var2 / n2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            test_df = err**2 / denom
            t_stat = log_fc / np.sqrt(err)

        finite = np.isfinite(test_df) & np.isfinite(t_stat)
        if np.any(finite):
            p_value[finite] = student_t.sf(t_stat[finite], df=test_df[finite])

    out = pd.DataFrame(
        {
            "logFC": log_fc,
            "p.value": p_value,
            "FDR": _bh_adjust(p_value),
        },
        index=gene_names,
    )
    return first, second, out


def _combine_markers_all(
    de_lists: list[pd.DataFrame],
    pairs: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Subset of scran::combineMarkers behavior used by get.exp.stat."""
    if len(de_lists) != pairs.shape[0]:
        raise ValueError("nrow(pairs) must equal length(de_lists)")
    if not de_lists:
        return {}

    gene_names = de_lists[0].index.astype(str)
    for cur in de_lists[1:]:
        if not gene_names.equals(cur.index.astype(str)):
            raise ValueError("row names should match across all DE tables")

    first_levels = list(dict.fromkeys(pairs["first"].astype(str).tolist()))
    output: dict[str, pd.DataFrame] = {}

    for host in first_levels:
        chosen_idx = np.flatnonzero(pairs["first"].to_numpy(dtype=str) == host)
        targets = pairs.iloc[chosen_idx]["second"].astype(str).tolist()
        cur_stats = [de_lists[int(i)] for i in chosen_idx]

        all_p = [x["p.value"].to_numpy(dtype=float) for x in cur_stats]
        all_effects = [x["logFC"].to_numpy(dtype=float) for x in cur_stats]

        pval = _berger_combine(all_p)
        fdr = _bh_adjust(pval)
        summary_lfc = _choose_effect_size_all(all_p, all_effects)

        marker_set = pd.DataFrame(index=gene_names)
        marker_set["p.value"] = pval
        marker_set["FDR"] = fdr
        marker_set["summary.logFC"] = summary_lfc

        for target, cur_stats_df in zip(targets, cur_stats, strict=True):
            marker_set[f"logFC.{target}"] = cur_stats_df["logFC"].to_numpy(dtype=float)

        output[host] = marker_set

    return output


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


def get_exp_stat(
    sc_dat: pd.DataFrame,
    cell_type_labels: Sequence[str],
    cell_state_labels: Sequence[str],
    pseudo_count: float = 0.1,
    cell_count_cutoff: int = 50,
    n_cores: int = 1,
) -> dict[str, pd.DataFrame]:
    """Perform differential-expression testing for marker gene selection."""
    _validate_get_exp_stat_inputs(
        sc_dat=sc_dat,
        cell_type_labels=cell_type_labels,
        cell_state_labels=cell_state_labels,
        pseudo_count=pseudo_count,
        cell_count_cutoff=cell_count_cutoff,
        n_cores=n_cores,
    )

    cell_type_labels = [str(x) for x in cell_type_labels]
    cell_state_labels = [str(x) for x in cell_state_labels]

    ct_to_cst = _unique_ct_to_state(cell_type_labels, cell_state_labels)
    cst_count_table = pd.Series(cell_state_labels).value_counts()
    low_count_cst = set(cst_count_table[cst_count_table < cell_count_cutoff].index.astype(str))

    lib_size = sc_dat.sum(axis=1).to_numpy(dtype=float)
    med = float(np.median(lib_size))
    if med <= 0:
        raise ValueError("median library size must be positive")

    lib_size = lib_size / med
    dat_tmp = sc_dat.div(lib_size, axis=0)
    dat_tmp = np.log2(dat_tmp + pseudo_count) - np.log2(pseudo_count)

    state_levels = sorted(set(cell_state_labels))
    state_to_idx = {
        state: np.flatnonzero(np.array(cell_state_labels, dtype=str) == state)
        for state in state_levels
    }

    pairs = [
        (first, second)
        for first in state_levels
        for second in state_levels
        if first != second
    ]

    gene_names = [str(gene) for gene in sc_dat.columns]
    dat_tmp_values = dat_tmp.to_numpy(dtype=float)

    if n_cores > 1:
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            pair_results = list(
                executor.map(
                    lambda pair: _pairwise_welch_up(
                        dat_tmp_values=dat_tmp_values,
                        state_to_idx=state_to_idx,
                        pair=pair,
                        gene_names=gene_names,
                    ),
                    pairs,
                )
            )
    else:
        pair_results = [
            _pairwise_welch_up(
                dat_tmp_values=dat_tmp_values,
                state_to_idx=state_to_idx,
                pair=pair,
                gene_names=gene_names,
            )
            for pair in pairs
        ]

    pair_df = pd.DataFrame(
        {
            "first": [first for first, _, _ in pair_results],
            "second": [second for _, second, _ in pair_results],
        }
    )
    stat_tables = [stats for _, _, stats in pair_results]

    state_to_type: dict[str, str] = {}
    for row in ct_to_cst.itertuples(index=False):
        if row.cell_state not in state_to_type:
            state_to_type[row.cell_state] = row.cell_type

    pair_first_type = [state_to_type.get(state, "") for state in pair_df["first"].astype(str)]
    pair_second_type = [state_to_type.get(state, "") for state in pair_df["second"].astype(str)]
    filter_idx = [
        first_type != second_type and second not in low_count_cst
        for first_type, second_type, second in zip(
            pair_first_type,
            pair_second_type,
            pair_df["second"].astype(str),
            strict=True,
        )
    ]

    pair_df = pair_df.loc[filter_idx].reset_index(drop=True)
    stat_tables = [stats for stats, keep in zip(stat_tables, filter_idx, strict=True) if keep]

    output_up = _combine_markers_all(stat_tables, pair_df) if not pair_df.empty else {}

    all_ct = list(dict.fromkeys(ct_to_cst["cell_type"].astype(str).tolist()))
    ct_stat_list: dict[str, pd.DataFrame] = {}

    for ct in all_ct:
        cst_i = ct_to_cst.loc[ct_to_cst["cell_type"] == ct, "cell_state"].astype(str).tolist()
        output_up_i = [output_up[cst] for cst in cst_i if cst in output_up]

        if not output_up_i:
            ct_stat_list[ct] = pd.DataFrame(
                {
                    "pval.up.min": np.full(len(gene_names), np.nan, dtype=float),
                    "min.lfc": np.full(len(gene_names), np.nan, dtype=float),
                },
                index=gene_names,
            )
            continue

        pval_up_i = np.column_stack(
            [entry["p.value"].to_numpy(dtype=float) for entry in output_up_i]
        )
        pval_up_min_i = np.min(pval_up_i, axis=1)

        lfc_state = []
        for entry in output_up_i:
            logfc_cols = [col for col in entry.columns if "logFC" in col]
            lfc_state.append(np.min(entry[logfc_cols].to_numpy(dtype=float), axis=1))

        lfc_i = np.max(np.column_stack(lfc_state), axis=1)

        ct_stat_list[ct] = pd.DataFrame(
            {
                "pval.up.min": pval_up_min_i,
                "min.lfc": lfc_i,
            },
            index=gene_names,
        )

    return ct_stat_list


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
