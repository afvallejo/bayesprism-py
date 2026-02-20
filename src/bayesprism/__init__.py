"""Public API for the BayesPrism Python translation scaffold."""

from .api import (
    get_exp,
    get_fraction,
    new_prism,
    run_prism,
    run_prism_st,
    update_theta,
)
from .embedding import learn_embedding, learn_embedding_nmf
from .models import (
    BayesPrismResult,
    BayesPrismSTResult,
    GibbsSampler,
    JointPost,
    Prism,
    RefPhi,
    RefTumor,
    ThetaPost,
)
from .qc import cleanup_genes, get_exp_stat, select_gene_type, select_marker

__all__ = [
    "new_prism",
    "run_prism",
    "run_prism_st",
    "update_theta",
    "get_fraction",
    "get_exp",
    "learn_embedding",
    "learn_embedding_nmf",
    "cleanup_genes",
    "select_gene_type",
    "get_exp_stat",
    "select_marker",
    "RefPhi",
    "RefTumor",
    "Prism",
    "ThetaPost",
    "JointPost",
    "GibbsSampler",
    "BayesPrismResult",
    "BayesPrismSTResult",
]

__version__ = "0.1.0"
