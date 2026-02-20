from __future__ import annotations

import bayesprism


def test_public_api_exports() -> None:
    required = [
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
    ]
    for name in required:
        assert hasattr(bayesprism, name)
