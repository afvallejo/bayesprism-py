# Parity Checklist

## Implemented
- `new_prism` preprocessing path (alignment, collapse, normalization)
- `sample.Z.theta_n` / `sample.theta_n` equivalents
- `newJointPost` / `newThetaPost` / `mergeK` equivalents
- `run_prism` + `update_theta` scaffold with update path
- MAP/MLE reference transform wrappers
- `get_fraction` / `get_exp`
- `cleanup_genes`, `select_gene_type`, `select_marker`
- `get_exp_stat` differential-expression helper with strict R-style output schema

## Staged / partial
- Runtime estimation and multiprocessing parity behavior in Gibbs loops
- Tumor-mode edge paths in `run_prism_st`
- Full embedding learning (`learn_embedding`, `learn_embedding_nmf`)
- Plot utility parity (`plot_functions.R`)

## Validation strategy
- Unit tests for shape/type invariants and deterministic helpers
- Integration tests comparing compact R golden fixtures
- Tolerances: deterministic (`atol=1e-12`), stochastic (`atol=5e-3`, `rtol=5e-2`)
