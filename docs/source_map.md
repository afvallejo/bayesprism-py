# BayesPrism Python Architecture Map

## Core object model
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/models.py`
- Typed containers and shape contracts for references, posterior summaries, and workflow outputs.

## Preprocessing and input validation
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/preprocess.py`
- Matrix coercion, normalization, label collapse, and bulk outlier filtering.

## Gibbs routines
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/gibbs.py`
- Sampling kernels (`Z`, `theta`), retained-index handling, and reference dispatch.

## Posterior construction
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/posterior.py`
- Joint/theta posterior builders and state-to-type merge utilities.

## Reference update and optimization
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/reference_update.py`
- MAP/MLE wrappers, transformed reference construction, and update orchestration.

## Public workflow API
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/api.py`
- Top-level constructors and workflows: `new_prism`, `run_prism`, `run_prism_st`, `update_theta`.

## Differential-expression and marker helpers
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/qc.py`
- `get_exp_stat`: library-size normalization, one-sided pairwise Welch tests, Berger max-p aggregation, and BH correction.
- `select_marker`: marker extraction by `pval.up.min` and `min.lfc` thresholds.

## Embedding API stubs
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/src/bayesprism/embedding.py`
- Public signatures and staged placeholders for future embedding internals.
