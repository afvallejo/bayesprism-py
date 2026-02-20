# bayesprism-py

Python-first BayesPrism implementation scaffold with deterministic regression tests.

## Scope in this bootstrap
- Standalone Python package `bayesprism`
- Stable API surface for core workflows
- Deterministic helper ports (preprocess, posterior helpers, reference transforms)
- Gibbs sampling scaffold and deconvolution workflow skeleton
- Synthetic fixture pipeline and regression tests

## Status
This repository is scaffolded for incremental feature completion. Differential-expression helpers are implemented in pure Python; advanced embedding internals remain staged.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Notebook validation tutorial
Notebook path:
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/notebooks/tutorial_deconvolution_validation.ipynb`

Install notebook extras:
```bash
pip install -e '.[dev,notebook]'
```

Run with default synthetic mode (fast):
```bash
BAYESPRISM_NOTEBOOK_MODE=synthetic \
BAYESPRISM_NOTEBOOK_RUN_PLOTS=1 \
BAYESPRISM_NOTEBOOK_RUN_DE=0 \
jupyter lab notebooks/tutorial_deconvolution_validation.ipynb
```

Run with optional real tutorial subset:
```bash
BAYESPRISM_NOTEBOOK_MODE=real_subset \
BAYESPRISM_TUTORIAL_RDATA=/Users/andresvallejo/Documents/00-Bioinformatics/BayesPrism/tutorial.dat/tutorial.gbm.rdata \
BAYESPRISM_NOTEBOOK_RUN_PLOTS=0 \
BAYESPRISM_NOTEBOOK_RUN_DE=0 \
jupyter lab notebooks/tutorial_deconvolution_validation.ipynb
```

Notes:
- `real_subset` loads a large tutorial matrix and then applies deterministic stratified subsetting.
- Full tutorial-scale deconvolution is intentionally not the default due runtime and memory cost.

## Equivalence notebook
Notebook path:
- `/Users/andresvallejo/Documents/00-Bioinformatics/bayesprism-py/notebooks/r_equivalence_deconvolution.ipynb`

Purpose:
- Run the Python and reference implementations on the same fixture input.
- Compare `phi`, `theta`, and `get_exp` outputs with explicit tolerances.
- Emit a machine-readable line: `EQUIVALENCE_REPORT_JSON=...`.

Environment controls:
- `BAYESPRISM_EQ_FIXTURE_DIR` (default: `tests/data/fixtures/small`)
- `BAYESPRISM_R_SOURCE_DIR` (default: `/Users/andresvallejo/Documents/00-Bioinformatics/BayesPrism/BayesPrism`)
- `BAYESPRISM_EQ_INSTALL_R_DEPS` (default: `1`)
- `BAYESPRISM_EQ_RUN_REAL_SUBSET` (default: `0`)

Run command:
```bash
BAYESPRISM_EQ_INSTALL_R_DEPS=1 \
BAYESPRISM_EQ_RUN_REAL_SUBSET=0 \
jupyter lab notebooks/r_equivalence_deconvolution.ipynb
```

Notes:
- First run can take longer while reference dependencies are installed.
- This notebook is manual/on-demand and is not run in CI.

## Project direction
- Pure Python implementation only
- Regression validation via committed synthetic fixtures
- Staged delivery for deconvolution, spatial workflows, and embedding APIs
