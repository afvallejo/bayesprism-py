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
pip install -e .[dev]
pytest -q
```

## Project direction
- Pure Python implementation only
- Regression validation via committed synthetic fixtures
- Staged delivery for deconvolution, spatial workflows, and embedding APIs
