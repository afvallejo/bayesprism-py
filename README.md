# bayesprism-py

Python translation scaffold for the BayesPrism R package with parity-first goals.

## Scope in this bootstrap
- Standalone Python package `bayesprism`
- R-parity API surface for core workflows
- Deterministic helper ports (preprocess, posterior helpers, reference transforms)
- Gibbs sampling scaffold and deconvolution workflow skeleton
- Golden fixture pipeline and tests against compact R outputs

## Status
This repository is scaffolded for parity-driven porting. Some advanced features (embedding learning internals and differential-expression helpers) are intentionally staged and currently raise `NotImplementedError`.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Source of truth
- Upstream R package: `BayesPrism/`
- Priority for behavior: R code > tutorials > paper
