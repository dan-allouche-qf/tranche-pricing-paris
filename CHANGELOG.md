# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-21

Initial public release.

### Added

- `tranche_pricing` Python package with sub-packages for data ingestion,
  market dynamics (GBM, Merton jump-diffusion, Vasicek), credit modelling
  (Gaussian / Student-t / Cox doubly-stochastic copulas, Beta LGD), the
  Andersen-Sidenius-Basu (2003) waterfall (with optional
  overcollateralisation test), the Monte Carlo simulation engine
  (independent PCG64 streams, antithetic sampling, Sobol QMC), pricing,
  risk metrics, insurance pricing (actuarial and option-theoretic) and
  visualisation.
- Schema-validated YAML configuration with `extends:` inheritance for
  stress overlays (`config/`).
- Closed-form and numerical maximum-likelihood estimators for every
  market model, plus a moving-block bootstrap utility.
- Five narrative Jupyter notebooks executed end-to-end.
- LaTeX working paper (`report/`) with eleven sections, four appendices
  and twenty citations; tables are auto-generated from
  `artifacts/*.csv` by `report/build_tables.py`.
- Streamlit multi-page dashboard (`dashboard/`) with overview, scenario
  builder, tranche pricer, stress laboratory and backtest replay.
- CLI entry point `tranche-cli` and `Makefile` for one-command
  reproducibility (`make install && make all`).
- Continuous-integration workflow (`.github/workflows/ci.yml`)
  exercising `ubuntu-latest` and `macos-latest` on Python 3.11 and
  3.12, with a separate job that compiles the LaTeX paper and uploads
  the PDF as a build artefact.
- Pre-commit configuration covering `ruff`, `nbstripout` and file
  hygiene hooks.
- Citation metadata in `CITATION.cff` for GitHub's "Cite this
  repository" widget.

### Findings reported in the paper

- The mezzanine fair coupon is `NaN` on the Paris baseline under both
  the sequential and the joint solvers; the structural cash-flow
  insufficiency is documented as an empirical finding.
- The Merton jump intensity is weakly identified on the Paris log-return
  sample (σ_J collapses), which the paper discusses explicitly.

[0.1.0]: https://github.com/dan-allouche-qf/tranche-pricing-paris/releases/tag/v0.1.0
