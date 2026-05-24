# =============================================================================
# tranche-pricing-paris — orchestration
# =============================================================================
#
# Common entry points
# -------------------
#   make install       editable install with dev extras
#   make data          download + cache every external series
#   make calibrate     fit GBM, Merton, Vasicek, copulas, hazards
#   make mc            run the Monte Carlo pipeline + write artifacts/results.csv
#   make figures       regenerate the 15 headline figures
#   make notebooks     execute notebooks 01-05 in place
#   make report        build the LaTeX working paper
#   make dashboard     launch the Streamlit dashboard
#   make test          pytest with coverage
#   make lint          ruff + mypy
#   make clean         remove caches, processed data, artifacts
#   make all           data -> calibrate -> mc -> figures -> notebooks -> report
# =============================================================================

PYTHON       ?= python
PIP          ?= $(PYTHON) -m pip
SHELL        := /bin/bash
.SHELLFLAGS  := -eu -o pipefail -c
.DEFAULT_GOAL := help

CONFIG       ?= config/paris_intermediate.yaml
N_SIMS       ?= 50000

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
.PHONY: install
install:
	$(PIP) install -e ".[all]"

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
.PHONY: data
data:
	$(PYTHON) -m tranche_pricing.cli data --config $(CONFIG)

.PHONY: calibrate
calibrate:
	$(PYTHON) -m tranche_pricing.cli calibrate --config $(CONFIG)

.PHONY: mc
mc:
	$(PYTHON) -m tranche_pricing.cli mc --config $(CONFIG) --n-sims $(N_SIMS)

.PHONY: figures
figures:
	$(PYTHON) -m tranche_pricing.cli figures --config $(CONFIG)

.PHONY: notebooks
notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

.PHONY: report
report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

.PHONY: dashboard
dashboard:
	streamlit run dashboard/app.py

# -----------------------------------------------------------------------------
# Quality gates
# -----------------------------------------------------------------------------
.PHONY: test
test:
	@$(PYTHON) -c "import tranche_pricing" 2>/dev/null || { echo "tranche_pricing is not importable — run 'make install' first."; exit 1; }
	pytest

.PHONY: test-fast
test-fast:
	@$(PYTHON) -c "import tranche_pricing" 2>/dev/null || { echo "tranche_pricing is not importable — run 'make install' first."; exit 1; }
	pytest -m "not slow" --no-cov

.PHONY: lint
lint:
	ruff check src tests dashboard
	ruff format --check src tests dashboard
	mypy src

.PHONY: fmt
fmt:
	ruff format src tests dashboard
	ruff check --fix src tests dashboard

# -----------------------------------------------------------------------------
# House-keeping
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf data/.cache data/processed data/interim
	rm -rf artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean-report
clean-report:
	cd report && latexmk -C

# -----------------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------------
.PHONY: all
all: data calibrate mc figures notebooks report

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1;36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Common: make install | make all | make test | make lint | make clean"
