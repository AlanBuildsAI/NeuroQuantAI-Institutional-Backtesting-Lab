# NeuroQuantAI — Synthetic Quant Research & Analytics Lab — common commands.
# Uses the project virtualenv at .venv if present, else system python.

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install run report test clean help

help:
	@echo "Available targets:"
	@echo "  install  Install dependencies into .venv"
	@echo "  run      Run the full analytics pipeline (charts + CSVs + HTML)"
	@echo "  report   Alias for run (regenerates all deliverables)"
	@echo "  test     Run the pytest suite"
	@echo "  clean    Remove generated artefacts and caches"

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m neuroquant.pipeline

report: run

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__ site
	rm -f sample_outputs/*.csv sample_outputs/dashboard.html
	rm -f docs/assets/*.svg docs/assets/*.png
