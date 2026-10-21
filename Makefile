.PHONY: eval test all help

PYTHON := $(shell which ./.venv/bin/python || which python3)
PYTEST := $(shell which ./.venv/bin/pytest || which pytest)

help:
	@echo "Available commands:"
	@echo "  make eval     Run DISPATCH v2 50-workflow benchmark harness"
	@echo "  make test     Run 100% of unit test suites via pytest"

eval:
	@$(PYTHON) eval.py

test:
	@$(PYTEST) tests/ -v
