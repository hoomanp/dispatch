SHELL := /bin/bash
.DEFAULT_GOAL := eval

PYTHON ?= $(shell if [ -f .venv/bin/python3 ]; then echo .venv/bin/python3; else echo python3; fi)

.PHONY: venv install test eval clean

venv:
	python3 -m venv .venv
	@echo "Run 'source .venv/bin/activate' to activate virtualenv"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) eval.py

clean:
	rm -rf __pycache__ .pytest_cache dispatch/__pycache__ tests/__pycache__ .cache
