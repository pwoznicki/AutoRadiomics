# Makefile
SHELL := /bin/bash

# Environment
.ONESHELL:
venv:
	virtualenv .venv
	source .venv/bin/activate
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	pre-commit install
	pre-commit autoupdate
