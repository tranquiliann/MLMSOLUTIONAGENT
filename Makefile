VENV ?= .venv
PYTHON ?= python3

.PHONY: venv install test lint run-agent run-infra

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .[dev]

test:
	PYTHONPATH=src $(VENV)/bin/pytest -q

lint:
	$(VENV)/bin/ruff check src tests

run-agent:
	$(VENV)/bin/python -m agent

run-infra:
	docker compose -f ../MLMSOLUTIONINFRA/docker-compose.yml up rag_service
