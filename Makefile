# Run from a worktree (not the Dropbox checkout). Uses this worktree's own .venv (see CONTRIBUTING.md).
UV := env -u UV_PROJECT_ENVIRONMENT uv
VENV := .venv/bin

.PHONY: setup test lint data help

help:
	@echo "make setup | test | lint | data"

setup:            ## create/refresh this worktree's .venv
	$(UV) sync

test:
	$(VENV)/pytest

lint:
	$(VENV)/ruff check src/retrievve tests

# Canonical live output is the shared Dropbox data/v1 (consumers read it there). Override DATA_OUT for a scratch build.
DATA_OUT ?= data/v1

data:             ## rebuild DATA_OUT (freeze-guarded); pass ARGS="--unfreeze <slice> --reason '...'" if needed
	$(VENV)/retrievve build-data --out "$(DATA_OUT)" $(ARGS)
