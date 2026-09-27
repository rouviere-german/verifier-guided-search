# Linux/CI convenience targets. Every target delegates to tasks.py, which is the single source
# of truth and also runs on Windows: `uv run python tasks.py <target>`.

UV ?= uv
TASK := $(UV) run python tasks.py
IMAGE ?= verifier-guided-search:local

.PHONY: install lint format typecheck test bench experiments escape-suite demo bandit pip-audit audit baseline record check docker-build compose-up

install:
	$(UV) sync --frozen

lint:
	$(TASK) lint

format:
	$(TASK) format

typecheck:
	$(TASK) typecheck

test:
	$(TASK) test

bench:
	$(TASK) bench

experiments:
	$(TASK) experiments

escape-suite:
	$(TASK) escape-suite

demo:
	$(TASK) demo

bandit:
	$(TASK) bandit

pip-audit:
	$(TASK) pip-audit

audit:
	$(TASK) audit

baseline:
	$(TASK) baseline

record:
	$(TASK) record

calibrate:
	$(TASK) calibrate

check:
	$(TASK) check

docker-build:
	docker build -t $(IMAGE) .

compose-up:
	docker compose up --build
