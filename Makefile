# Convenience targets for the Mistral verbatim workflow (uv-based).
# Requires: uv, Python 3.12+, a .env with MISTRAL_API_KEY and DEPLOYMENT_NAME.

CONFIG ?= config/categories.example.yaml
INPUT  ?= data/verbatims.example.csv
OUTPUT ?= out.xlsx

.PHONY: install install-workflows test run evaluate start-worker execute

install:            ## Install core deps into a uv-managed venv
	uv sync

install-workflows:  ## Add the workflows packages (worker + mistralai plugin)
	uv pip install mistralai-workflows mistralai-workflows-plugins-mistralai

test:               ## Run the unit test suite (no network)
	uv run pytest -q

run:                ## Run the pipeline locally against the API (no platform)
	uv run python -m verbatim_workflow.run_local -i $(INPUT) -c $(CONFIG) -o $(OUTPUT)

evaluate:           ## Run locally on the labelled example and score it against the human labels
	uv run python -m verbatim_workflow.run_local -i data/verbatims.labeled.example.csv -c $(CONFIG) -o $(OUTPUT) --label-column label

start-worker:       ## Start the durable-workflow worker (keep this running)
	uv run python -m verbatim_workflow.worker

execute:            ## Trigger a workflow run (in a second terminal)
	uv run python -m verbatim_workflow.trigger -i $(INPUT) -c $(CONFIG)
