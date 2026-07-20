"""Durable workflow definition.

The workflow is deterministic on purpose: parse -> classify -> discover is a
fixed pipeline, so we orchestrate the activities directly rather than letting an
agent decide the order. This is more robust and predictable than an agentic loop
while remaining a first-class Mistral durable workflow (state survives worker
restarts, each activity is retried independently).

To give it a conversational front-end in Le Chat, wrap this workflow with a
Durable Agent that exposes ``run_pipeline_activity`` as a tool (see README).

The ``mistralai.workflows`` import is guarded so the module imports anywhere; the
decorators only do real work on the worker.
"""

from __future__ import annotations

from .activities import (
    classify_activity,
    discover_activity,
    parse_file_activity,
    run_pipeline_activity,
)

try:  # pragma: no cover - depends on installed SDK
    import mistralai.workflows as workflows

    _HAS_WORKFLOWS = True
except Exception:  # noqa: BLE001
    workflows = None  # type: ignore
    _HAS_WORKFLOWS = False


def _define(*dargs, **dkwargs):
    if _HAS_WORKFLOWS:
        return workflows.workflow.define(*dargs, **dkwargs)

    def _decorator(cls):
        return cls

    return _decorator


def _entrypoint(fn):
    if _HAS_WORKFLOWS:
        return workflows.workflow.entrypoint(fn)
    return fn


@_define(name="verbatim-classification")
class VerbatimClassificationWorkflow:
    """Classify verbatims then propose new categories from the Other bucket."""

    @_entrypoint
    async def run(self, texts: list[str], config_path: str = "config/categories.yaml") -> dict:
        results = await classify_activity(texts, config_path)

        other_texts = [
            r["text"] for r in results if r.get("is_other") and not r.get("error")
        ]
        proposals: list[dict] = []
        if other_texts:
            proposals = await discover_activity(other_texts, config_path)

        per_category: dict[str, int] = {}
        errors = 0
        for r in results:
            if r.get("error"):
                errors += 1
            for c in r.get("categories", []):
                per_category[c] = per_category.get(c, 0) + 1

        return {
            "summary": {
                "total": len(results),
                "errors": errors,
                "per_category": per_category,
                "proposed_categories": [p["name"] for p in proposals],
            },
            "results": results,
            "proposals": proposals,
        }


# Alternative single-step entrypoint (one durable activity does everything).
@_define(name="verbatim-classification-oneshot")
class VerbatimClassificationOneShot:
    @_entrypoint
    async def run(self, texts: list[str], config_path: str = "config/categories.yaml") -> dict:
        return await run_pipeline_activity(texts, config_path)


# Trigger with just a file path on the worker (easiest for local testing):
#   execute_workflow("verbatim-classification-file",
#                    input={"input_path": "data/verbatims.example.csv"})
@_define(name="verbatim-classification-file")
class VerbatimClassificationFromFile:
    @_entrypoint
    async def run(
        self, input_path: str, config_path: str = "config/categories.yaml"
    ) -> dict:
        texts = await parse_file_activity(input_path, config_path)
        return await run_pipeline_activity(texts, config_path)
