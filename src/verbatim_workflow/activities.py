"""Durable-workflow activities.

Activities run on YOUR worker process (``worker.py``), which has full network
access and the MISTRAL_API_KEY — this is exactly why the classification loop and
the embedding calls live here rather than in a code-interpreter sandbox (which
has no internet).

The ``mistralai.workflows`` import is guarded so this module still imports in
environments without the workflows extra (local runs, unit tests). In that case
the activities are plain async functions; only the worker needs the real
decorator to make them durable.
"""

from __future__ import annotations

from dataclasses import asdict

from .classify import RowClassification, classify_texts
from .config import PipelineConfig, load_config
from .discovery import discover_new_categories
from .pipeline import run_pipeline

try:  # pragma: no cover - depends on installed SDK
    import mistralai.workflows as workflows

    _HAS_WORKFLOWS = True
except Exception:  # noqa: BLE001
    workflows = None  # type: ignore
    _HAS_WORKFLOWS = False


def _activity(*dargs, **dkwargs):
    """Return the real ``@workflows.activity`` decorator, or a no-op fallback."""
    if _HAS_WORKFLOWS:
        return workflows.activity(*dargs, **dkwargs)

    def _decorator(fn):
        return fn

    return _decorator


def _make_client():
    # Imported lazily so the module imports without the SDK present.
    from .mistral_client import make_client

    return make_client()


@_activity()
async def classify_activity(texts: list[str], config_path: str) -> list[dict]:
    """Classify verbatims. Returns a JSON-serializable list of row dicts."""
    config = load_config(config_path)
    client = _make_client()
    results = await classify_texts(client, texts, config)
    return [asdict(r) for r in results]


@_activity()
async def discover_activity(other_texts: list[str], config_path: str) -> list[dict]:
    """Propose new categories from the Other bucket. Returns row dicts."""
    config = load_config(config_path)
    client = _make_client()
    proposals = await discover_new_categories(client, other_texts, config)
    return [asdict(p) for p in proposals]


@_activity()
async def run_pipeline_activity(texts: list[str], config_path: str) -> dict:
    """Full classify + discover flow as a single activity.

    Convenient when you want one durable step rather than orchestrating the
    two activities above yourself.
    """
    config = load_config(config_path)
    client = _make_client()
    result = await run_pipeline(client, texts, config)
    return {
        "summary": result.summary,
        "results": [asdict(r) for r in result.results],
        "proposals": [asdict(p) for p in result.proposals],
    }
