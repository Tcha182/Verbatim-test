"""End-to-end orchestration, independent of the durable-workflow runtime.

``run_pipeline`` is the single source of truth for the classify -> discover flow.
Both the local runner (``run_local.py``) and the durable workflow activities call
into this same logic, so behaviour is identical whether you run it as a plain
script or deploy it as a Mistral workflow surfaced in Le Chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .classify import RowClassification, classify_texts
from .config import PipelineConfig
from .discovery import ProposedCategoryResult, discover_new_categories


@dataclass
class PipelineResult:
    results: list[RowClassification]
    proposals: list[ProposedCategoryResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


async def run_pipeline(
    client, texts: list[str], config: PipelineConfig, *, progress=None
) -> PipelineResult:
    results = await classify_texts(client, texts, config, progress=progress)

    other_texts = [r.text for r in results if r.is_other and not r.error]
    proposals: list[ProposedCategoryResult] = []
    if config.discovery.enabled and other_texts:
        proposals = await discover_new_categories(client, other_texts, config)

    from .io_utils import summarize  # local import: pandas only needed for IO/summary

    summary = summarize(results, config)
    summary["proposed_categories"] = [p.name for p in proposals]
    return PipelineResult(results=results, proposals=proposals, summary=summary)
