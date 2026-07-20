"""Core classification loop.

This module never imports the Mistral SDK. It talks to *any* object that
exposes the async surface we need (``client.chat.parse_async`` and, for
discovery, ``client.embeddings.create_async``). The real client is injected by
the caller; tests inject a fake. That keeps the loop logic fully testable.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .config import PipelineConfig
from .prompts import classification_system_prompt, classification_user_prompt
from .schemas import build_classification_model


@dataclass
class RowClassification:
    text: str
    categories: list[str] = field(default_factory=list)
    is_other: bool = False
    confidence: str = ""
    rationale: str = ""
    error: str | None = None


def _extract_parsed(response):
    """Pull the parsed pydantic object out of a chat.parse response."""
    choice = response.choices[0]
    parsed = getattr(choice.message, "parsed", None)
    if parsed is None:
        raise ValueError("Model returned no parseable structured output.")
    return parsed


async def _classify_once(client, model_cls, config, text):
    messages = [
        {"role": "system", "content": classification_system_prompt(config)},
        {"role": "user", "content": classification_user_prompt(text, config.language)},
    ]
    response = await client.chat.parse_async(
        model=config.settings.classification_model,
        messages=messages,
        response_format=model_cls,
        temperature=0,
    )
    parsed = _extract_parsed(response)
    # Enum members serialize to their string value.
    labels = [getattr(c, "value", c) for c in parsed.categories]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    labels = [x for x in labels if not (x in seen or seen.add(x))]
    return RowClassification(
        text=text,
        categories=labels,
        is_other=(config.other_label in labels),
        confidence=getattr(parsed.confidence, "value", str(parsed.confidence)),
        rationale=parsed.rationale,
    )


async def _classify_with_retry(client, model_cls, config, text) -> RowClassification:
    attempts = max(1, config.settings.max_retries)
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return await _classify_once(client, model_cls, config, text)
        except Exception as exc:  # noqa: BLE001 - surfaced per-row, not swallowed globally
            last_err = exc
            if attempt < attempts - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 16.0)
    return RowClassification(text=text, error=str(last_err))


async def classify_texts(
    client, texts: list[str], config: PipelineConfig, *, progress=None
) -> list[RowClassification]:
    """Classify a list of verbatims concurrently.

    - Identical verbatims are classified once and the result is reused.
    - Concurrency is bounded by ``config.settings.concurrency``.
    - A failing row does not abort the run; its ``error`` field is populated.
    """
    model_cls, _ = build_classification_model(config.all_labels, config.multilabel)
    sem = asyncio.Semaphore(config.settings.concurrency)

    # Deduplicate on normalized text to save API calls.
    unique: dict[str, RowClassification] = {}
    order: list[str] = []
    for t in texts:
        key = (t or "").strip()
        if key not in unique:
            unique[key] = None  # type: ignore[assignment]
            order.append(key)

    async def worker(key: str) -> None:
        async with sem:
            result = await _classify_with_retry(client, model_cls, config, key)
        unique[key] = result
        if progress is not None:
            progress(1)

    await asyncio.gather(*(worker(k) for k in order))

    # Map back to the original order (with copies so callers can mutate safely).
    out: list[RowClassification] = []
    for t in texts:
        base = unique[(t or "").strip()]
        out.append(
            RowClassification(
                text=t,
                categories=list(base.categories),
                is_other=base.is_other,
                confidence=base.confidence,
                rationale=base.rationale,
                error=base.error,
            )
        )
    return out
