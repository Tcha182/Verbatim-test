"""Thin factory for the Mistral client. Only place that imports the SDK directly."""

from __future__ import annotations

import os


def make_client(api_key: str | None = None):
    """Instantiate a Mistral client.

    Import is done lazily and defensively: the public v1 path is
    ``from mistralai import Mistral``; some builds expose it under
    ``mistralai.client``. We try both so the factory is resilient.
    """
    try:
        from mistralai import Mistral  # type: ignore
    except Exception:  # pragma: no cover - depends on installed SDK layout
        from mistralai.client import Mistral  # type: ignore

    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is not set.")
    return Mistral(api_key=key)
