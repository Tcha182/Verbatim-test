"""Fake Mistral clients for testing the core without the SDK or network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _Msg:
    parsed: Any


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list


@dataclass
class _EmbItem:
    embedding: list[float]


@dataclass
class _EmbResp:
    data: list


class _Chat:
    def __init__(self, parse_fn):
        self._parse_fn = parse_fn

    async def parse_async(self, *, model, messages, response_format, **kwargs):
        parsed = self._parse_fn(messages, response_format)
        return _Resp(choices=[_Choice(message=_Msg(parsed=parsed))])


class _Embeddings:
    def __init__(self, embed_fn):
        self._embed_fn = embed_fn

    async def create_async(self, *, model, inputs):
        return _EmbResp(data=[_EmbItem(embedding=self._embed_fn(t)) for t in inputs])


class FakeClient:
    """Configurable fake exposing chat.parse_async and embeddings.create_async."""

    def __init__(self, parse_fn, embed_fn=None):
        self.chat = _Chat(parse_fn)
        self.embeddings = _Embeddings(embed_fn or (lambda t: [0.0]))
        self.calls: list[str] = []


def keyword_parse_fn(mapping: dict[str, str], other_label: str = "Autre"):
    """Build a parse_fn that assigns categories by keyword found in the verbatim."""

    def parse_fn(messages, response_format):
        text = messages[-1]["content"].lower()
        labels = [cat for kw, cat in mapping.items() if kw in text]
        if not labels:
            labels = [other_label]
        return response_format(
            categories=labels, rationale="test", confidence="high"
        )

    return parse_fn
