"""Pydantic schemas for Mistral structured outputs.

The classification schema is built dynamically from the taxonomy so the model is
constrained to a closed enum of labels (known categories + Other). This is what
makes ``chat.parse`` reliable: the JSON schema itself forbids invented labels.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


def _label_enum(labels: list[str]) -> type[Enum]:
    # Member *names* must be valid identifiers; member *values* are the real
    # labels the model must emit. We key members by position to avoid any
    # collision with exotic category names.
    members = {f"label_{i}": label for i, label in enumerate(labels)}
    return Enum("CategoryLabel", members, type=str)


def build_classification_model(
    labels: list[str], multilabel: bool = True
) -> tuple[type[BaseModel], type[Enum]]:
    """Return a Pydantic model constraining output to the given labels.

    - multilabel=True  -> ``categories: list[CategoryLabel]`` (>=1)
    - multilabel=False -> ``categories: list[CategoryLabel]`` of length 1
      (kept as a list for a stable downstream shape; the prompt asks for one)
    """
    label_enum = _label_enum(labels)

    if multilabel:
        cats_field: Any = (
            list[label_enum],  # type: ignore[valid-type]
            Field(
                ...,
                min_length=1,
                description="All applicable categories. Use the Other label only when no known category fits.",
            ),
        )
    else:
        cats_field = (
            list[label_enum],  # type: ignore[valid-type]
            Field(
                ...,
                min_length=1,
                max_length=1,
                description="Exactly one category (the single best fit).",
            ),
        )

    model = create_model(
        "VerbatimClassification",
        __config__=ConfigDict(extra="forbid"),
        categories=cats_field,
        rationale=(
            str,
            Field(..., description="One short sentence justifying the choice."),
        ),
        confidence=(
            Confidence,
            Field(..., description="Confidence in the classification."),
        ),
    )
    return model, label_enum


class ProposedCategory(BaseModel):
    """A new category proposed from a cluster of Other verbatims."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short, human-readable category name (2-4 words).")
    description: str = Field(
        ..., description="One or two sentences describing what this category covers."
    )
    theme_summary: str = Field(
        ..., description="A one-line summary of the recurring theme in these verbatims."
    )
