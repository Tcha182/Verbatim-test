"""Configuration model: taxonomy + runtime settings, loaded from YAML.

Deliberately free of any Mistral SDK import so it can be validated anywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Category:
    name: str
    description: str = ""


@dataclass(frozen=True)
class Settings:
    classification_model: str = "mistral-medium-latest"
    embedding_model: str = "mistral-embed"
    naming_model: str = "mistral-medium-latest"
    concurrency: int = 8
    max_retries: int = 5
    embedding_batch_size: int = 100

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        return int(raw) if raw else default

    def with_env_overrides(self) -> "Settings":
        """Environment variables win over the YAML values."""
        return Settings(
            classification_model=os.environ.get(
                "VERBATIM_CLASSIFICATION_MODEL", self.classification_model
            ),
            embedding_model=os.environ.get(
                "VERBATIM_EMBEDDING_MODEL", self.embedding_model
            ),
            naming_model=os.environ.get("VERBATIM_NAMING_MODEL", self.naming_model),
            concurrency=self._env_int("VERBATIM_CONCURRENCY", self.concurrency),
            max_retries=self._env_int("VERBATIM_MAX_RETRIES", self.max_retries),
            embedding_batch_size=self._env_int(
                "VERBATIM_EMBEDDING_BATCH_SIZE", self.embedding_batch_size
            ),
        )


@dataclass(frozen=True)
class DiscoveryConfig:
    enabled: bool = True
    min_cluster_size: int = 5
    max_proposals: int = 10
    samples_per_cluster: int = 10


@dataclass(frozen=True)
class PipelineConfig:
    categories: list[Category]
    other_label: str = "Autre"
    language: str = "fr"
    multilabel: bool = True
    text_column: str | None = None
    id_column: str | None = None
    settings: Settings = field(default_factory=Settings)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError("At least one category must be defined.")
        names = [c.name for c in self.categories]
        if self.other_label in names:
            raise ValueError(
                f"other_label {self.other_label!r} must not also be a category name."
            )
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate category names: {sorted(dupes)}")
        if self.language not in ("fr", "en"):
            raise ValueError("language must be 'fr' or 'en'.")

    @property
    def category_names(self) -> list[str]:
        return [c.name for c in self.categories]

    @property
    def all_labels(self) -> list[str]:
        """Known categories plus the Other label — the closed label set."""
        return self.category_names + [self.other_label]


def _as_categories(raw: Any) -> list[Category]:
    cats: list[Category] = []
    for item in raw or []:
        if isinstance(item, str):
            cats.append(Category(name=item))
        elif isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError(f"Category entry missing 'name': {item!r}")
            cats.append(
                Category(name=str(name).strip(), description=str(item.get("description", "")).strip())
            )
        else:
            raise ValueError(f"Unsupported category entry: {item!r}")
    return cats


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline config from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    models = data.get("models", {}) or {}
    runtime = data.get("runtime", {}) or {}
    settings = Settings(
        classification_model=models.get("classification", Settings.classification_model),
        embedding_model=models.get("embedding", Settings.embedding_model),
        naming_model=models.get("naming", Settings.naming_model),
        concurrency=int(runtime.get("concurrency", Settings.concurrency)),
        max_retries=int(runtime.get("max_retries", Settings.max_retries)),
        embedding_batch_size=int(
            runtime.get("embedding_batch_size", Settings.embedding_batch_size)
        ),
    ).with_env_overrides()

    disc = data.get("discovery", {}) or {}
    discovery = DiscoveryConfig(
        enabled=bool(disc.get("enabled", True)),
        min_cluster_size=int(disc.get("min_cluster_size", DiscoveryConfig.min_cluster_size)),
        max_proposals=int(disc.get("max_proposals", DiscoveryConfig.max_proposals)),
        samples_per_cluster=int(
            disc.get("samples_per_cluster", DiscoveryConfig.samples_per_cluster)
        ),
    )

    return PipelineConfig(
        categories=_as_categories(data.get("categories")),
        other_label=str(data.get("other_label", "Autre")),
        language=str(data.get("language", "fr")),
        multilabel=bool(data.get("multilabel", True)),
        text_column=data.get("text_column"),
        id_column=data.get("id_column"),
        settings=settings,
        discovery=discovery,
    )
