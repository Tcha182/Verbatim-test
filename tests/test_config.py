import pytest

from verbatim_workflow.config import (
    Category,
    PipelineConfig,
    Settings,
    load_config,
)


def test_load_example_config():
    cfg = load_config("config/categories.example.yaml")
    assert cfg.language == "fr"
    assert cfg.other_label == "Autre"
    assert cfg.multilabel is True
    assert "Prix" in cfg.category_names
    assert cfg.all_labels[-1] == "Autre"
    assert cfg.all_labels.count("Autre") == 1
    assert cfg.settings.embedding_model == "mistral-embed"
    assert cfg.discovery.min_cluster_size == 5


def test_other_label_cannot_be_a_category():
    with pytest.raises(ValueError):
        PipelineConfig(
            categories=[Category("Prix"), Category("Autre")],
            other_label="Autre",
        )


def test_duplicate_category_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(categories=[Category("Prix"), Category("Prix")])


def test_empty_categories_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(categories=[])


def test_env_override(monkeypatch):
    monkeypatch.setenv("VERBATIM_CONCURRENCY", "3")
    s = Settings().with_env_overrides()
    assert s.concurrency == 3
