import pytest

from verbatim_workflow.classify import classify_texts
from verbatim_workflow.config import Category, PipelineConfig, Settings
from tests.conftest import FakeClient, keyword_parse_fn


def _config():
    return PipelineConfig(
        categories=[Category("Prix"), Category("Livraison"), Category("Service client")],
        other_label="Autre",
        settings=Settings(concurrency=4, max_retries=2),
    )


async def test_multilabel_classification():
    cfg = _config()
    client = FakeClient(
        keyword_parse_fn({"cher": "Prix", "livraison": "Livraison", "colis": "Livraison"})
    )
    texts = [
        "Livraison en retard et colis abîmé",  # -> Livraison (dedup of two keywords)
        "Beaucoup trop cher",  # -> Prix
        "Quelque chose de hors sujet",  # -> Autre
    ]
    results = await classify_texts(client, texts, cfg)
    assert results[0].categories == ["Livraison"]
    assert results[1].categories == ["Prix"]
    assert results[2].is_other is True


async def test_dedupe_identical_texts_classified_once():
    cfg = _config()
    counter = {"n": 0}

    def parse_fn(messages, response_format):
        counter["n"] += 1
        return response_format(categories=["Prix"], rationale="x", confidence="high")

    client = FakeClient(parse_fn)
    texts = ["Trop cher"] * 5
    results = await classify_texts(client, texts, cfg)
    assert len(results) == 5
    assert counter["n"] == 1  # only one API call for identical verbatims


async def test_row_error_does_not_abort_run():
    cfg = _config()

    def boom(messages, response_format):
        raise RuntimeError("api down")

    client = FakeClient(boom)
    results = await classify_texts(client, ["a", "b"], cfg)
    assert all(r.error for r in results)
    assert all(r.categories == [] for r in results)
