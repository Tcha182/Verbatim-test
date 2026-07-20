import pytest
from pydantic import ValidationError

from verbatim_workflow.schemas import build_classification_model


def test_multilabel_model_accepts_known_labels():
    model, label_enum = build_classification_model(["Prix", "Livraison", "Autre"], True)
    obj = model(categories=["Prix", "Livraison"], rationale="x", confidence="high")
    values = [c.value for c in obj.categories]
    assert values == ["Prix", "Livraison"]


def test_unknown_label_rejected():
    model, _ = build_classification_model(["Prix", "Autre"], True)
    with pytest.raises(ValidationError):
        model(categories=["Inexistant"], rationale="x", confidence="high")


def test_multilabel_requires_at_least_one():
    model, _ = build_classification_model(["Prix", "Autre"], True)
    with pytest.raises(ValidationError):
        model(categories=[], rationale="x", confidence="high")


def test_singlelabel_max_one():
    model, _ = build_classification_model(["Prix", "Livraison", "Autre"], False)
    with pytest.raises(ValidationError):
        model(categories=["Prix", "Livraison"], rationale="x", confidence="high")
