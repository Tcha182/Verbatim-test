import numpy as np

from verbatim_workflow.config import Category, DiscoveryConfig, PipelineConfig
from verbatim_workflow.discovery import (
    cluster_embeddings,
    discover_new_categories,
    representative_indices,
)
from verbatim_workflow.schemas import ProposedCategory
from tests.conftest import FakeClient


def _two_cluster_embed_fn():
    # "paiement..." -> near axis 0; "taille..." -> near axis 1.
    def embed(text: str):
        t = text.lower()
        if "paiement" in t or "payer" in t or "3x" in t:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]

    return embed


def test_cluster_embeddings_finds_two_groups():
    emb = np.array(
        [[1, 0, 0]] * 6 + [[0, 1, 0]] * 6, dtype=np.float32
    )
    labels = cluster_embeddings(emb, min_cluster_size=3)
    # Two distinct clusters, no all-noise.
    assert len(set(l for l in labels if l >= 0)) == 2


def test_representative_indices_prefers_centroid():
    emb = np.array([[1, 0], [0.9, 0.1], [-1, 0]], dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    reps = representative_indices(emb, [0, 1, 2], k=2)
    assert set(reps) == {0, 1}


async def test_discover_new_categories_end_to_end():
    cfg = PipelineConfig(
        categories=[Category("Prix"), Category("Livraison")],
        other_label="Autre",
        discovery=DiscoveryConfig(min_cluster_size=3, samples_per_cluster=5, max_proposals=5),
    )

    def parse_fn(messages, response_format):
        text = messages[-1]["content"].lower()
        if "payer" in text or "paiement" in text or "3x" in text:
            return ProposedCategory(
                name="Facilités de paiement",
                description="Demandes de paiement en plusieurs fois.",
                theme_summary="paiement fractionné",
            )
        return ProposedCategory(
            name="Disponibilité des tailles",
            description="Problèmes de tailles indisponibles.",
            theme_summary="tailles manquantes",
        )

    client = FakeClient(parse_fn, embed_fn=_two_cluster_embed_fn())
    others = (
        ["Je voudrais payer en 3x", "Paiement en plusieurs fois svp", "Pouvoir payer en 3x sans frais"]
        + ["Taille indisponible", "Impossible de trouver ma taille", "Plus de taille M"]
    )
    proposals = await discover_new_categories(client, others, cfg)
    names = {p.name for p in proposals}
    assert len(proposals) == 2
    assert "Facilités de paiement" in names
    assert all(p.size >= 3 for p in proposals)
