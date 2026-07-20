"""Discover new categories from the 'Other' bucket.

Pipeline: embed the Other verbatims -> cluster them -> for each dense cluster,
show representative verbatims to the LLM and ask it to name a new category.

No Mistral SDK import here either: embeddings come through the injected client's
``embeddings.create_async``. Clustering is pure scikit-learn/numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PipelineConfig
from .prompts import naming_system_prompt, naming_user_prompt
from .schemas import ProposedCategory


@dataclass
class ProposedCategoryResult:
    name: str
    description: str
    theme_summary: str
    size: int
    sample_texts: list[str] = field(default_factory=list)


async def embed_texts(client, texts: list[str], config: PipelineConfig) -> np.ndarray:
    """Return an (n, d) float32 array of L2-normalized embeddings."""
    batch = max(1, config.settings.embedding_batch_size)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch):
        chunk = [t.strip() or " " for t in texts[start : start + batch]]
        resp = await client.embeddings.create_async(
            model=config.settings.embedding_model, inputs=chunk
        )
        for item in resp.data:
            vectors.append(item.embedding)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Cluster L2-normalized embeddings. Returns a label per row (-1 = noise).

    Prefers HDBSCAN (no need to pick k, has a noise class). Falls back to
    KMeans + silhouette if HDBSCAN is unavailable or degenerates to all-noise.
    """
    n = len(embeddings)
    if n < max(2, min_cluster_size):
        return np.full(n, -1, dtype=int)

    try:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            metric="euclidean",  # on L2-normalized vectors ~ cosine
            copy=True,
        ).fit_predict(embeddings)
        if (labels >= 0).any():
            return labels
    except Exception:
        pass

    return _kmeans_fallback(embeddings, min_cluster_size)


def _kmeans_fallback(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    max_k = min(10, n // max(1, min_cluster_size))
    if max_k < 2:
        return np.zeros(n, dtype=int)

    best_labels = np.zeros(n, dtype=int)
    best_score = -1.0
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except Exception:
            continue
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels


def representative_indices(
    embeddings: np.ndarray, member_idx: list[int], k: int
) -> list[int]:
    """Pick the k members closest to the cluster centroid."""
    if len(member_idx) <= k:
        return list(member_idx)
    members = embeddings[member_idx]
    centroid = members.mean(axis=0)
    centroid /= np.linalg.norm(centroid) or 1.0
    sims = members @ centroid
    top = np.argsort(-sims)[:k]
    return [member_idx[i] for i in top]


async def _name_cluster(client, config, samples: list[str]) -> ProposedCategory:
    messages = [
        {"role": "system", "content": naming_system_prompt(config)},
        {"role": "user", "content": naming_user_prompt(samples, config.language)},
    ]
    resp = await client.chat.parse_async(
        model=config.settings.naming_model,
        messages=messages,
        response_format=ProposedCategory,
        temperature=0.2,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Naming step returned no structured output.")
    return parsed


async def discover_new_categories(
    client, other_texts: list[str], config: PipelineConfig
) -> list[ProposedCategoryResult]:
    """Cluster the Other verbatims and propose new categories for dense themes."""
    disc = config.discovery
    texts = [t for t in other_texts if t and t.strip()]
    if not disc.enabled or len(texts) < disc.min_cluster_size:
        return []

    embeddings = await embed_texts(client, texts, config)
    labels = cluster_embeddings(embeddings, disc.min_cluster_size)

    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        if lab < 0:
            continue
        clusters.setdefault(int(lab), []).append(idx)

    # Largest, densest themes first.
    ordered = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)

    proposals: list[ProposedCategoryResult] = []
    for _lab, member_idx in ordered:
        if len(member_idx) < disc.min_cluster_size:
            continue
        if len(proposals) >= disc.max_proposals:
            break
        rep_idx = representative_indices(embeddings, member_idx, disc.samples_per_cluster)
        samples = [texts[i] for i in rep_idx]
        proposed = await _name_cluster(client, config, samples)
        proposals.append(
            ProposedCategoryResult(
                name=proposed.name,
                description=proposed.description,
                theme_summary=proposed.theme_summary,
                size=len(member_idx),
                sample_texts=samples[:5],
            )
        )
    return proposals
