# Verbatim Classification — Mistral Durable Workflow

Classify free-text customer **verbatims** into a known taxonomy (multi-label),
route everything that fits nothing into an **`Autre`** bucket, then **mine that
bucket to propose new categories** for recurring themes.

It is built as a **Mistral Durable Workflow** so it can be surfaced in **Le Chat**
(the workflow runtime is Mistral's, the loop over verbatims runs on your worker),
and it ships with a **local runner** so you can validate everything against the
API before deploying anything.

---

## Should we trust the LLM, or use semantic search + KNN?

You raised this directly — here is the reasoning and the choice made.

| Approach | Needs labeled data? | Multi-label | New categories | Nuance | Cost/consistency |
|---|---|---|---|---|---|
| **LLM structured classification** | No | Native | — | High | Higher cost, needs `temperature=0` for stability |
| **Embeddings + KNN** | Yes (labeled seeds per category) | Awkward | — | Low/medium | Cheap, very consistent |
| **Embeddings + clustering** | No | — | **Yes** | — | Cheap |

**Decision:**

1. **The LLM is the classifier.** Your taxonomy is known but expressed in natural
   language, classification is **multi-label**, and you have no labeled training
   set — this is exactly where an LLM with a **closed-enum structured output**
   shines. We constrain the model with a JSON schema whose labels are *only* your
   categories + `Autre`, so it physically cannot invent a label. KNN would need a
   labeled seed set per category and handles multi-label poorly.
2. **Embeddings do the discovery, not the classification.** The right job for
   semantic vectors here is your **last step**: embed the `Autre` verbatims,
   **cluster** them, and ask the LLM to name each dense cluster → proposed new
   categories. This is unsupervised and needs no labels.
3. **Bonus uses of embeddings** (easy to add): dedupe near-identical verbatims,
   and *audit* the LLM by flagging verbatims that are semantically close but were
   classified differently.

So: **LLM (structured, multi-label) for classification + embeddings/clustering
for new-category discovery.** If you later build a labeled set, a KNN pre-filter
or LLM-vs-KNN agreement check can be bolted on without changing the core.

---

## Why a *durable workflow* and not a Le Chat "code interpreter" agent

Mistral's **code interpreter has no internet access** and cannot call the API or
compute embeddings — so it can't "loop over the API to classify each verbatim".
The primitive that *can* is a **Durable Workflow**: you define
`@workflows.activity()` functions that run on **your** worker (full network + your
`MISTRAL_API_KEY`), orchestrated with durable state that survives restarts. The
classification loop and the embedding calls live in those activities.

```
Le Chat  ──►  Durable Workflow  ──►  activity: classify_verbatims  ──► loop over API (chat.parse, structured)
                    │                 activity: discover_categories ──► embeddings ──► cluster ──► LLM names clusters
                    └── returns: enriched rows + proposed new categories
```

---

## Architecture

The **core is fully decoupled from the SDK** (the client is injected), so it is
unit-tested without network or credentials, and reused identically by the local
runner and the workflow activities.

```
src/verbatim_workflow/
  config.py        Taxonomy + runtime settings (YAML).            [no SDK]
  schemas.py       Dynamic closed-enum Pydantic model + proposals. [no SDK]
  prompts.py       FR/EN prompt builders.                          [no SDK]
  classify.py      Async classification loop: concurrency,         [no SDK]
                   retries+backoff, per-row errors, dedupe.
  discovery.py     Embeddings → HDBSCAN/KMeans → LLM naming.        [no SDK]
  pipeline.py      classify → discover orchestration.              [no SDK]
  io_utils.py      Read Excel/CSV, write enriched .xlsx.           [pandas]
  mistral_client.py  The one place that imports the SDK.
  activities.py    Durable-workflow activities (wrap the core).
  workflow.py      Durable workflow definitions.
  worker.py        `run_worker([...])` entrypoint.
  run_local.py     Run the whole thing locally against the API.
```

Classification output is a closed enum, guaranteed by JSON schema:

```jsonc
{
  "categories": ["Livraison", "Prix"],   // >=1, only known labels or "Autre"
  "rationale": "Colis en retard et jugé trop cher.",
  "confidence": "high"
}
```

---

## Configure

Everything lives in one YAML (`config/categories.example.yaml` → copy to
`config/categories.yaml`). The **category descriptions are the main lever** — be
concrete, add inclusion/exclusion criteria.

```yaml
language: fr            # instruction language (verbatims can be any language)
other_label: Autre
multilabel: true
text_column: verbatim   # optional; auto-detected otherwise
categories:
  - name: Prix
    description: Coût, tarif, rapport qualité/prix, promotions, frais…
  - name: Livraison
    description: Délais, transporteur, colis endommagé, suivi, retard…
models: { classification: mistral-medium-latest, embedding: mistral-embed, naming: mistral-medium-latest }
runtime: { concurrency: 8, max_retries: 5, embedding_batch_size: 100 }
discovery: { enabled: true, min_cluster_size: 5, max_proposals: 10, samples_per_cluster: 10 }
```

---

## Run it locally (recommended first step)

```bash
pip install -e .
export MISTRAL_API_KEY=...            # or use a .env
python -m verbatim_workflow.run_local \
    --input  data/verbatims.example.csv \
    --config config/categories.example.yaml \
    --output out.xlsx
```

You get `out.xlsx` with two sheets:
- **classified** — your rows + `categories`, `is_other`, `confidence`,
  `rationale`, and one `cat::<label>` boolean column per category.
- **proposed_categories** — new categories mined from the `Autre` bucket
  (name, description, theme, cluster size, sample verbatims).

---

## Deploy as a workflow (Le Chat)

```bash
pip install "mistralai[workflows]"
export MISTRAL_API_KEY=...
python -m verbatim_workflow.worker      # registers workflows, polls for tasks
```

`worker.py` registers two workflows:
- `verbatim-classification` — deterministic `classify → discover` (two durable
  activities, retried independently).
- `verbatim-classification-oneshot` — the whole pipeline as a single activity.

To give it a conversational front-end in Le Chat, wrap it with a **Durable Agent**
that exposes `run_pipeline_activity` as a tool (the agent asks for the file /
verbatims, calls the activity, and reports the summary + proposed categories).
Uploaded files reach the worker as text/rows passed to the entrypoint —
`io_utils.read_verbatims` handles Excel/CSV/TSV parsing on the worker side.

---

## Test

```bash
pip install -e ".[dev]"
pytest        # 15 tests, no network — fakes injected for chat.parse & embeddings
```

---

## Notes & knobs

- **Determinism:** classification runs at `temperature=0`.
- **Cost control:** identical verbatims are classified once (dedupe); tune
  `concurrency` for throughput vs. rate limits.
- **Resilience:** a failing verbatim never aborts the run — its `error` column is
  filled and the rest proceeds. Transient errors retry with exponential backoff.
- **Clustering:** HDBSCAN by default (no need to choose *k*, has a noise class);
  falls back to KMeans + silhouette. Raise `min_cluster_size` to only surface
  strong recurring themes.
- **Single vs multi-label:** set `multilabel: false` to force exactly one category.
