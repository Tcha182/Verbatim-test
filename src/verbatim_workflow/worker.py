"""Worker entrypoint: registers the workflows and polls the Workflows platform.

Run it (after ``pip install "mistralai[workflows]"`` and setting MISTRAL_API_KEY):

    python -m verbatim_workflow.worker

The process runs until terminated. Once connected, the registered workflows can
be triggered from the platform / surfaced in Le Chat via a connector or agent.
"""

from __future__ import annotations

import asyncio

from .workflow import VerbatimClassificationOneShot, VerbatimClassificationWorkflow


async def main() -> None:
    import mistralai.workflows as workflows

    await workflows.run_worker(
        [VerbatimClassificationWorkflow, VerbatimClassificationOneShot]
    )


if __name__ == "__main__":
    asyncio.run(main())
