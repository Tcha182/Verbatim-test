"""Worker entrypoint: registers the workflows and polls the Workflows platform.

Requires the workflows packages and Python 3.12+:

    uv pip install mistralai-workflows mistralai-workflows-plugins-mistralai

Then run (a DEPLOYMENT_NAME is mandatory — the worker refuses to start without it):

    DEPLOYMENT_NAME=verbatim-dev MISTRAL_API_KEY=... \
        uv run python -m verbatim_workflow.worker

The process runs until terminated. Once connected, trigger a run from the
Mistral Console (Workflows -> Start Workflow) or with
``verbatim_workflow.trigger`` / ``client.workflows.execute_workflow``.
"""

from __future__ import annotations

import asyncio
import os
import sys

from .workflow import (
    VerbatimClassificationFromFile,
    VerbatimClassificationOneShot,
    VerbatimClassificationWorkflow,
)


async def main() -> None:
    if not os.environ.get("DEPLOYMENT_NAME"):
        sys.exit("DEPLOYMENT_NAME must be set (e.g. DEPLOYMENT_NAME=verbatim-dev).")
    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("MISTRAL_API_KEY must be set.")

    import mistralai.workflows as workflows

    await workflows.run_worker(
        [
            VerbatimClassificationWorkflow,
            VerbatimClassificationOneShot,
            VerbatimClassificationFromFile,
        ]
    )


def cli() -> None:
    """Synchronous entrypoint for the ``verbatim-worker`` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
