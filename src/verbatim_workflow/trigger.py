"""Trigger a workflow execution on the Mistral Workflows platform.

The worker (``verbatim_workflow.worker``) must already be running and connected.
This starts an execution of the ``verbatim-classification-file`` workflow, which
reads the file *on the worker* and runs the full classify + discover pipeline.

    uv run python -m verbatim_workflow.trigger \
        --input data/verbatims.example.csv \
        --config config/categories.example.yaml
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trigger the verbatim workflow.")
    ap.add_argument("--input", "-i", required=True, help="Path to the file ON THE WORKER.")
    ap.add_argument("--config", "-c", default="config/categories.yaml")
    ap.add_argument(
        "--label-column",
        "-l",
        default=None,
        help="Column with human labels in the file, to score the run.",
    )
    ap.add_argument(
        "--workflow", default="verbatim-classification-file", help="Workflow identifier."
    )
    ap.add_argument(
        "--deployment",
        default=os.environ.get("DEPLOYMENT_NAME"),
        help="Deployment name (needed only if several deployments own the workflow).",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from .mistral_client import make_client

    client = make_client()

    kwargs = {
        "workflow_identifier": args.workflow,
        "input": {"input_path": args.input, "config_path": args.config},
    }
    if args.label_column:
        kwargs["input"]["label_column"] = args.label_column
    if args.deployment:
        kwargs["deployment_name"] = args.deployment

    execution = client.workflows.execute_workflow(**kwargs)
    print(execution.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
