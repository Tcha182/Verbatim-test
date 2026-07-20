"""Run the full pipeline locally against the Mistral API — no workflow platform.

This is the fastest way to validate your taxonomy and prompts on a real file
before deploying the durable workflow to Le Chat. It uses the exact same core
(``run_pipeline``) as the workflow activities.

    python -m verbatim_workflow.run_local \
        --input data/verbatims.example.csv \
        --config config/categories.example.yaml \
        --output out.xlsx
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_config
from .io_utils import build_results_frame, read_verbatims, write_results
from .mistral_client import make_client
from .pipeline import run_pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Classify verbatims with Mistral.")
    ap.add_argument("--input", "-i", required=True, help="Excel/CSV/TSV of verbatims.")
    ap.add_argument("--config", "-c", required=True, help="Path to categories YAML.")
    ap.add_argument("--output", "-o", default="verbatims_classified.xlsx")
    return ap.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    df, text_col = read_verbatims(args.input, config)
    texts = df[text_col].fillna("").astype(str).tolist()

    done = {"n": 0}

    def progress(_):
        done["n"] += 1
        print(f"\rClassified {done['n']}/{len(texts)} unique verbatims", end="", flush=True)

    client = make_client()
    result = await run_pipeline(client, texts, config, progress=progress)
    print()

    results_df = build_results_frame(df, text_col, result.results, config)
    write_results(args.output, results_df, result.proposals)

    s = result.summary
    print(f"\nTotal: {s['total']}  |  Errors: {s['errors']}  |  Other: {s['other_count']}")
    print("Per category:")
    for label, count in sorted(s["per_category"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<28} {count}")
    if result.proposals:
        print("\nProposed new categories (from the Other bucket):")
        for p in result.proposals:
            print(f"  - {p.name} (n={p.size}): {p.theme_summary}")
    else:
        print("\nNo recurring theme found in the Other bucket.")
    print(f"\nWritten: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
