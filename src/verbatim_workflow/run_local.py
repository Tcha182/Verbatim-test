"""Run the full pipeline locally against the Mistral API — no workflow platform.

This is the fastest way to validate your taxonomy and prompts on a real file
before deploying the durable workflow to Le Chat. It uses the exact same core
(``run_pipeline``) as the workflow activities.

    python -m verbatim_workflow.run_local \
        --input data/verbatims.example.csv \
        --config config/categories.example.yaml \
        --output out.xlsx

Add ``--label-column label`` (or ``label_column`` in the YAML) to score the run
against human labels, e.g. with data/verbatims.labeled.example.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_config
from .evaluation import EvaluationResult, evaluate_results
from .io_utils import build_results_frame, label_values, read_verbatims, write_results
from .mistral_client import make_client
from .pipeline import run_pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Classify verbatims with Mistral.")
    ap.add_argument("--input", "-i", required=True, help="Excel/CSV/TSV of verbatims.")
    ap.add_argument("--config", "-c", required=True, help="Path to categories YAML.")
    ap.add_argument("--output", "-o", default="verbatims_classified.xlsx")
    ap.add_argument(
        "--label-column",
        "-l",
        default=None,
        help="Column with human labels (';' between labels) to score the run. "
        "Overrides label_column from the YAML.",
    )
    return ap.parse_args(argv)


def _print_evaluation(evaluation: EvaluationResult, label_col: str) -> None:
    m = evaluation.metrics
    print(f"\nEvaluation against human labels ({label_col!r}):")
    print(
        f"  Rows scored: {m['evaluated_rows']}  "
        f"(skipped: {m['skipped_unlabelled']} unlabelled, {m['skipped_errors']} errors)"
    )
    if not m["evaluated_rows"]:
        print("  Nothing to score: the label column is empty.")
        return
    print(
        f"  Exact match: {m['exact_match']:.1%}  |  Partial match: {m['partial_match']:.1%}"
        f"  |  Jaccard: {m['jaccard']:.3f}"
    )
    print(f"  Micro F1: {m['micro_f1']:.3f}  |  Macro F1: {m['macro_f1']:.3f}")
    print("  Per category (precision / recall / F1, support):")
    for row in evaluation.per_category.itertuples():
        print(
            f"    {row.category:<26} {row.precision:.2f} / {row.recall:.2f} / {row.f1:.2f}"
            f"  (n={row.support})"
        )
    if evaluation.unknown_labels:
        print(f"  WARNING - labels not in the taxonomy (ignored): {evaluation.unknown_labels}")


async def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    df, text_col = read_verbatims(args.input, config)
    label_col = args.label_column or config.label_column
    labels = label_values(df, label_col) if label_col else None  # fail fast if missing
    texts = df[text_col].fillna("").astype(str).tolist()

    done = {"n": 0}

    def progress(_):
        done["n"] += 1
        print(f"\rClassified {done['n']}/{len(texts)} unique verbatims", end="", flush=True)

    client = make_client()
    result = await run_pipeline(client, texts, config, progress=progress)
    print()

    evaluation = evaluate_results(labels, result.results, config) if labels is not None else None
    results_df = build_results_frame(df, text_col, result.results, config, evaluation)
    write_results(args.output, results_df, result.proposals, evaluation)

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
    if evaluation is not None:
        _print_evaluation(evaluation, label_col)
    print(f"\nWritten: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
