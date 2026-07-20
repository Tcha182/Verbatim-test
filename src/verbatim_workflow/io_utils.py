"""Reading input files and writing enriched results. Uses pandas only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .classify import RowClassification
from .config import PipelineConfig
from .discovery import ProposedCategoryResult


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    raise ValueError(f"Unsupported input format: {suffix}")


def _pick_text_column(df: pd.DataFrame, configured: str | None) -> str:
    if configured:
        if configured not in df.columns:
            raise ValueError(
                f"text_column {configured!r} not found. Columns: {list(df.columns)}"
            )
        return configured
    # Heuristic: the object/string column with the longest average content.
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        raise ValueError("No text-like column found; set text_column in the config.")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def read_verbatims(path: str | Path, config: PipelineConfig) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, text_column_name)."""
    df = _read_dataframe(path)
    text_col = _pick_text_column(df, config.text_column)
    return df, text_col


def build_results_frame(
    df: pd.DataFrame,
    text_col: str,
    results: list[RowClassification],
    config: PipelineConfig,
) -> pd.DataFrame:
    out = df.copy()
    out["categories"] = ["; ".join(r.categories) for r in results]
    out["is_other"] = [r.is_other for r in results]
    out["confidence"] = [r.confidence for r in results]
    out["rationale"] = [r.rationale for r in results]
    out["error"] = [r.error or "" for r in results]
    # One boolean column per known label for easy pivoting/filtering.
    for label in config.all_labels:
        out[f"cat::{label}"] = [label in r.categories for r in results]
    return out


def proposals_frame(proposals: list[ProposedCategoryResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "proposed_name": p.name,
                "description": p.description,
                "theme_summary": p.theme_summary,
                "cluster_size": p.size,
                "sample_verbatims": " | ".join(p.sample_texts),
            }
            for p in proposals
        ]
    )


def write_results(
    out_path: str | Path,
    results_df: pd.DataFrame,
    proposals: list[ProposedCategoryResult],
) -> None:
    """Write an Excel workbook: 'classified' sheet + 'proposed_categories' sheet."""
    out_path = Path(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="classified", index=False)
        prop_df = proposals_frame(proposals)
        if prop_df.empty:
            prop_df = pd.DataFrame(
                [{"proposed_name": "(no recurring theme found in Other)"}]
            )
        prop_df.to_excel(writer, sheet_name="proposed_categories", index=False)


def summarize(results: list[RowClassification], config: PipelineConfig) -> dict[str, Any]:
    counts: dict[str, int] = {label: 0 for label in config.all_labels}
    errors = 0
    for r in results:
        if r.error:
            errors += 1
        for c in r.categories:
            counts[c] = counts.get(c, 0) + 1
    return {
        "total": len(results),
        "errors": errors,
        "other_count": counts.get(config.other_label, 0),
        "per_category": counts,
    }
