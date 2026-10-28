"""Score the classification against human labels (multi-label aware).

A label cell holds one or more labels separated by ";" (e.g. "Prix; Livraison").
Labels are matched to the taxonomy case-insensitively. Rows with an empty label
cell are treated as "not labelled" and skipped, so a file can be partly labelled.
Rows whose classification failed (``error``) are skipped too: they say nothing
about the classifier's quality and are reported separately.

Metrics:
- exact_match: predicted label set == human label set
- partial_match: at least one predicted label is in the human set
- jaccard: mean |pred ∩ truth| / |pred ∪ truth|
- micro / macro precision, recall, F1 over the labels (incl. the Other label)
- confusion: rows = human label, columns = predicted label, cell = number of
  verbatims where the human set contains the row label and the predicted set
  contains the column label. For single-label data this is the usual matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .classify import RowClassification
from .config import PipelineConfig

LABEL_SEPARATOR = ";"


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    per_category: pd.DataFrame
    confusion: pd.DataFrame
    # One entry per input row (None when the row was not evaluated).
    row_checks: list[dict[str, Any] | None] = field(default_factory=list)
    unknown_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (for workflow outputs)."""
        return {
            "metrics": self.metrics,
            "per_category": self.per_category.to_dict(orient="records"),
            "unknown_labels": self.unknown_labels,
        }


def parse_label_cell(value: Any, known: list[str]) -> tuple[list[str], list[str]]:
    """Split a label cell into (known labels, unknown labels)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return [], []
    lookup = {k.casefold(): k for k in known}
    labels, unknown = [], []
    for part in str(value).split(LABEL_SEPARATOR):
        part = part.strip()
        if not part:
            continue
        match = lookup.get(part.casefold())
        if match is None:
            unknown.append(part)
        elif match not in labels:
            labels.append(match)
    return labels, unknown


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def evaluate(
    truth: list[list[str]], pred: list[list[str]], labels: list[str]
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compute metrics on aligned lists of label sets."""
    if len(truth) != len(pred):
        raise ValueError("truth and pred must have the same length")
    n = len(truth)

    per_rows = []
    tp_sum = fp_sum = fn_sum = 0
    for label in labels:
        tp = sum(label in t and label in p for t, p in zip(truth, pred))
        fp = sum(label not in t and label in p for t, p in zip(truth, pred))
        fn = sum(label in t and label not in p for t, p in zip(truth, pred))
        tp_sum, fp_sum, fn_sum = tp_sum + tp, fp_sum + fp, fn_sum + fn
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        per_rows.append(
            {
                "category": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(_f1(precision, recall), 4),
                "support": tp + fn,  # verbatims a human put in this category
                "predicted": tp + fp,
            }
        )
    per_category = pd.DataFrame(per_rows)

    micro_p, micro_r = _safe_div(tp_sum, tp_sum + fp_sum), _safe_div(tp_sum, tp_sum + fn_sum)
    # Macro averages only over labels that appear in the human labels.
    present = per_category[per_category["support"] > 0]
    metrics = {
        "evaluated_rows": n,
        "exact_match": round(_safe_div(sum(set(t) == set(p) for t, p in zip(truth, pred)), n), 4),
        "partial_match": round(_safe_div(sum(bool(set(t) & set(p)) for t, p in zip(truth, pred)), n), 4),
        "jaccard": round(
            _safe_div(
                sum(_safe_div(len(set(t) & set(p)), len(set(t) | set(p))) for t, p in zip(truth, pred)),
                n,
            ),
            4,
        ),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(_f1(micro_p, micro_r), 4),
        "macro_precision": round(float(present["precision"].mean()) if len(present) else 0.0, 4),
        "macro_recall": round(float(present["recall"].mean()) if len(present) else 0.0, 4),
        "macro_f1": round(float(present["f1"].mean()) if len(present) else 0.0, 4),
    }

    confusion = pd.DataFrame(0, index=labels, columns=labels)
    for t, p in zip(truth, pred):
        for tl in t:
            for pl in p:
                confusion.loc[tl, pl] += 1
    confusion.index.name = "human \\ predicted"
    return metrics, per_category, confusion


def evaluate_results(
    label_values: list[Any], results: list[RowClassification], config: PipelineConfig
) -> EvaluationResult:
    """Score classification results against a column of human labels."""
    if len(label_values) != len(results):
        raise ValueError("label column and results must have the same length")
    known = config.all_labels

    truth, pred = [], []
    row_checks: list[dict[str, Any] | None] = []
    unknown: list[str] = []
    skipped_unlabelled = skipped_errors = 0
    for value, res in zip(label_values, results):
        labels, bad = parse_label_cell(value, known)
        unknown.extend(b for b in bad if b not in unknown)
        if not labels:
            skipped_unlabelled += 1
            row_checks.append(None)
            continue
        if res.error:
            skipped_errors += 1
            row_checks.append(None)
            continue
        truth.append(labels)
        pred.append(list(res.categories))
        row_checks.append(
            {
                "exact_match": set(labels) == set(res.categories),
                "missed": [x for x in labels if x not in res.categories],
                "extra": [x for x in res.categories if x not in labels],
            }
        )

    metrics, per_category, confusion = evaluate(truth, pred, known)
    metrics["skipped_unlabelled"] = skipped_unlabelled
    metrics["skipped_errors"] = skipped_errors
    return EvaluationResult(
        metrics=metrics,
        per_category=per_category,
        confusion=confusion,
        row_checks=row_checks,
        unknown_labels=unknown,
    )
