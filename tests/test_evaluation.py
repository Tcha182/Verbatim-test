import pandas as pd

from verbatim_workflow.activities import evaluate_file_activity
from verbatim_workflow.classify import RowClassification
from verbatim_workflow.config import Category, PipelineConfig, load_config
from verbatim_workflow.evaluation import evaluate, evaluate_results, parse_label_cell
from verbatim_workflow.io_utils import build_results_frame, write_results
from verbatim_workflow.prompts import classification_system_prompt

LABELS = ["Prix", "Livraison", "Autre"]


def _config():
    return PipelineConfig(categories=[Category("Prix"), Category("Livraison")], other_label="Autre")


def test_parse_label_cell_multi_case_insensitive_and_unknown():
    labels, unknown = parse_label_cell(" prix ; Livraison;;Inconnu; Prix", LABELS)
    assert labels == ["Prix", "Livraison"]
    assert unknown == ["Inconnu"]
    assert parse_label_cell(float("nan"), LABELS) == ([], [])
    assert parse_label_cell("", LABELS) == ([], [])


def test_evaluate_single_label_perfect():
    truth = [["Prix"], ["Livraison"], ["Autre"]]
    metrics, per_cat, cm = evaluate(truth, truth, LABELS)
    assert metrics["exact_match"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert cm.loc["Prix", "Prix"] == 1
    assert cm.values.sum() == 3


def test_evaluate_multilabel_partial():
    truth = [["Prix", "Livraison"], ["Livraison"]]
    pred = [["Prix"], ["Prix"]]
    metrics, per_cat, cm = evaluate(truth, pred, LABELS)
    assert metrics["exact_match"] == 0.0
    assert metrics["partial_match"] == 0.5
    assert metrics["jaccard"] == 0.25  # (1/2 + 0/2) / 2
    # Prix: tp=1, fp=1 -> precision 0.5, recall 1.0
    prix = per_cat.set_index("category").loc["Prix"]
    assert prix["precision"] == 0.5 and prix["recall"] == 1.0
    # Livraison: never predicted -> recall 0
    assert per_cat.set_index("category").loc["Livraison", "recall"] == 0.0
    # Macro averages only over labels a human used (Prix, Livraison), not Autre.
    assert metrics["macro_recall"] == 0.5
    assert cm.loc["Livraison", "Prix"] == 2


def test_evaluate_results_skips_unlabelled_and_errors():
    cfg = _config()
    results = [
        RowClassification(text="a", categories=["Prix"]),
        RowClassification(text="b", categories=["Autre"], is_other=True),
        RowClassification(text="c", error="api down"),
    ]
    ev = evaluate_results(["Prix", None, "Livraison"], results, cfg)
    assert ev.metrics["evaluated_rows"] == 1
    assert ev.metrics["skipped_unlabelled"] == 1
    assert ev.metrics["skipped_errors"] == 1
    assert ev.row_checks[0] == {"exact_match": True, "missed": [], "extra": []}
    assert ev.row_checks[1] is None and ev.row_checks[2] is None


def test_results_frame_and_workbook_with_evaluation(tmp_path):
    cfg = _config()
    df = pd.DataFrame({"verbatim": ["trop cher", "colis en retard"], "label": ["Prix", "Livraison; Prix"]})
    results = [
        RowClassification(text="trop cher", categories=["Prix"]),
        RowClassification(text="colis en retard", categories=["Livraison"]),
    ]
    ev = evaluate_results(df["label"].tolist(), results, cfg)
    out = build_results_frame(df, "verbatim", results, cfg, ev)
    assert out["eval_exact_match"].tolist() == [True, False]
    assert out["eval_missed"].tolist() == ["", "Prix"]

    path = tmp_path / "out.xlsx"
    write_results(path, out, [], ev)
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"classified", "proposed_categories", "evaluation", "confusion"} <= set(sheets)


async def test_evaluate_file_activity_reads_labeled_example():
    cfg = load_config("config/categories.example.yaml")
    df = pd.read_csv("data/verbatims.labeled.example.csv")
    # Pretend the classifier returned exactly the human labels.
    results = []
    for value in df["label"]:
        labels, unknown = parse_label_cell(value, cfg.all_labels)
        assert not unknown, f"example file uses a label outside the taxonomy: {unknown}"
        results.append(
            {"text": "", "categories": labels, "is_other": "Autre" in labels,
             "confidence": "high", "rationale": "", "error": None}
        )
    out = await evaluate_file_activity(
        "data/verbatims.labeled.example.csv", "config/categories.example.yaml", results, "label"
    )
    assert out["label_column"] == "label"
    assert out["metrics"]["evaluated_rows"] == len(df)
    assert out["metrics"]["exact_match"] == 1.0

    # No label column configured or passed -> no evaluation.
    assert await evaluate_file_activity(
        "data/verbatims.labeled.example.csv", "config/categories.example.yaml", results
    ) is None


def test_category_examples_loaded_and_in_prompt():
    cfg = load_config("config/categories.example.yaml")
    prix = cfg.categories[0]
    assert prix.examples
    assert prix.examples[0] in classification_system_prompt(cfg)
