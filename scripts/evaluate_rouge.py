
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from rouge_score import rouge_scorer


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from src.summarizer import PBANNSummarizer


DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "pbann_phase2_best.pt"
VOCAB_PATH = ROOT / "models" / "vocab.json"
OUTPUT_DIR = ROOT / "evaluation_results"

MAX_SUMMARY_SENTENCES = 3
MAX_EVALUATION_EXAMPLES = None  # None = evaluate the full split

THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# ROUGE F1 is the primary summary-quality metric.
PRIMARY_METRIC = "rougeL"


# ============================================================
# DATASET HELPERS
# ============================================================

def find_dataset(split):
    """Find the processed JSONL file for a dataset split."""

    candidates = {
        "validation": [
            DATA_DIR / "validation.jsonl",
            DATA_DIR / "val.jsonl",
            DATA_DIR / "valid.jsonl",
        ],
        "test": [
            DATA_DIR / "test.jsonl",
        ],
    }

    for path in candidates[split]:
        if path.is_file():
            return path

    attempted = "\n".join(
        f"  - {path}" for path in candidates[split]
    )

    raise FileNotFoundError(
        f"Could not find the {split} dataset.\n"
        f"Checked:\n{attempted}"
    )


def load_records(split):
    """Load a processed JSONL dataset."""

    path = find_dataset(split)
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

            reference = record.get("summary", "")

            if not isinstance(reference, str) or not reference.strip():
                continue

            sentences = record.get("sentences")

            if isinstance(sentences, list):
                sentences = [
                    " ".join(str(s).split())
                    for s in sentences
                    if str(s).strip()
                ]
                article = " ".join(sentences)
            else:
                article = record.get("article", "")

            if not isinstance(article, str) or not article.strip():
                continue

            records.append({
                "article": article,
                "sentences": (
                    sentences if isinstance(sentences, list) else None
                ),
                "reference": " ".join(reference.split()),
            })

    if MAX_EVALUATION_EXAMPLES is not None:
        records = records[:MAX_EVALUATION_EXAMPLES]

    if not records:
        raise ValueError(
            f"No valid article/reference records found in {path}"
        )

    print(
        f"Loaded {len(records):,} {split} examples from {path}"
    )

    return records


# ============================================================
# ROUGE METRICS
# ============================================================

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def calculate_rouge(prediction, reference):
    """Calculate ROUGE precision, recall, and F1."""

    result = ROUGE.score(reference, prediction)

    return {
        metric: {
            "precision": result[metric].precision,
            "recall": result[metric].recall,
            "fmeasure": result[metric].fmeasure,
        }
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def mean_rouge(results):
    """Calculate mean ROUGE metrics across examples."""

    if not results:
        raise ValueError("Cannot calculate metrics for zero examples.")

    output = {}

    for metric in ("rouge1", "rouge2", "rougeL"):
        for field in ("precision", "recall", "fmeasure"):
            output[f"{metric}_{field}"] = mean(
                item[metric][field] for item in results
            )

    return output


def print_metrics(title, metrics):
    """Print readable aggregate metrics."""

    print(f"\n{title}")
    print("-" * len(title))

    for metric in ("rouge1", "rouge2", "rougeL"):
        print(
            f"{metric.upper():7s} "
            f"P={metrics[f'{metric}_precision']:.4f}  "
            f"R={metrics[f'{metric}_recall']:.4f}  "
            f"F1={metrics[f'{metric}_fmeasure']:.4f}"
        )


# ============================================================
# BASELINE
# ============================================================

def lead3_summary(record):
    """Use the first three source sentences as the baseline."""

    sentences = record.get("sentences")

    if isinstance(sentences, list) and sentences:
        selected = sentences[:MAX_SUMMARY_SENTENCES]
        return " ".join(selected)

    # Fallback: use the same sentence splitter as the summariser.
    from src.text_encoder import split_sentences

    selected = split_sentences(record["article"])
    return " ".join(selected[:MAX_SUMMARY_SENTENCES])


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(records, summarizer, threshold, split_name):
    """Generate summaries and compute aggregate ROUGE."""

    all_results = []
    examples = []

    total = len(records)

    print(
        f"\nEvaluating PB-ANN on {split_name}: "
        f"{total:,} examples, threshold={threshold:.2f}"
    )

    for index, record in enumerate(records, start=1):
        try:
            prediction = summarizer.summarize(
                record["article"],
                threshold=threshold,
                max_summary_sentences=MAX_SUMMARY_SENTENCES,
            )

            scores = calculate_rouge(
                prediction,
                record["reference"],
            )

            all_results.append(scores)

            examples.append({
                "article": record["article"],
                "reference": record["reference"],
                "prediction": prediction,
                "scores": scores,
            })

        except Exception as exc:
            print(
                f"WARNING: Skipping {split_name} example "
                f"{index}: {exc}"
            )

        if index % 50 == 0 or index == total:
            print(f"  Processed {index:,}/{total:,}")

    if not all_results:
        raise RuntimeError(
            f"All {split_name} examples failed during evaluation."
        )

    return mean_rouge(all_results), examples


def evaluate_lead3(records):
    """Evaluate the Lead-3 baseline on the same records."""

    results = []

    for record in records:
        prediction = lead3_summary(record)

        results.append(
            calculate_rouge(prediction, record["reference"])
        )

    return mean_rouge(results)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================

def main():
    print("=" * 68)
    print("PB-ANN TEXT SUMMARISER — ROUGE EVALUATION")
    print("=" * 68)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Phase 2 checkpoint not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCAB_PATH}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Checkpoint: {MODEL_PATH}")

    # Load both splits. Threshold selection uses validation only.
    validation_records = load_records("validation")
    test_records = load_records("test")

    summarizer = PBANNSummarizer(
        checkpoint_path=str(MODEL_PATH),
        vocab_path=str(VOCAB_PATH),
        threshold=0.60,
        max_summary_sentences=MAX_SUMMARY_SENTENCES,
        device=device,
    )

    # --------------------------------------------------------
    # 1. Tune the decision threshold on validation data
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("STAGE 1: VALIDATION THRESHOLD SELECTION")
    print("=" * 68)

    validation_results = []

    for threshold in THRESHOLDS:
        metrics, _ = evaluate_model(
            validation_records,
            summarizer,
            threshold,
            split_name="validation",
        )

        primary_score = metrics[
            f"{PRIMARY_METRIC}_fmeasure"
        ]

        validation_results.append({
            "threshold": threshold,
            "primary_rouge_f1": primary_score,
            "metrics": metrics,
        })

        print_metrics(
            f"Validation — threshold {threshold:.2f}",
            metrics,
        )

    best_result = max(
        validation_results,
        key=lambda item: item["primary_rouge_f1"],
    )

    best_threshold = best_result["threshold"]

    print("\nSelected validation threshold:", best_threshold)
    print(
        f"Validation {PRIMARY_METRIC.upper()} F1: "
        f"{best_result['primary_rouge_f1']:.4f}"
    )

    save_json(
        OUTPUT_DIR / "validation_threshold_results.json",
        {
            "checkpoint": str(MODEL_PATH),
            "primary_metric": PRIMARY_METRIC,
            "thresholds": validation_results,
            "selected_threshold": best_threshold,
        },
    )

    # --------------------------------------------------------
    # 2. Evaluate the selected model on the test set
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("STAGE 2: FINAL TEST EVALUATION")
    print("=" * 68)

    pbann_metrics, pbann_examples = evaluate_model(
        test_records,
        summarizer,
        best_threshold,
        split_name="test",
    )

    lead3_metrics = evaluate_lead3(test_records)

    print_metrics("PB-ANN — Test Set", pbann_metrics)
    print_metrics("Lead-3 Baseline — Test Set", lead3_metrics)

    print("\n" + "=" * 68)
    print("PB-ANN MINUS LEAD-3 (ROUGE F1)")
    print("=" * 68)

    for metric in ("rouge1", "rouge2", "rougeL"):
        pbann_score = pbann_metrics[f"{metric}_fmeasure"]
        lead3_score = lead3_metrics[f"{metric}_fmeasure"]
        difference = pbann_score - lead3_score

        print(
            f"{metric.upper():7s}: "
            f"{difference:+.4f} "
            f"({difference * 100:+.2f} percentage points)"
        )

    # --------------------------------------------------------
    # 3. Save final reports and sample summaries
    # --------------------------------------------------------

    report = {
        "checkpoint": str(MODEL_PATH),
        "device": device,
        "test_examples": len(pbann_examples),
        "selected_threshold": best_threshold,
        "primary_metric": PRIMARY_METRIC,
        "pbann_test": pbann_metrics,
        "lead3_test": lead3_metrics,
        "rouge_f1_difference_pbann_minus_lead3": {
            metric: (
                pbann_metrics[f"{metric}_fmeasure"]
                - lead3_metrics[f"{metric}_fmeasure"]
            )
            for metric in ("rouge1", "rouge2", "rougeL")
        },
    }

    save_json(OUTPUT_DIR / "test_report.json", report)
    save_json(OUTPUT_DIR / "test_examples.json", pbann_examples)

    print("\nSaved reports:")
    print(" ", OUTPUT_DIR / "validation_threshold_results.json")
    print(" ", OUTPUT_DIR / "test_report.json")
    print(" ", OUTPUT_DIR / "test_examples.json")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
