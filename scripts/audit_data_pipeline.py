
import json
from pathlib import Path
from collections import Counter
from statistics import mean


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "train": ROOT / "data" / "processed" / "train.jsonl",
    "validation": ROOT / "data" / "processed" / "validation.jsonl",
    "test": ROOT / "data" / "processed" / "test.jsonl",
}

REPORT_PATH = (
    ROOT / "evaluation_results" / "data_pipeline_audit.json"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from exc

    return records


def get_sentence_tokens(sentence):
    """
    Approximate the project's whitespace-independent word count.
    This is a diagnostic only, not the model's exact tokenizer.
    """
    return len(str(sentence).split())


def distribution(values):
    return {
        str(key): value
        for key, value in sorted(Counter(values).items())
    }


# ============================================================
# AUDIT ONE DATASET SPLIT
# ============================================================

def audit_split(name, path):
    if not path.is_file():
        return {
            "split": name,
            "file": str(path),
            "status": "missing",
            "message": "Dataset file not found.",
        }

    records = load_jsonl(path)

    report = {
        "split": name,
        "file": str(path),
        "status": "audited",
        "examples": len(records),
        "sentence_label_length_mismatches": 0,
        "empty_sentences": 0,
        "empty_label_lists": 0,
        "invalid_labels": 0,
        "positive_labels_total": 0,
        "negative_labels_total": 0,
        "positive_labels_after_limit": 0,
        "positive_labels_retained": 0,
        "articles_over_limit": 0,
        "articles_with_positive_labels_after_limit": 0,
        "sentence_count_distribution": {},
        "positive_label_count_distribution": {},
        "retained_positive_count_distribution": {},
        "mean_sentence_count": 0.0,
        "mean_positive_labels_per_article": 0.0,
        "mean_retained_positive_labels": 0.0,
        "long_sentence_count": 0,
        "long_sentence_examples": [],
        "mismatch_examples": [],
        "truncation_examples": [],
    }

    sentence_counts = []
    positive_counts = []
    retained_positive_counts = []

    total_positive = 0
    total_negative = 0
    total_after_limit = 0
    total_retained_positive = 0
    total_sentences = 0

    for index, record in enumerate(records):
        sentences = record.get("sentences")
        labels = record.get("labels")

        if not isinstance(sentences, list):
            report["sentence_label_length_mismatches"] += 1

            if len(report["mismatch_examples"]) < 10:
                report["mismatch_examples"].append({
                    "record_index": index,
                    "problem": "Missing or invalid sentences list",
                })

            continue

        if not isinstance(labels, list):
            report["sentence_label_length_mismatches"] += 1

            if len(report["mismatch_examples"]) < 10:
                report["mismatch_examples"].append({
                    "record_index": index,
                    "problem": "Missing or invalid labels list",
                })

            continue

        if len(sentences) != len(labels):
            report["sentence_label_length_mismatches"] += 1

            if len(report["mismatch_examples"]) < 10:
                report["mismatch_examples"].append({
                    "record_index": index,
                    "sentence_count": len(sentences),
                    "label_count": len(labels),
                })

            # Do not calculate positional alignment metrics on mismatched
            # records; their sentence-label correspondence is uncertain.
            continue

        clean_labels = []
        invalid_record = False

        for label in labels:
            if isinstance(label, bool) or label not in (0, 1):
                report["invalid_labels"] += 1
                invalid_record = True
            else:
                clean_labels.append(int(label))

        if invalid_record:
            continue

        empty_count = sum(
            1 for sentence in sentences
            if not str(sentence).strip()
        )
        report["empty_sentences"] += empty_count

        if len(labels) == 0:
            report["empty_label_lists"] += 1

        count = len(sentences)
        retained_count = min(count, MAX_SENTENCES)

        positives = sum(clean_labels)
        retained_positives = sum(clean_labels[:MAX_SENTENCES])
        positives_after_limit = sum(clean_labels[MAX_SENTENCES:])

        sentence_counts.append(count)
        positive_counts.append(positives)
        retained_positive_counts.append(retained_positives)

        total_sentences += count
        total_positive += positives
        total_negative += count - positives
        total_retained_positive += retained_positives
        total_after_limit += positives_after_limit

        if count > MAX_SENTENCES:
            report["articles_over_limit"] += 1

        if positives_after_limit > 0:
            report["articles_with_positive_labels_after_limit"] += 1

            if len(report["truncation_examples"]) < 20:
                report["truncation_examples"].append({
                    "record_index": index,
                    "sentence_count": count,
                    "positive_labels_total": positives,
                    "positive_labels_retained": retained_positives,
                    "positive_labels_after_limit": positives_after_limit,
                })

        for sentence_index, sentence in enumerate(sentences):
            token_count = get_sentence_tokens(sentence)

            if token_count > MAX_TOKENS:
                report["long_sentence_count"] += 1

                if len(report["long_sentence_examples"]) < 10:
                    report["long_sentence_examples"].append({
                        "record_index": index,
                        "sentence_index": sentence_index,
                        "approximate_word_count": token_count,
                        "label": clean_labels[sentence_index],
                        "sentence": str(sentence)[:300],
                    })

    report["positive_labels_total"] = total_positive
    report["negative_labels_total"] = total_negative
    report["positive_labels_after_limit"] = total_after_limit
    report["positive_labels_retained"] = total_retained_positive

    if sentence_counts:
        report["mean_sentence_count"] = round(
            mean(sentence_counts), 3
        )
        report["mean_positive_labels_per_article"] = round(
            mean(positive_counts), 3
        )
        report["mean_retained_positive_labels"] = round(
            mean(retained_positive_counts), 3
        )

    report["sentence_count_distribution"] = distribution(
        sentence_counts
    )
    report["positive_label_count_distribution"] = distribution(
        positive_counts
    )
    report["retained_positive_count_distribution"] = distribution(
        retained_positive_counts
    )

    report["positive_fraction_before_limit"] = round(
        total_positive / max(total_sentences, 1), 6
    )

    retained_sentence_total = sum(
        min(count, MAX_SENTENCES) for count in sentence_counts
    )

    report["positive_fraction_after_limit"] = round(
        total_retained_positive / max(retained_sentence_total, 1),
        6,
    )

    report["positive_labels_removed_by_limit"] = total_after_limit

    return report


# ============================================================
# PRINT RESULTS
# ============================================================

def print_split(report):
    print("\n" + "=" * 68)
    print(f"SPLIT: {report['split'].upper()}")
    print("=" * 68)

    if report["status"] != "audited":
        print(report["message"])
        return

    print(f"Examples: {report['examples']}")
    print(
        "Sentence/label mismatches:",
        report["sentence_label_length_mismatches"],
    )
    print("Invalid labels:", report["invalid_labels"])
    print("Empty sentences:", report["empty_sentences"])
    print("Articles over 60 sentences:", report["articles_over_limit"])

    print("\nLABEL COUNTS")
    print("Positive labels:", report["positive_labels_total"])
    print("Negative labels:", report["negative_labels_total"])
    print(
        "Positive fraction before limit:",
        report["positive_fraction_before_limit"],
    )

    print("\nTRUNCATION ANALYSIS")
    print(
        "Positive labels retained:",
        report["positive_labels_retained"],
    )
    print(
        "Positive labels beyond sentence 60:",
        report["positive_labels_after_limit"],
    )
    print(
        "Articles with positive labels beyond sentence 60:",
        report["articles_with_positive_labels_after_limit"],
    )
    print(
        "Positive fraction after limit:",
        report["positive_fraction_after_limit"],
    )

    print("\nSENTENCE STATISTICS")
    print("Mean source sentences:", report["mean_sentence_count"])
    print(
        "Mean positive labels per article:",
        report["mean_positive_labels_per_article"],
    )
    print(
        "Mean retained positive labels:",
        report["mean_retained_positive_labels"],
    )
    print(
        "Sentences exceeding 64 whitespace-separated words:",
        report["long_sentence_count"],
    )

    if report["mismatch_examples"]:
        print("\nMISMATCH EXAMPLES")
        for example in report["mismatch_examples"][:5]:
            print(example)

    if report["truncation_examples"]:
        print("\nTRUNCATION EXAMPLES")
        for example in report["truncation_examples"][:5]:
            print(example)

    if report["long_sentence_examples"]:
        print("\nLONG SENTENCE EXAMPLES")
        for example in report["long_sentence_examples"][:3]:
            print(
                f"Record {example['record_index']}, "
                f"sentence {example['sentence_index']}, "
                f"words={example['approximate_word_count']}, "
                f"label={example['label']}"
            )
            print(example["sentence"])


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("PB-ANN DATA PIPELINE AUDIT")
    print("=" * 68)
    print(f"Maximum sentences used by model: {MAX_SENTENCES}")
    print(f"Maximum tokens per sentence: {MAX_TOKENS}")

    reports = []

    for name, path in DATASETS.items():
        report = audit_split(name, path)
        reports.append(report)
        print_split(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {"reports": reports},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 68)
    print("AUDIT COMPLETE")
    print("=" * 68)
    print("Report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
