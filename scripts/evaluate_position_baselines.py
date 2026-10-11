import json
import sys
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
VAL_PATH = ROOT / "data" / "processed" / "validation.jsonl"
OUTPUT_PATH = (
    ROOT / "evaluation_results" / "position_only_baselines.json"
)

MAX_SENTENCES = 60
TOP_K = 3

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)

METRICS = ("rouge1", "rouge2", "rougeL")


def load_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            sentences = record.get("sentences", [])
            reference = record.get("summary", "")

            if not isinstance(sentences, list):
                raise ValueError(
                    f"Invalid sentences at line {line_number}"
                )

            if not isinstance(reference, str) or not reference.strip():
                continue

            sentences = [
                " ".join(str(s).split())
                for s in sentences[:MAX_SENTENCES]
                if str(s).strip()
            ]

            if not sentences:
                continue

            records.append({
                "sentences": sentences,
                "reference": " ".join(reference.split()),
            })

    if not records:
        raise ValueError(f"No valid records found: {path}")

    return records


def select_indices(n, strategy):
    if n <= TOP_K:
        return list(range(n))

    if strategy == "first3":
        indices = [0, 1, 2]

    elif strategy == "middle3":
        start = (n - TOP_K) // 2
        indices = list(range(start, start + TOP_K))

    elif strategy == "last3":
        indices = list(range(n - TOP_K, n))

    elif strategy == "evenly_spaced3":
        # Choose first, middle, and last positions.
        indices = [
            0,
            n // 2,
            n - 1,
        ]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return sorted(set(indices))


def score_summary(prediction, reference):
    scores = ROUGE.score(reference, prediction)

    return {
        metric: scores[metric].fmeasure
        for metric in METRICS
    }


def evaluate(records, strategy):
    per_article = []

    for record in records:
        sentences = record["sentences"]
        indices = select_indices(len(sentences), strategy)

        prediction = " ".join(sentences[i] for i in indices)

        per_article.append(
            score_summary(prediction, record["reference"])
        )

    return {
        metric: mean(item[metric] for item in per_article)
        for metric in METRICS
    }


def main():
    records = load_records(VAL_PATH)

    strategies = {
        "first3": "First-3 / Lead-3",
        "middle3": "Middle-3",
        "last3": "Last-3",
        "evenly_spaced3": "Evenly-spaced-3",
    }

    results = {
        "validation_articles": len(records),
        "selection": "deterministic position-only baselines",
        "strategies": {},
    }

    print("=" * 68)
    print("POSITION-ONLY BASELINES — VALIDATION SET")
    print("=" * 68)
    print("Articles:", len(records))

    for key, label in strategies.items():
        metrics = evaluate(records, key)
        results["strategies"][key] = metrics

        print(f"\n{label}")

        for metric in METRICS:
            print(f"  {metric.upper():7s} F1={metrics[metric]:.4f}")

    print("\n" + "=" * 68)
    print("COMPARISON")
    print("=" * 68)
    print(
        f"{'Strategy':22s} {'ROUGE-1':>10s} "
        f"{'ROUGE-2':>10s} {'ROUGE-L':>10s}"
    )

    for key, label in strategies.items():
        metrics = results["strategies"][key]
        print(
            f"{label:22s} "
            f"{metrics['rouge1']:10.4f} "
            f"{metrics['rouge2']:10.4f} "
            f"{metrics['rougeL']:10.4f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print("\nSaved:", OUTPUT_PATH)
    print("No model checkpoints were modified.")


if __name__ == "__main__":
    main()
