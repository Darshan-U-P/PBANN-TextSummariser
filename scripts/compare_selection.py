
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rouge_score import rouge_scorer
from src.summarizer import PBANNSummarizer


DATA_PATH = ROOT / "data" / "processed" / "validation.jsonl"
CHECKPOINT = ROOT / "models" / "pbann_phase2_best.pt"
VOCAB = ROOT / "models" / "vocab.json"
OUTPUT_PATH = ROOT / "evaluation_results" / "selection_comparison.json"

MAX_SUMMARY_SENTENCES = 3
THRESHOLD = 0.75

rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            reference = item.get("summary", "")
            sentences = item.get("sentences")

            if not isinstance(reference, str) or not reference.strip():
                continue

            if not isinstance(sentences, list):
                article = item.get("article", "")
                if not article:
                    continue

                from src.text_encoder import split_sentences
                sentences = split_sentences(article)

            sentences = [
                " ".join(str(s).split())
                for s in sentences
                if str(s).strip()
            ]

            if not sentences:
                continue

            records.append({
                "sentences": sentences,
                "article": " ".join(sentences),
                "reference": " ".join(reference.split()),
            })

    if not records:
        raise ValueError("No valid validation records found.")

    return records


def score_summary(prediction, reference):
    result = rouge.score(reference, prediction)
    return {
        metric: result[metric].fmeasure
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def aggregate(results):
    return {
        metric: mean(item[metric] for item in results)
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def main():
    records = load_data()

    summarizer = PBANNSummarizer(
        checkpoint_path=str(CHECKPOINT),
        vocab_path=str(VOCAB),
        threshold=THRESHOLD,
        max_summary_sentences=MAX_SUMMARY_SENTENCES,
    )

    threshold_results = []
    top3_results = []
    lead3_results = []
    examples = []

    print(f"Validation examples: {len(records)}")

    for index, record in enumerate(records, start=1):
        # Use the model's normal scoring pipeline.
        scored = summarizer.score_sentences(record["article"])

        # Existing threshold-based selection.
        threshold_summary = summarizer.summarize(
            record["article"],
            threshold=THRESHOLD,
            max_summary_sentences=MAX_SUMMARY_SENTENCES,
        )

        # Top-3 selection without a probability threshold.
        top3 = sorted(
            scored,
            key=lambda item: item["score"],
            reverse=True,
        )[:MAX_SUMMARY_SENTENCES]

        top3.sort(key=lambda item: item["index"])
        top3_summary = " ".join(
            item["sentence"] for item in top3
        )

        # Simple Lead-3 baseline.
        lead3_summary = " ".join(
            record["sentences"][:MAX_SUMMARY_SENTENCES]
        )

        threshold_score = score_summary(
            threshold_summary, record["reference"]
        )
        top3_score = score_summary(
            top3_summary, record["reference"]
        )
        lead3_score = score_summary(
            lead3_summary, record["reference"]
        )

        threshold_results.append(threshold_score)
        top3_results.append(top3_score)
        lead3_results.append(lead3_score)

        examples.append({
            "reference": record["reference"],
            "threshold_summary": threshold_summary,
            "top3_summary": top3_summary,
            "lead3_summary": lead3_summary,
            "threshold_rouge": threshold_score,
            "top3_rouge": top3_score,
            "lead3_rouge": lead3_score,
            "selected_count_threshold": len([
                item for item in scored
                if item["score"] >= THRESHOLD
            ]),
            "selected_count_top3": len(top3),
        })

        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    report = {
        "checkpoint": str(CHECKPOINT),
        "split": "validation",
        "examples": len(records),
        "threshold": THRESHOLD,
        "threshold_selection": aggregate(threshold_results),
        "top3_selection": aggregate(top3_results),
        "lead3_baseline": aggregate(lead3_results),
        "example_details": examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nVALIDATION ROUGE F1")
    print("-" * 55)

    for name, result in (
        ("Threshold selection", report["threshold_selection"]),
        ("PB-ANN top-3", report["top3_selection"]),
        ("Lead-3 baseline", report["lead3_baseline"]),
    ):
        print(
            f"{name:24s} "
            f"R1={result['rouge1']:.4f}  "
            f"R2={result['rouge2']:.4f}  "
            f"RL={result['rougeL']:.4f}"
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
