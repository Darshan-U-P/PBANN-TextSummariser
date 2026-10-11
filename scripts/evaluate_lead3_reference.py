
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from rouge_score import rouge_scorer

VAL_PATH = ROOT / "data/processed/validation.jsonl"
OUTPUT_PATH = (
    ROOT / "evaluation_results/multiseed/lead3_reference_validation.json"
)

SUMMARY_SENTENCES = 3

rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def main():
    scores = []

    with VAL_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            sentences = record["sentences"][:60]
            reference = record.get("summary", "").strip()

            if not sentences or not reference:
                continue

            # Lead-3: take the first three source sentences.
            prediction = " ".join(
                sentences[:SUMMARY_SENTENCES]
            )

            result = rouge.score(reference, prediction)

            scores.append({
                "rouge1": result["rouge1"].fmeasure,
                "rouge2": result["rouge2"].fmeasure,
                "rougeL": result["rougeL"].fmeasure,
            })

    if not scores:
        raise RuntimeError("No validation examples were evaluated.")

    summary = {
        "baseline": "Lead-3",
        "reference": "original summary field",
        "articles": len(scores),
        "metrics": {
            metric: {
                "mean": float(np.mean([row[metric] for row in scores])),
                "std": float(np.std(
                    [row[metric] for row in scores], ddof=1
                )),
            }
            for metric in ("rouge1", "rouge2", "rougeL")
        },
        "per_article": scores,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=" * 60)
    print("LEAD-3 BASELINE — ORIGINAL REFERENCE SUMMARIES")
    print("=" * 60)
    print(f"Articles evaluated: {len(scores)}")

    for metric, values in summary["metrics"].items():
        print(
            f"{metric}: mean={values['mean']:.4f}, "
            f"std={values['std']:.4f}"
        )

    print(f"\nSaved: {OUTPUT_PATH}")
    print("No model checkpoints modified.")
    print("Held-out test set not used.")


if __name__ == "__main__":
    main()
