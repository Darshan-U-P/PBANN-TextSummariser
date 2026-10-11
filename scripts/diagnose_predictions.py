
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.summarizer import PBANNSummarizer
from src.dataset import Vocabulary

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT = (
    ROOT / "models" / "rouge_selection_experiment"
    / "pbann_best_rouge.pt"
)
VOCAB_PATH = ROOT / "models" / "vocab.json"

OUTPUT_JSON = (
    ROOT / "evaluation_results"
    / "validation_prediction_diagnostics.json"
)
OUTPUT_TXT = (
    ROOT / "evaluation_results"
    / "validation_prediction_diagnostics.txt"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3
SAMPLE_COUNT = 10

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_records():
    path = DATA_DIR / "validation.jsonl"
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            sentences = [
                " ".join(str(s).split())
                for s in item.get("sentences", [])[:MAX_SENTENCES]
                if str(s).strip()
            ]
            reference = " ".join(
                str(item.get("summary", "")).split()
            )

            if sentences and reference:
                records.append({
                    "sentences": sentences,
                    "reference": reference,
                })

    if not records:
        raise ValueError("No usable validation records found.")

    return records


def score(prediction, reference):
    values = ROUGE.score(reference, prediction)
    return {
        key: values[key].fmeasure
        for key in ("rouge1", "rouge2", "rougeL")
    }


def infer_top_k(record, summarizer, vocab):
    sentences = record["sentences"]
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    device = summarizer.device
    sentence_count = len(encoded)
    token_count = max(map(len, encoded))

    token_ids = torch.zeros(
        (1, sentence_count, token_count),
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(encoded):
        token_ids[0, i, :len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=device
        )

    token_mask = token_ids != 0
    sentence_mask = torch.ones(
        (1, sentence_count),
        dtype=torch.bool,
        device=device,
    )

    with torch.inference_mode():
        embeddings = summarizer.encoder(token_ids, token_mask)
        logits = summarizer.scorer(embeddings, sentence_mask)
        probabilities = torch.sigmoid(logits[0]).cpu().tolist()

    ranked = sorted(
        range(sentence_count),
        key=lambda i: probabilities[i],
        reverse=True,
    )
    selected = sorted(ranked[:min(TOP_K, sentence_count)])

    prediction = " ".join(sentences[i] for i in selected)

    return {
        "prediction": prediction,
        "selected_indices": selected,
        "selected_count": len(selected),
        "selected_positions": [
            (i + 1) / sentence_count for i in selected
        ],
        "scores": score(prediction, record["reference"]),
        "lead3_scores": score(
            " ".join(sentences[:TOP_K]),
            record["reference"],
        ),
    }


def mean_scores(results, field):
    return {
        metric: mean(item[field][metric] for item in results)
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    vocab = Vocabulary.load(VOCAB_PATH)
    summarizer = PBANNSummarizer(
        checkpoint_path=str(CHECKPOINT),
        vocab_path=str(VOCAB_PATH),
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
        max_summary_sentences=TOP_K,
    )

    records = load_records()
    results = []

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {summarizer.device}")
    print(f"Evaluating validation examples: {len(records)}")

    for index, record in enumerate(records, 1):
        prediction = infer_top_k(record, summarizer, vocab)
        results.append({
            "article": " ".join(record["sentences"]),
            "reference": record["reference"],
            **prediction,
        })

        if index % 100 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    counts = {}
    for item in results:
        count = item["selected_count"]
        counts[count] = counts.get(count, 0) + 1

    position_values = [
        position
        for item in results
        for position in item["selected_positions"]
    ]

    pbann = mean_scores(results, "scores")
    lead3 = mean_scores(results, "lead3_scores")

    report = {
        "split": "validation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "examples": len(results),
        "selection_method": "Top-3 by score, restored to source order",
        "selected_sentence_count_distribution": counts,
        "average_prediction_words": mean(
            len(item["prediction"].split()) for item in results
        ),
        "average_reference_words": mean(
            len(item["reference"].split()) for item in results
        ),
        "average_selected_relative_position": (
            mean(position_values) if position_values else None
        ),
        "pbann_rouge_f1": pbann,
        "lead3_rouge_f1": lead3,
        "pbann_minus_lead3": {
            metric: pbann[metric] - lead3[metric]
            for metric in pbann
        },
        "examples": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    weak = sorted(
        results, key=lambda item: item["scores"]["rougeL"]
    )[:SAMPLE_COUNT // 2]
    strong = sorted(
        results, key=lambda item: item["scores"]["rougeL"],
        reverse=True,
    )[:SAMPLE_COUNT // 2]

    lines = [
        "PB-ANN VALIDATION PREDICTION DIAGNOSTICS",
        "=" * 55,
        f"Checkpoint: {report['checkpoint']}",
        f"Examples: {len(results)}",
        "Selection: Top-3 by model score, original order restored",
        "",
        "SELECTED SENTENCE COUNT",
        str(dict(sorted(counts.items()))),
        "",
        f"Average prediction words: "
        f"{report['average_prediction_words']:.2f}",
        f"Average reference words: "
        f"{report['average_reference_words']:.2f}",
        f"Average selected relative position: "
        f"{report['average_selected_relative_position']:.3f}",
        "",
        "MEAN ROUGE F1",
        f"PB-ANN: {pbann}",
        f"Lead-3: {lead3}",
        f"Difference: {report['pbann_minus_lead3']}",
    ]

    for heading, examples in (
        ("LOW ROUGE-L EXAMPLES", weak),
        ("HIGH ROUGE-L EXAMPLES", strong),
    ):
        lines.extend(["", heading, "-" * 55])

        for item in examples:
            lines.extend([
                "",
                "REFERENCE:",
                item["reference"],
                "",
                "PB-ANN:",
                item["prediction"],
                "",
                f"Selected sentence indices (0-based): "
                f"{item['selected_indices']}",
                f"ROUGE F1: {item['scores']}",
            ])

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\nVALIDATION SUMMARY")
    print("-" * 45)
    print("Selected sentence count:", dict(sorted(counts.items())))
    print(f"Average prediction words: "
          f"{report['average_prediction_words']:.2f}")
    print(f"Average reference words: "
          f"{report['average_reference_words']:.2f}")
    print("PB-ANN ROUGE F1:", pbann)
    print("Lead-3 ROUGE F1:", lead3)
    print("PB-ANN minus Lead-3:", report["pbann_minus_lead3"])
    print("\nSaved JSON:", OUTPUT_JSON)
    print("Saved report:", OUTPUT_TXT)


if __name__ == "__main__":
    main()
