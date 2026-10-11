import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary
from src.summarizer import PBANNSummarizer

DATA_PATH = ROOT / "data" / "processed" / "validation.jsonl"
CHECKPOINT = (
    ROOT / "models" / "rouge_selection_experiment"
    / "pbann_best_rouge.pt"
)
VOCAB_PATH = ROOT / "models" / "vocab.json"
OUTPUT_PATH = ROOT / "evaluation_results" / "current_ranking_audit.json"

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = (1, 3)
EXAMPLES_PER_CATEGORY = 5


def load_records():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            item = json.loads(line)
            sentences = item.get("sentences", [])
            labels = item.get("labels", [])

            if not isinstance(sentences, list):
                continue
            if not isinstance(labels, list):
                continue
            if len(sentences) != len(labels):
                continue

            # Preserve sentence-label alignment while cleaning.
            pairs = [
                (" ".join(str(sentence).split()), int(label))
                for sentence, label in zip(sentences, labels)
                if str(sentence).strip()
            ]

            pairs = pairs[:MAX_SENTENCES]

            if not pairs:
                continue

            records.append({
                "sentences": [pair[0] for pair in pairs],
                "labels": [pair[1] for pair in pairs],
                "source_line": line_number,
            })

    if not records:
        raise ValueError("No usable validation records with labels.")

    return records


def rank_sentences(sentences, summarizer, vocab):
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    device = summarizer.device
    token_count = max(max(map(len, encoded)), 1)

    token_ids = torch.zeros(
        (1, len(encoded), token_count),
        dtype=torch.long,
        device=device,
    )

    for index, ids in enumerate(encoded):
        if ids:
            token_ids[0, index, :len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )

    token_mask = token_ids != 0
    sentence_mask = torch.ones(
        (1, len(encoded)),
        dtype=torch.bool,
        device=device,
    )

    with torch.inference_mode():
        embeddings = summarizer.encoder(token_ids, token_mask)
        logits = summarizer.scorer(embeddings, sentence_mask)
        scores = torch.sigmoid(logits[0]).cpu().tolist()

    return sorted(
        range(len(sentences)),
        key=lambda index: scores[index],
        reverse=True,
    ), scores


def length_group(sentence_count):
    if sentence_count <= 10:
        return "short_1_10"
    if sentence_count <= 30:
        return "medium_11_30"
    return "long_31_plus"


def ranking_metrics(labels, ranked):
    positive_ranks = [
        rank
        for rank, index in enumerate(ranked, start=1)
        if labels[index] == 1
    ]

    result = {
        "positive_label_count": sum(labels),
        "sentence_count": len(labels),
        "top1_hit": int(any(labels[index] == 1 for index in ranked[:1])),
        "top3_hit": int(any(labels[index] == 1 for index in ranked[:3])),
        "first_positive_rank": (
            positive_ranks[0] if positive_ranks else None
        ),
        "reciprocal_rank": (
            1.0 / positive_ranks[0] if positive_ranks else 0.0
        ),
        "top3_positive_count": sum(
            labels[index] for index in ranked[:3]
        ),
        "top3_precision": (
            sum(labels[index] for index in ranked[:3])
            / min(3, len(labels))
        ),
    }

    return result


def summarize_group(items):
    if not items:
        return {"articles": 0}

    return {
        "articles": len(items),
        "mean_sentence_count": mean(
            item["sentence_count"] for item in items
        ),
        "positive_label_fraction": mean(
            item["positive_label_fraction"] for item in items
        ),
        "top1_hit_rate": mean(item["top1_hit"] for item in items),
        "top3_hit_rate": mean(item["top3_hit"] for item in items),
        "mean_reciprocal_rank": mean(
            item["reciprocal_rank"] for item in items
        ),
        "mean_first_positive_rank_when_present": (
            mean([
                item["first_positive_rank"]
                for item in items
                if item["first_positive_rank"] is not None
            ])
            if any(
                item["first_positive_rank"] is not None
                for item in items
            )
            else None
        ),
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError(VOCAB_PATH)

    vocab = Vocabulary.load(VOCAB_PATH)
    summarizer = PBANNSummarizer(
        checkpoint_path=str(CHECKPOINT),
        vocab_path=str(VOCAB_PATH),
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
        max_summary_sentences=3,
    )

    records = load_records()
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {summarizer.device}")
    print(f"Validation articles: {len(records)}")
    print("Running ranking diagnostics...")

    article_results = []
    examples = []

    for number, record in enumerate(records, 1):
        sentences = record["sentences"]
        labels = record["labels"]

        ranked, scores = rank_sentences(
            sentences, summarizer, vocab
        )
        metrics = ranking_metrics(labels, ranked)

        result = {
            **metrics,
            "source_line": record["source_line"],
            "sentence_count": len(sentences),
            "positive_label_fraction": mean(labels),
            "length_group": length_group(len(sentences)),
        }
        article_results.append(result)

        # Capture the model's highest-ranked sentence and top three.
        examples.append({
            "source_line": record["source_line"],
            "sentence_count": len(sentences),
            "top1_hit": metrics["top1_hit"],
            "top1": {
                "index": ranked[0],
                "score": scores[ranked[0]],
                "label": labels[ranked[0]],
                "sentence": sentences[ranked[0]],
            },
            "top3": [
                {
                    "index": index,
                    "score": scores[index],
                    "label": labels[index],
                    "sentence": sentences[index],
                }
                for index in ranked[:3]
            ],
        })

        if number % 100 == 0 or number == len(records):
            print(f"Processed {number}/{len(records)}")

    groups = defaultdict(list)
    for item in article_results:
        groups[item["length_group"]].append(item)

    # Top-ranked false-label examples are high-score predictions
    # that disagree with the heuristic labels.
    missed = [
        item for item in examples if item["top1"]["label"] == 0
    ]
    hits = [
        item for item in examples if item["top1"]["label"] == 1
    ]

    missed.sort(key=lambda item: item["top1"]["score"], reverse=True)
    hits.sort(key=lambda item: item["top1"]["score"], reverse=True)

    labeled_articles = [
        item for item in article_results
        if item["positive_label_count"] > 0
    ]

    report = {
        "split": "validation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "articles": len(article_results),
        "sentences": sum(
            item["sentence_count"] for item in article_results
        ),
        "overall": {
            "positive_label_fraction": mean(
                item["positive_label_fraction"]
                for item in article_results
            ),
            "top1_hit_rate": mean(
                item["top1_hit"] for item in article_results
            ),
            "top3_hit_rate": mean(
                item["top3_hit"] for item in article_results
            ),
            "mean_reciprocal_rank": mean(
                item["reciprocal_rank"] for item in article_results
            ),
            "mean_first_positive_rank_when_present": (
                mean([
                    item["first_positive_rank"]
                    for item in labeled_articles
                    if item["first_positive_rank"] is not None
                ])
                if any(
                    item["first_positive_rank"] is not None
                    for item in labeled_articles
                )
                else None
            ),
        },
        "by_article_length": {
            name: summarize_group(items)
            for name, items in sorted(groups.items())
        },
        "examples_top1_label_zero": missed[:EXAMPLES_PER_CATEGORY],
        "examples_top1_label_one": hits[:EXAMPLES_PER_CATEGORY],
        "interpretation_note": (
            "Labels are heuristic extractive targets, not human annotations. "
            "A label-zero sentence may still be useful in a summary."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nOVERALL RANKING RESULTS")
    print("-" * 55)
    for key, value in report["overall"].items():
        if value is not None:
            print(f"{key}: {value:.4f}")

    print("\nBY ARTICLE LENGTH")
    for name, values in report["by_article_length"].items():
        print(
            f"{name}: articles={values['articles']}, "
            f"top1_hit={values['top1_hit_rate']:.4f}, "
            f"top3_hit={values['top3_hit_rate']:.4f}, "
            f"MRR={values['mean_reciprocal_rank']:.4f}"
        )

    print("\nExamples with top sentence label 0:")
    for item in missed[:3]:
        print(
            f"\nLine {item['source_line']} "
            f"| score={item['top1']['score']:.4f}"
        )
        print(item["top1"]["sentence"])

    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
