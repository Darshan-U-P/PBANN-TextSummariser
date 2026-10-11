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
OUTPUT_PATH = ROOT / "evaluation_results" / "position_bias_audit.json"

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3

BANDS = ("beginning", "middle", "end")


def get_band(index, sentence_count):
    relative_position = index / sentence_count

    if relative_position < 0.20:
        return "beginning"
    if relative_position >= 0.80:
        return "end"
    return "middle"


def get_length_group(sentence_count):
    if sentence_count <= 10:
        return "short_1_10"
    if sentence_count <= 30:
        return "medium_11_30"
    return "long_31_plus"


def load_records():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
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

            pairs = [
                (" ".join(str(sentence).split()), int(label))
                for sentence, label in zip(sentences, labels)
                if str(sentence).strip()
            ][:MAX_SENTENCES]

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


def score_sentences(sentences, summarizer, vocab):
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

    return scores


def safe_mean(values):
    return mean(values) if values else None


def summarize_sentences(items):
    positive_scores = [
        item["score"] for item in items if item["label"] == 1
    ]
    negative_scores = [
        item["score"] for item in items if item["label"] == 0
    ]

    return {
        "sentence_count": len(items),
        "positive_label_fraction": safe_mean(
            [item["label"] for item in items]
        ),
        "mean_score_all": safe_mean(
            [item["score"] for item in items]
        ),
        "mean_score_positive_labels": safe_mean(positive_scores),
        "mean_score_negative_labels": safe_mean(negative_scores),
        "positive_minus_negative_mean_score": (
            safe_mean(positive_scores) - safe_mean(negative_scores)
            if positive_scores and negative_scores
            else None
        ),
        "top3_selection_rate": (
            sum(item["selected_top3"] for item in items) / len(items)
            if items else None
        ),
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    vocab = Vocabulary.load(VOCAB_PATH)
    summarizer = PBANNSummarizer(
        checkpoint_path=str(CHECKPOINT),
        vocab_path=str(VOCAB_PATH),
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
        max_summary_sentences=TOP_K,
    )

    records = load_records()

    # Aggregate sentence-level observations across articles.
    overall_bands = defaultdict(list)
    length_bands = defaultdict(lambda: defaultdict(list))
    examples = []

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {summarizer.device}")
    print(f"Validation articles: {len(records)}")
    print("Analysing sentence position and model scores...")

    for number, record in enumerate(records, start=1):
        sentences = record["sentences"]
        labels = record["labels"]
        scores = score_sentences(sentences, summarizer, vocab)

        ranked = sorted(
            range(len(sentences)),
            key=lambda index: scores[index],
            reverse=True,
        )
        selected_top3 = set(ranked[:min(TOP_K, len(sentences))])
        group = get_length_group(len(sentences))

        article_items = []

        for index, (sentence, label, score) in enumerate(
            zip(sentences, labels, scores)
        ):
            band = get_band(index, len(sentences))
            item = {
                "article_line": record["source_line"],
                "index": index,
                "band": band,
                "label": label,
                "score": float(score),
                "selected_top3": int(index in selected_top3),
                "sentence": sentence,
            }

            overall_bands[band].append(item)
            length_bands[group][band].append(item)
            article_items.append(item)

        # Keep a few examples illustrating score and label patterns.
        if len(examples) < 12:
            examples.append({
                "source_line": record["source_line"],
                "sentence_count": len(sentences),
                "length_group": group,
                "top3": [
                    {
                        "index": index,
                        "band": get_band(index, len(sentences)),
                        "score": float(scores[index]),
                        "label": labels[index],
                        "sentence": sentences[index],
                    }
                    for index in ranked[:3]
                ],
            })

        if number % 100 == 0 or number == len(records):
            print(f"Processed {number}/{len(records)}")

    overall_report = {
        band: summarize_sentences(overall_bands[band])
        for band in BANDS
    }

    length_report = {
        group: {
            band: summarize_sentences(length_bands[group][band])
            for band in BANDS
        }
        for group in ("short_1_10", "medium_11_30", "long_31_plus")
        if group in length_bands
    }

    report = {
        "split": "validation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "articles": len(records),
        "position_band_definition": {
            "beginning": "first 20% of sentence indices",
            "middle": "20% to less than 80%",
            "end": "final 20%",
        },
        "overall_by_position": overall_report,
        "by_article_length_and_position": length_report,
        "examples": examples,
        "interpretation_note": (
            "Labels are heuristic targets, not human annotations. "
            "Score differences are descriptive and do not prove causation."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPOSITION BIAS RESULTS")
    print("-" * 100)
    print(
        f"{'Band':<12} {'Sentences':>10} {'Positive %':>12} "
        f"{'Mean score':>12} {'Pos score':>12} "
        f"{'Neg score':>12} {'Top-3 rate':>12}"
    )

    for band, result in overall_report.items():
        def fmt(value):
            return f"{value:.4f}" if value is not None else "N/A"

        print(
            f"{band:<12} "
            f"{result['sentence_count']:>10} "
            f"{fmt(result['positive_label_fraction']):>12} "
            f"{fmt(result['mean_score_all']):>12} "
            f"{fmt(result['mean_score_positive_labels']):>12} "
            f"{fmt(result['mean_score_negative_labels']):>12} "
            f"{fmt(result['top3_selection_rate']):>12}"
        )

    print("\nLONG ARTICLES: POSITION BREAKDOWN")
    for band, result in length_report.get(
        "long_31_plus", {}
    ).items():
        print(
            f"{band}: "
            f"positive_fraction={result['positive_label_fraction']}, "
            f"mean_score={result['mean_score_all']}, "
            f"top3_rate={result['top3_selection_rate']}"
        )

    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
