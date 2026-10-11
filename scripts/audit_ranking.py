
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary, SummarizationDataset
from src.text_encoder import TextEncoder
from src.pbann import PBANNTextScorer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = ROOT / "data" / "processed" / "validation.jsonl"
CHECKPOINT_PATH = ROOT / "models" / "pbann_phase2_best.pt"
VOCAB_PATH = ROOT / "models" / "vocab.json"
OUTPUT_PATH = ROOT / "evaluation_results" / "ranking_audit.json"

BATCH_SIZE = 2
TOP_K_VALUES = (1, 3)
DECISION_THRESHOLD = 0.75

MAX_SENTENCES = 60
MAX_TOKENS = 64
EMBEDDING_DIM = 256
NUM_NEURONS = 512
RECURRENT_STEPS = 6
SPARSITY = 0.8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_records():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Validation dataset not found: {DATA_PATH}"
        )

    records = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            sentences = record.get("sentences")
            labels = record.get("labels")

            if not isinstance(sentences, list):
                continue

            if not isinstance(labels, list):
                continue

            if len(sentences) != len(labels):
                continue

            clean_sentences = []
            clean_labels = []

            for sentence, label in zip(sentences, labels):
                if not str(sentence).strip():
                    continue

                clean_sentences.append(
                    " ".join(str(sentence).split())
                )
                clean_labels.append(int(label))

            if not clean_sentences:
                continue

            records.append({
                "sentences": clean_sentences[:MAX_SENTENCES],
                "labels": clean_labels[:MAX_SENTENCES],
            })

    if not records:
        raise ValueError(
            "No usable records containing both sentences and labels "
            "were found. Check the processed validation JSONL schema."
        )

    return records


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    for path in (CHECKPOINT_PATH, VOCAB_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    vocab = Vocabulary.load(VOCAB_PATH)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

    embedding_weights = checkpoint[
        "encoder_state_dict"
    ]["embedding.weight"]

    vocab_size = embedding_weights.shape[0]
    embedding_dim = embedding_weights.shape[1]

    if len(vocab) != vocab_size:
        raise ValueError(
            f"Vocabulary mismatch: {len(vocab)} entries, "
            f"checkpoint expects {vocab_size}."
        )

    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    ).to(DEVICE)

    scorer = PBANNTextScorer(
        embedding_dim=embedding_dim,
        num_neurons=checkpoint.get(
            "num_neurons", NUM_NEURONS
        ),
        steps=checkpoint.get(
            "recurrent_steps", RECURRENT_STEPS
        ),
        sparsity=checkpoint.get(
            "sparsity", SPARSITY
        ),
    ).to(DEVICE)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    encoder.eval()
    scorer.eval()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Vocabulary size: {len(vocab)}")

    return vocab, encoder, scorer


# ============================================================
# SENTENCE ENCODING AND SCORING
# ============================================================

def score_record(record, vocab, encoder, scorer):
    sentences = record["sentences"]

    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    longest = max(len(ids) for ids in encoded)

    token_ids = torch.zeros(
        (1, len(encoded), longest),
        dtype=torch.long,
        device=DEVICE,
    )

    for index, ids in enumerate(encoded):
        if ids:
            token_ids[0, index, :len(ids)] = torch.tensor(
                ids,
                dtype=torch.long,
                device=DEVICE,
            )

    token_mask = token_ids != 0

    sentence_mask = torch.ones(
        (1, len(encoded)),
        dtype=torch.bool,
        device=DEVICE,
    )

    with torch.inference_mode():
        embeddings = encoder(token_ids, token_mask)
        logits = scorer(embeddings, sentence_mask)
        scores = torch.sigmoid(logits[0]).cpu().tolist()

    return [
        {
            "index": index,
            "sentence": sentence,
            "label": int(label),
            "score": float(score),
        }
        for index, (sentence, label, score) in enumerate(
            zip(sentences, record["labels"], scores)
        )
    ]


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_top_k_hit_rate(scored_records, k):
    """
    Fraction of articles where at least one of the top-k
    predicted sentences has a positive heuristic label.
    """

    hits = 0

    for items in scored_records:
        top_items = sorted(
            items,
            key=lambda item: item["score"],
            reverse=True,
        )[:k]

        if any(item["label"] == 1 for item in top_items):
            hits += 1

    return hits / max(len(scored_records), 1)


def calculate_positive_rank(scored_records):
    """
    Mean reciprocal rank of the first positive-labelled sentence.
    Articles with no positive labels are excluded.
    """

    reciprocal_ranks = []

    for items in scored_records:
        ranked = sorted(
            items,
            key=lambda item: item["score"],
            reverse=True,
        )

        for rank, item in enumerate(ranked, start=1):
            if item["label"] == 1:
                reciprocal_ranks.append(1.0 / rank)
                break

    return (
        mean(reciprocal_ranks)
        if reciprocal_ranks else 0.0
    )


def calculate_classification_metrics(scored_records):
    labels = []
    predictions = []

    for items in scored_records:
        for item in items:
            labels.append(item["label"])
            predictions.append(
                int(item["score"] >= DECISION_THRESHOLD)
            )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "threshold": DECISION_THRESHOLD,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "actual_positive_fraction": mean(labels),
        "predicted_positive_fraction": mean(predictions),
        "total_sentences": len(labels),
    }


# ============================================================
# LENGTH GROUP ANALYSIS
# ============================================================

def get_length_group(sentence_count):
    if sentence_count <= 10:
        return "short (1-10 sentences)"
    if sentence_count <= 30:
        return "medium (11-30 sentences)"
    return "long (31+ sentences)"


def calculate_length_groups(scored_records):
    groups = defaultdict(list)

    for items in scored_records:
        groups[get_length_group(len(items))].append(items)

    results = {}

    for name, group in groups.items():
        flat_items = [
            item for article in group for item in article
        ]

        labels = [item["label"] for item in flat_items]
        predictions = [
            int(item["score"] >= DECISION_THRESHOLD)
            for item in flat_items
        ]

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )

        results[name] = {
            "articles": len(group),
            "sentences": len(flat_items),
            "positive_label_fraction": mean(labels),
            "predicted_positive_fraction": mean(predictions),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "top1_hit_rate": calculate_top_k_hit_rate(group, 1),
            "top3_hit_rate": calculate_top_k_hit_rate(group, 3),
        }

    return results


# ============================================================
# EXAMPLE COLLECTION
# ============================================================

def collect_examples(scored_records, limit=15):
    missed = []
    successful = []

    for article_index, items in enumerate(scored_records):
        ranked = sorted(
            items,
            key=lambda item: item["score"],
            reverse=True,
        )

        top_item = ranked[0]

        example = {
            "article_index": article_index,
            "sentence_count": len(items),
            "top_sentence": top_item["sentence"],
            "top_score": top_item["score"],
            "top_label": top_item["label"],
            "top3": [
                {
                    "sentence": item["sentence"],
                    "score": item["score"],
                    "label": item["label"],
                }
                for item in ranked[:3]
            ],
        }

        if top_item["label"] == 0:
            missed.append(example)

        if top_item["label"] == 1:
            successful.append(example)

    missed.sort(key=lambda item: item["top_score"], reverse=True)
    successful.sort(
        key=lambda item: item["top_score"], reverse=True
    )

    return {
        "top1_predicted_negative_label": missed[:limit],
        "top1_predicted_positive_label": successful[:limit],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("PB-ANN SENTENCE RANKING AUDIT")
    print("=" * 68)

    records = load_records()
    vocab, encoder, scorer = load_model()

    print(f"Validation articles: {len(records)}")

    scored_records = []

    for index, record in enumerate(records, start=1):
        scored = score_record(record, vocab, encoder, scorer)
        scored_records.append(scored)

        if index % 50 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    all_items = [
        item for article in scored_records for item in article
    ]

    labels = [item["label"] for item in all_items]
    predictions = [
        int(item["score"] >= DECISION_THRESHOLD)
        for item in all_items
    ]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    per_article_positive_counts = [
        sum(item["label"] for item in article)
        for article in scored_records
    ]

    top_k_results = {
        f"top{k}_hit_rate": calculate_top_k_hit_rate(
            scored_records, k
        )
        for k in TOP_K_VALUES
    }

    classification = calculate_classification_metrics(
        scored_records
    )

    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "split": "validation",
        "articles": len(scored_records),
        "sentences": len(all_items),
        "threshold_metrics": classification,
        "ranking_metrics": {
            **top_k_results,
            "mean_reciprocal_rank_first_positive":
                calculate_positive_rank(scored_records),
        },
        "label_statistics": {
            "positive_sentences": sum(labels),
            "negative_sentences": len(labels) - sum(labels),
            "positive_fraction": mean(labels),
            "mean_positive_labels_per_article":
                mean(per_article_positive_counts),
            "positive_label_count_distribution": dict(
                Counter(per_article_positive_counts)
            ),
        },
        "article_length_groups": calculate_length_groups(
            scored_records
        ),
        "examples": collect_examples(scored_records),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 68)
    print("RANKING RESULTS")
    print("=" * 68)

    print(f"Sentence precision: {precision:.4f}")
    print(f"Sentence recall:    {recall:.4f}")
    print(f"Sentence F1:        {f1:.4f}")
    print(
        "Actual positive fraction: "
        f"{classification['actual_positive_fraction']:.4f}"
    )
    print(
        "Predicted positive fraction: "
        f"{classification['predicted_positive_fraction']:.4f}"
    )

    print("\nTOP-K HIT RATES")
    for name, value in top_k_results.items():
        print(f"{name}: {value:.4f}")

    print(
        "Mean reciprocal rank: "
        f"{report['ranking_metrics']['mean_reciprocal_rank_first_positive']:.4f}"
    )

    print("\nLABEL STATISTICS")
    print(
        "Average positive labels per article: "
        f"{mean(per_article_positive_counts):.3f}"
    )
    print(
        "Positive label count distribution: "
        f"{dict(Counter(per_article_positive_counts))}"
    )

    print("\nARTICLE LENGTH GROUPS")
    for name, values in report["article_length_groups"].items():
        print(
            f"{name}: articles={values['articles']}, "
            f"F1={values['f1']:.4f}, "
            f"top1_hit={values['top1_hit_rate']:.4f}, "
            f"top3_hit={values['top3_hit_rate']:.4f}"
        )

    print("\nEXAMPLES: TOP SCORING SENTENCE HAS LABEL 0")
    for item in report["examples"]["top1_predicted_negative_label"][:5]:
        print(
            f"\nArticle {item['article_index']} "
            f"| score={item['top_score']:.4f}"
        )
        print(item["top_sentence"])

    print("\nEXAMPLES: TOP SCORING SENTENCE HAS LABEL 1")
    for item in report["examples"]["top1_predicted_positive_label"][:5]:
        print(
            f"\nArticle {item['article_index']} "
            f"| score={item['top_score']:.4f}"
        )
        print(item["top_sentence"])

    print("\nSaved report:")
    print(OUTPUT_PATH)
    print("\nRanking audit complete.")


if __name__ == "__main__":
    main()
