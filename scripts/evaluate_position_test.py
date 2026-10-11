import json
import sys
from pathlib import Path
from statistics import mean

import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary
from src.text_encoder import TextEncoder
from src.pbann import PBANNTextScorer
from src.pbann_position import PBANNPositionTextScorer


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
VOCAB_PATH = ROOT / "models" / "vocab.json"

ORIGINAL_CHECKPOINT = (
    ROOT / "models" / "rouge_selection_experiment"
    / "pbann_best_rouge.pt"
)

POSITION_CHECKPOINT = (
    ROOT / "models" / "position_experiment"
    / "pbann_position_best.pt"
)

OUTPUT_PATH = (
    ROOT / "evaluation_results"
    / "position_test_comparison.json"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)

METRICS = ("rouge1", "rouge2", "rougeL")


# ============================================================
# DATA
# ============================================================

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
                    f"Invalid sentences at {path}:{line_number}"
                )

            if not isinstance(reference, str) or not reference.strip():
                continue

            sentences = [
                " ".join(str(sentence).split())
                for sentence in sentences[:MAX_SENTENCES]
                if str(sentence).strip()
            ]

            if not sentences:
                continue

            records.append({
                "sentences": sentences,
                "reference": " ".join(reference.split()),
            })

    if not records:
        raise ValueError(f"No valid records found in {path}")

    return records


def make_positions(sentence_count, device):
    if sentence_count <= 1:
        return torch.zeros(
            (1, sentence_count),
            dtype=torch.float32,
            device=device,
        )

    return torch.linspace(
        0.0,
        1.0,
        sentence_count,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def prepare_inputs(sentences, vocab, device):
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    sentence_count = len(encoded)
    token_count = max(len(ids) for ids in encoded)

    token_ids = torch.zeros(
        (1, sentence_count, token_count),
        dtype=torch.long,
        device=device,
    )

    for index, ids in enumerate(encoded):
        token_ids[0, index, :len(ids)] = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

    token_mask = token_ids != 0

    sentence_mask = torch.ones(
        (1, sentence_count),
        dtype=torch.bool,
        device=device,
    )

    positions = make_positions(sentence_count, device)

    return token_ids, token_mask, sentence_mask, positions


# ============================================================
# MODEL LOADING
# ============================================================

def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


def load_original_model(checkpoint, vocab_size, device):
    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get("embedding_dim", 256),
    ).to(device)

    scorer = PBANNTextScorer(
        embedding_dim=checkpoint.get("embedding_dim", 256),
        num_neurons=checkpoint.get("num_neurons", 512),
        steps=checkpoint.get("recurrent_steps", 6),
        sparsity=checkpoint.get("sparsity", 0.8),
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    encoder.eval()
    scorer.eval()

    return encoder, scorer


def load_position_model(checkpoint, vocab_size, device):
    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=checkpoint.get("embedding_dim", 256),
    ).to(device)

    scorer = PBANNPositionTextScorer(
        embedding_dim=checkpoint.get("embedding_dim", 256),
        num_neurons=checkpoint.get("num_neurons", 512),
        steps=checkpoint.get("recurrent_steps", 6),
        sparsity=checkpoint.get("sparsity", 0.8),
        position_dim=checkpoint.get("position_dim", 32),
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    encoder.eval()
    scorer.eval()

    return encoder, scorer


# ============================================================
# PREDICTIONS AND ROUGE
# ============================================================

def select_top_k(sentences, scores, k=TOP_K):
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    # Extractive summaries preserve the source sentence order.
    selected = sorted(ranked)

    return " ".join(sentences[index] for index in selected)


def score_prediction(prediction, reference):
    scores = ROUGE.score(reference, prediction)

    result = {}

    for metric in METRICS:
        result[metric] = {
            "precision": scores[metric].precision,
            "recall": scores[metric].recall,
            "f1": scores[metric].fmeasure,
        }

    return result


def mean_results(per_article):
    result = {}

    for metric in METRICS:
        result[metric] = {}

        for measure in ("precision", "recall", "f1"):
            result[metric][measure] = mean(
                item[metric][measure]
                for item in per_article
            )

    return result


def evaluate_model(
    name,
    encoder,
    scorer,
    records,
    vocab,
    device,
    position_aware,
):
    per_article = []

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            sentences = record["sentences"]

            token_ids, token_mask, sentence_mask, positions = (
                prepare_inputs(sentences, vocab, device)
            )

            embeddings = encoder(token_ids, token_mask)

            if position_aware:
                logits = scorer(
                    embeddings,
                    sentence_mask,
                    positions,
                )
            else:
                logits = scorer(
                    embeddings,
                    sentence_mask,
                )

            probabilities = torch.sigmoid(
                logits[0]
            ).cpu().tolist()

            prediction = select_top_k(sentences, probabilities)

            per_article.append(
                score_prediction(prediction, record["reference"])
            )

            if index % 100 == 0 or index == len(records):
                print(f"  {name}: {index}/{len(records)}")

    return mean_results(per_article)


def evaluate_lead3(records):
    per_article = []

    for record in records:
        prediction = " ".join(record["sentences"][:TOP_K])

        per_article.append(
            score_prediction(prediction, record["reference"])
        )

    return mean_results(per_article)


# ============================================================
# REPORT
# ============================================================

def print_results(name, results):
    print(f"\n{name}")

    for metric in METRICS:
        values = results[metric]

        print(
            f"  {metric.upper():7s} "
            f"P={values['precision']:.4f} "
            f"R={values['recall']:.4f} "
            f"F1={values['f1']:.4f}"
        )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 68)
    print("HELD-OUT TEST: POSITION-AWARE PB-ANN")
    print("=" * 68)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for path in (
        TEST_PATH,
        VOCAB_PATH,
        ORIGINAL_CHECKPOINT,
        POSITION_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    vocab = Vocabulary.load(VOCAB_PATH)
    records = load_records(TEST_PATH)

    print("Vocabulary size:", len(vocab))
    print("Test articles:", len(records))
    print("Selection method: top-3 sentences")

    original_data = load_checkpoint(
        ORIGINAL_CHECKPOINT, device
    )
    position_data = load_checkpoint(
        POSITION_CHECKPOINT, device
    )

    if original_data["vocab_size"] != len(vocab):
        raise ValueError(
            "Original checkpoint vocabulary size does not match."
        )

    if position_data["vocab_size"] != len(vocab):
        raise ValueError(
            "Position checkpoint vocabulary size does not match."
        )

    original_encoder, original_scorer = load_original_model(
        original_data, len(vocab), device
    )

    position_encoder, position_scorer = load_position_model(
        position_data, len(vocab), device
    )

    print("\nEvaluating original PB-ANN...")
    original_results = evaluate_model(
        "Original PB-ANN",
        original_encoder,
        original_scorer,
        records,
        vocab,
        device,
        position_aware=False,
    )

    print("\nEvaluating position-aware PB-ANN...")
    position_results = evaluate_model(
        "Position-aware PB-ANN",
        position_encoder,
        position_scorer,
        records,
        vocab,
        device,
        position_aware=True,
    )

    print("\nEvaluating Lead-3...")
    lead3_results = evaluate_lead3(records)

    results = {
        "test_articles": len(records),
        "selection": "top_3_source_sentences",
        "original_checkpoint": str(ORIGINAL_CHECKPOINT),
        "position_checkpoint": str(POSITION_CHECKPOINT),
        "original_pbann": original_results,
        "position_aware_pbann": position_results,
        "lead3": lead3_results,
        "position_minus_original_f1": {
            metric: (
                position_results[metric]["f1"]
                - original_results[metric]["f1"]
            )
            for metric in METRICS
        },
        "position_minus_lead3_f1": {
            metric: (
                position_results[metric]["f1"]
                - lead3_results[metric]["f1"]
            )
            for metric in METRICS
        },
    }

    print("\n" + "=" * 68)
    print("HELD-OUT TEST RESULTS")
    print("=" * 68)

    print_results("Original PB-ANN", original_results)
    print_results("Position-aware PB-ANN", position_results)
    print_results("Lead-3 baseline", lead3_results)

    print("\nPosition-aware minus original PB-ANN (F1):")
    for metric, difference in results[
        "position_minus_original_f1"
    ].items():
        print(f"  {metric.upper()}: {difference:+.4f}")

    print("\nPosition-aware minus Lead-3 (F1):")
    for metric, difference in results[
        "position_minus_lead3_f1"
    ].items():
        print(f"  {metric.upper()}: {difference:+.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved results:", OUTPUT_PATH)
    print("Evaluation complete. No models were modified.")


if __name__ == "__main__":
    main()
