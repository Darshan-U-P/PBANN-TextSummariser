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
OUTPUT_PATH = (
    ROOT / "evaluation_results" / "budget_test_comparison.json"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3
WORD_BUDGET = 50

METRICS = ("rouge1", "rouge2", "rougeL")

ROUGE = rouge_scorer.RougeScorer(
    list(METRICS),
    use_stemmer=True,
)


def load_test_records():
    path = DATA_DIR / "test.jsonl"

    if not path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)

            sentences = [
                " ".join(str(sentence).split())
                for sentence in item.get("sentences", [])[:MAX_SENTENCES]
                if str(sentence).strip()
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
        raise ValueError("No usable test examples found.")

    return records


def encode_and_rank(sentences, summarizer, vocab):
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    device = summarizer.device
    sentence_count = len(encoded)
    token_count = max(max(map(len, encoded)), 1)

    token_ids = torch.zeros(
        (1, sentence_count, token_count),
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(encoded):
        if ids:
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

    return sorted(
        range(sentence_count),
        key=lambda index: probabilities[index],
        reverse=True,
    )


def select_top_k(sentences, ranked):
    selected = sorted(ranked[:min(TOP_K, len(sentences))])
    return " ".join(sentences[index] for index in selected)


def select_with_budget(sentences, ranked):
    selected = []
    used_words = 0

    for index in ranked:
        sentence_words = len(sentences[index].split())

        # Always include the highest-ranked sentence, even if it
        # exceeds the requested budget.
        if not selected:
            selected.append(index)
            used_words += sentence_words
            continue

        if used_words + sentence_words <= WORD_BUDGET:
            selected.append(index)
            used_words += sentence_words

        if used_words >= WORD_BUDGET:
            break

    # Preserve the original order of the source article.
    selected.sort()
    return " ".join(sentences[index] for index in selected)


def score_prediction(prediction, reference):
    values = ROUGE.score(reference, prediction)

    return {
        metric: {
            "precision": values[metric].precision,
            "recall": values[metric].recall,
            "f1": values[metric].fmeasure,
        }
        for metric in METRICS
    }


def average_scores(rows):
    return {
        metric: {
            measure: mean(
                row[metric][measure] for row in rows
            )
            for measure in ("precision", "recall", "f1")
        }
        for metric in METRICS
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

    records = load_test_records()

    methods = ("pbann_top3", "pbann_budget50", "lead3")
    all_scores = {method: [] for method in methods}
    word_lengths = {method: [] for method in methods}

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {summarizer.device}")
    print(f"Test examples: {len(records)}")
    print("Evaluating the held-out test set; no training or tuning.")

    for number, record in enumerate(records, 1):
        sentences = record["sentences"]
        reference = record["reference"]

        ranked = encode_and_rank(sentences, summarizer, vocab)

        predictions = {
            "pbann_top3": select_top_k(sentences, ranked),
            "pbann_budget50": select_with_budget(sentences, ranked),
            "lead3": " ".join(sentences[:TOP_K]),
        }

        for method, prediction in predictions.items():
            all_scores[method].append(
                score_prediction(prediction, reference)
            )
            word_lengths[method].append(len(prediction.split()))

        if number % 100 == 0 or number == len(records):
            print(f"Processed {number}/{len(records)}")

    results = {
        method: {
            "rouge": average_scores(all_scores[method]),
            "average_summary_words": mean(word_lengths[method]),
        }
        for method in methods
    }

    output = {
        "split": "test",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "examples": len(records),
        "selection": {
            "top_k": TOP_K,
            "word_budget": WORD_BUDGET,
            "max_sentences_per_article": MAX_SENTENCES,
            "max_tokens_per_sentence": MAX_TOKENS,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTEST RESULTS")
    print("=" * 100)

    for method in methods:
        result = results[method]
        print(f"\n{method} | Average words: "
              f"{result['average_summary_words']:.2f}")

        for metric in METRICS:
            values = result["rouge"][metric]
            print(
                f"  {metric:7s} "
                f"P={values['precision']:.4f} "
                f"R={values['recall']:.4f} "
                f"F1={values['f1']:.4f}"
            )

    print("\nSaved results to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
