
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
    ROOT / "evaluation_results" / "length_budget_validation.json"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3

# Compare several budgets rather than assuming 35 words is optimal.
WORD_BUDGETS = [30, 35, 40, 45, 50]

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
        raise ValueError("No usable validation examples found.")

    return records


def rouge_f1(prediction, reference):
    scores = ROUGE.score(reference, prediction)
    return {
        metric: scores[metric].fmeasure
        for metric in ("rouge1", "rouge2", "rougeL")
    }


def encode_and_score(sentences, summarizer, vocab):
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    device = summarizer.device
    count = len(encoded)
    token_count = max(map(len, encoded))

    token_ids = torch.zeros(
        (1, count, token_count),
        dtype=torch.long,
        device=device,
    )

    for i, ids in enumerate(encoded):
        token_ids[0, i, :len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=device
        )

    token_mask = token_ids != 0
    sentence_mask = torch.ones(
        (1, count), dtype=torch.bool, device=device
    )

    with torch.inference_mode():
        embeddings = summarizer.encoder(token_ids, token_mask)
        logits = summarizer.scorer(embeddings, sentence_mask)
        probabilities = torch.sigmoid(logits[0]).cpu().tolist()

    return sorted(
        range(count),
        key=lambda i: probabilities[i],
        reverse=True,
    )


def select_top_k(sentences, ranked):
    indices = sorted(ranked[:min(TOP_K, len(sentences))])
    return " ".join(sentences[i] for i in indices)


def select_with_budget(sentences, ranked, budget):
    selected = []
    used_words = 0

    # Greedy selection in score order. Always consider the best sentence,
    # even if it exceeds the budget, to avoid returning an empty summary.
    for index in ranked:
        sentence_words = len(sentences[index].split())

        if not selected:
            selected.append(index)
            used_words += sentence_words
            continue

        if used_words + sentence_words <= budget:
            selected.append(index)
            used_words += sentence_words

        if used_words >= budget:
            break

    # Restore the original article order.
    selected.sort()
    return " ".join(sentences[i] for i in selected)


def average_scores(rows):
    return {
        metric: mean(row[metric] for row in rows)
        for metric in ("rouge1", "rouge2", "rougeL")
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
    methods = ["pbann_top3", "lead3"] + [
        f"budget_{budget}" for budget in WORD_BUDGETS
    ]
    scores = {method: [] for method in methods}
    word_lengths = {method: [] for method in methods}

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {summarizer.device}")
    print(f"Validation examples: {len(records)}")

    for example_number, record in enumerate(records, 1):
        sentences = record["sentences"]
        reference = record["reference"]
        ranked = encode_and_score(sentences, summarizer, vocab)

        predictions = {
            "pbann_top3": select_top_k(sentences, ranked),
            "lead3": " ".join(sentences[:TOP_K]),
        }

        for budget in WORD_BUDGETS:
            predictions[f"budget_{budget}"] = select_with_budget(
                sentences, ranked, budget
            )

        for method, prediction in predictions.items():
            scores[method].append(
                rouge_f1(prediction, reference)
            )
            word_lengths[method].append(
                len(prediction.split())
            )

        if example_number % 100 == 0 or example_number == len(records):
            print(f"Processed {example_number}/{len(records)}")

    results = {
        method: {
            "rouge_f1": average_scores(scores[method]),
            "average_summary_words": mean(word_lengths[method]),
        }
        for method in methods
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({
            "split": "validation",
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "examples": len(records),
            "word_budgets": WORD_BUDGETS,
            "results": results,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nVALIDATION RESULTS")
    print("-" * 85)
    print(
        f"{'Method':<15} {'Words':>8} "
        f"{'ROUGE-1 F1':>13} {'ROUGE-2 F1':>13} {'ROUGE-L F1':>13}"
    )

    for method, result in results.items():
        metrics = result["rouge_f1"]
        print(
            f"{method:<15} "
            f"{result['average_summary_words']:>8.2f} "
            f"{metrics['rouge1']:>13.4f} "
            f"{metrics['rouge2']:>13.4f} "
            f"{metrics['rougeL']:>13.4f}"
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
