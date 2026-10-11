
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
VOCAB = ROOT / "models" / "vocab.json"
OUTPUT = ROOT / "evaluation_results" / "rouge_checkpoint_comparison.json"

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_records(split):
    path = DATA_DIR / f"{split}.jsonl"
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            sentences = item.get("sentences", [])
            reference = item.get("summary", "")

            if not sentences or not reference.strip():
                continue

            records.append({
                "sentences": [
                    " ".join(str(s).split())
                    for s in sentences[:MAX_SENTENCES]
                    if str(s).strip()
                ],
                "reference": " ".join(reference.split()),
            })

    return records


def score(prediction, reference):
    values = ROUGE.score(reference, prediction)
    return {
        key: values[key].fmeasure
        for key in ("rouge1", "rouge2", "rougeL")
    }


def average(items):
    return {
        key: mean(item[key] for item in items)
        for key in ("rouge1", "rouge2", "rougeL")
    }


def evaluate(split, summarizer, vocab):
    records = load_records(split)
    model_scores, lead_scores = [], []

    print(f"\nEvaluating {split}: {len(records)} examples")

    for index, record in enumerate(records, 1):
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
            (1, sentence_count), dtype=torch.bool, device=device
        )

        with torch.inference_mode():
            embeddings = summarizer.encoder(token_ids, token_mask)
            logits = summarizer.scorer(embeddings, sentence_mask)
            probabilities = torch.sigmoid(logits[0]).cpu().tolist()

        selected = sorted(
            sorted(
                range(sentence_count),
                key=lambda i: probabilities[i],
                reverse=True,
            )[:TOP_K]
        )
        prediction = " ".join(sentences[i] for i in selected)
        lead3 = " ".join(sentences[:TOP_K])

        model_scores.append(score(prediction, record["reference"]))
        lead_scores.append(score(lead3, record["reference"]))

        if index % 100 == 0 or index == len(records):
            print(f"Processed {index}/{len(records)}")

    return {
        "split": split,
        "examples": len(records),
        "pbann_top3": average(model_scores),
        "lead3": average(lead_scores),
        "difference_pbann_minus_lead3": {
            key: average(model_scores)[key] - average(lead_scores)[key]
            for key in ("rouge1", "rouge2", "rougeL")
        },
    }


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    vocab = Vocabulary.load(VOCAB)
    summarizer = PBANNSummarizer(
        checkpoint_path=str(CHECKPOINT),
        vocab_path=str(VOCAB),
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
        max_summary_sentences=TOP_K,
    )

    results = {
        split: evaluate(split, summarizer, vocab)
        for split in ("validation", "test")
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    for split, result in results.items():
        print(f"\n{split.upper()} — ROUGE F1")
        for method in ("pbann_top3", "lead3"):
            values = result[method]
            print(
                f"{method:12s} "
                f"R1={values['rouge1']:.4f} "
                f"R2={values['rouge2']:.4f} "
                f"RL={values['rougeL']:.4f}"
            )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
