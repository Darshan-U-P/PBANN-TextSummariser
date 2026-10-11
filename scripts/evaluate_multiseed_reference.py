
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rouge_score import rouge_scorer

from src.dataset import Vocabulary, SummarizationDataset, create_dataloader
from scripts.train_position_multiseed import make_model, get_logits


# ============================================================
# CONFIGURATION
# ============================================================

VAL_PATH = ROOT / "data/processed/validation.jsonl"
VOCAB_PATH = ROOT / "models/vocab.json"
CHECKPOINT_DIR = ROOT / "models/multiseed"
OUTPUT_DIR = ROOT / "evaluation_results/multiseed"

SEEDS = [42, 123, 2026]
MODEL_TYPES = ["no_position", "position"]

BATCH_SIZE = 2
MAX_SENTENCES = 60
MAX_TOKENS = 64
SUMMARY_SENTENCES = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


# ============================================================
# LOAD ORIGINAL VALIDATION RECORDS
# ============================================================

def load_reference_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)

            if not record.get("summary", "").strip():
                raise ValueError(
                    f"Missing original summary at line {line_number}"
                )

            records.append({
                "sentences": record["sentences"][:MAX_SENTENCES],
                "summary": record["summary"].strip(),
            })

    return records


# ============================================================
# EVALUATION AGAINST ORIGINAL SUMMARIES
# ============================================================

def evaluate_checkpoint(model_type, seed, vocab, records, loader):
    checkpoint_path = (
        CHECKPOINT_DIR / f"{model_type}_seed_{seed}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    encoder, scorer = make_model(model_type, len(vocab))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    encoder.eval()
    scorer.eval()

    article_scores = []
    record_index = 0

    with torch.no_grad():
        for batch in loader:
            logits = get_logits(encoder, scorer, batch)

            for i in range(logits.size(0)):
                record = records[record_index]
                record_index += 1

                sentences = record["sentences"]
                reference = record["summary"]

                valid_count = min(
                    len(sentences),
                    int(batch["sentence_mask"][i].sum().item()),
                )

                if valid_count == 0:
                    continue

                count = min(SUMMARY_SENTENCES, valid_count)

                selected = torch.topk(
                    logits[i, :valid_count],
                    k=count,
                ).indices.tolist()

                # Keep selected sentences in original article order.
                selected.sort()

                prediction = " ".join(
                    sentences[j] for j in selected
                )

                scores = ROUGE.score(reference, prediction)

                article_scores.append({
                    "rouge1": scores["rouge1"].fmeasure,
                    "rouge2": scores["rouge2"].fmeasure,
                    "rougeL": scores["rougeL"].fmeasure,
                })

    if not article_scores:
        raise RuntimeError(
            f"No articles evaluated for {model_type}, seed {seed}"
        )

    metrics = {
        metric: float(np.mean([
            item[metric] for item in article_scores
        ]))
        for metric in ("rouge1", "rouge2", "rougeL")
    }

    return {
        "model": model_type,
        "seed": seed,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "articles_evaluated": len(article_scores),
        **metrics,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("MULTI-SEED EVALUATION AGAINST ORIGINAL REFERENCE SUMMARIES")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    print("Dataset: validation only")
    print("Held-out test set: NOT USED")
    print("Checkpoints: loaded without modification")

    for path in (VAL_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    vocab = Vocabulary.load(VOCAB_PATH)
    records = load_reference_records(VAL_PATH)

    dataset = SummarizationDataset(
        VAL_PATH,
        vocab,
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
    )

    loader = create_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    results = []

    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            print(f"\nEvaluating {model_type}, seed {seed}...")

            result = evaluate_checkpoint(
                model_type=model_type,
                seed=seed,
                vocab=vocab,
                records=records,
                loader=loader,
            )

            results.append(result)

            print(
                f"ROUGE-1={result['rouge1']:.4f} | "
                f"ROUGE-2={result['rouge2']:.4f} | "
                f"ROUGE-L={result['rougeL']:.4f}"
            )

    summary = {}

    for model_type in MODEL_TYPES:
        subset = [
            row for row in results
            if row["model"] == model_type
        ]

        summary[model_type] = {}

        for metric in ("rouge1", "rouge2", "rougeL"):
            values = np.array([row[metric] for row in subset])

            summary[model_type][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "per_seed": values.tolist(),
            }

    output = {
        "evaluation": {
            "reference": "original summary field from validation JSONL",
            "selection": f"top {SUMMARY_SENTENCES} sentences",
            "validation_articles": len(records),
            "test_set_used": False,
        },
        "runs": results,
        "summary": summary,
    }

    output_path = (
        OUTPUT_DIR / "reference_summary_evaluation.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("\n" + "=" * 72)
    print("FINAL REFERENCE-SUMMARY RESULTS")
    print("=" * 72)
    print(f"{'Model':<16} {'Metric':<10} {'Mean':>10} {'Std':>10}")

    for model_type in MODEL_TYPES:
        for metric in ("rouge1", "rouge2", "rougeL"):
            values = summary[model_type][metric]

            print(
                f"{model_type:<16} {metric:<10} "
                f"{values['mean']:>10.4f} "
                f"{values['std']:>10.4f}"
            )

    print(f"\nSaved: {output_path}")
    print("No model checkpoints were modified.")
    print("Held-out test set was not used.")


if __name__ == "__main__":
    main()
