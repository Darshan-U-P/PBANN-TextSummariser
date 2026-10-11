
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
OUTPUT_PATH = (
    ROOT / "evaluation_results/multiseed/position_vs_lead3_bootstrap.json"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 2
MAX_SENTENCES = 60
MAX_TOKENS = 64
SUMMARY_SENTENCES = 3
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261012

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)

METRICS = ["rouge1", "rouge2", "rougeL"]


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
            sentences = record["sentences"][:MAX_SENTENCES]
            reference = record.get("summary", "").strip()

            if not reference:
                raise ValueError(
                    f"Missing original summary at line {line_number}"
                )

            records.append({
                "sentences": sentences,
                "reference": reference,
            })

    return records


def score_texts(prediction, reference):
    result = ROUGE.score(reference, prediction)

    return {
        "rouge1": result["rouge1"].fmeasure,
        "rouge2": result["rouge2"].fmeasure,
        "rougeL": result["rougeL"].fmeasure,
    }


# ============================================================
# LEAD-3 BASELINE
# ============================================================

def evaluate_lead3(records):
    scores = []

    for record in records:
        prediction = " ".join(
            record["sentences"][:SUMMARY_SENTENCES]
        )

        scores.append(
            score_texts(prediction, record["reference"])
        )

    return scores


# ============================================================
# POSITION-AWARE MODEL
# ============================================================

def evaluate_position_checkpoint(seed, vocab, records, loader):
    checkpoint_path = (
        CHECKPOINT_DIR / f"position_seed_{seed}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    encoder, scorer = make_model("position", len(vocab))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])

    encoder.eval()
    scorer.eval()

    scores = []
    record_index = 0

    with torch.no_grad():
        for batch in loader:
            logits = get_logits(encoder, scorer, batch)

            for i in range(logits.size(0)):
                record = records[record_index]
                record_index += 1

                sentences = record["sentences"]
                valid_count = min(
                    len(sentences),
                    int(batch["sentence_mask"][i].sum().item()),
                )

                if valid_count == 0:
                    prediction = ""
                else:
                    count = min(SUMMARY_SENTENCES, valid_count)

                    selected = torch.topk(
                        logits[i, :valid_count],
                        k=count,
                    ).indices.tolist()

                    # Preserve original source order.
                    selected.sort()

                    prediction = " ".join(
                        sentences[j] for j in selected
                    )

                scores.append(
                    score_texts(prediction, record["reference"])
                )

    if len(scores) != len(records):
        raise RuntimeError(
            f"Article count mismatch for seed {seed}: "
            f"{len(scores)} scores vs {len(records)} records"
        )

    return scores


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(position_scores, baseline_scores, rng):
    if len(position_scores) != len(baseline_scores):
        raise ValueError("Paired score arrays must have equal length.")

    results = {}

    n = len(position_scores)

    for metric in METRICS:
        position_values = np.array(
            [row[metric] for row in position_scores],
            dtype=np.float64,
        )

        baseline_values = np.array(
            [row[metric] for row in baseline_scores],
            dtype=np.float64,
        )

        # Positive difference means position-aware PB-ANN wins.
        differences = position_values - baseline_values
        observed_difference = float(differences.mean())

        bootstrap_means = np.empty(
            BOOTSTRAP_RESAMPLES,
            dtype=np.float64,
        )

        # Resample article indices, preserving paired differences.
        for start in range(0, BOOTSTRAP_RESAMPLES, 250):
            size = min(250, BOOTSTRAP_RESAMPLES - start)

            indices = rng.integers(
                0, n, size=(size, n)
            )

            bootstrap_means[start:start + size] = (
                differences[indices].mean(axis=1)
            )

        low, high = np.percentile(
            bootstrap_means, [2.5, 97.5]
        )

        results[metric] = {
            "position_mean": float(position_values.mean()),
            "lead3_mean": float(baseline_values.mean()),
            "difference_position_minus_lead3": observed_difference,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "significant_at_95_percent": bool(
                low > 0 or high < 0
            ),
        }

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("PAIRED BOOTSTRAP: POSITION-AWARE PB-ANN VS LEAD-3")
    print("=" * 76)
    print(f"Device: {DEVICE}")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}")
    print("Dataset: validation only")
    print("Held-out test set: NOT USED")

    for path in (VAL_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    records = load_records(VAL_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    dataset = SummarizationDataset(
        VAL_PATH,
        vocab,
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
    )

    if len(dataset) != len(records):
        raise RuntimeError(
            "Dataset and reference records differ in length."
        )

    loader = create_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    lead3_scores = evaluate_lead3(records)

    output = {
        "validation_articles": len(records),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "comparison": "position-aware PB-ANN minus Lead-3",
        "runs": {},
    }

    for seed in SEEDS:
        print(f"\nEvaluating position-aware checkpoint, seed {seed}...")

        position_scores = evaluate_position_checkpoint(
            seed, vocab, records, loader
        )

        rng = np.random.default_rng(BOOTSTRAP_SEED + seed)

        comparison = paired_bootstrap(
            position_scores,
            lead3_scores,
            rng,
        )

        output["runs"][str(seed)] = comparison

        for metric in METRICS:
            result = comparison[metric]
            status = (
                "CI EXCLUDES ZERO"
                if result["significant_at_95_percent"]
                else "CI INCLUDES ZERO"
            )

            print(
                f"{metric}: "
                f"PB-ANN={result['position_mean']:.4f}, "
                f"Lead-3={result['lead3_mean']:.4f}, "
                f"diff={result['difference_position_minus_lead3']:+.4f}, "
                f"95% CI=[{result['ci95_low']:+.4f}, "
                f"{result['ci95_high']:+.4f}] "
                f"({status})"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nSaved: {OUTPUT_PATH}")
    print("No checkpoints were modified.")
    print("Held-out test set was not used.")


if __name__ == "__main__":
    main()
