import json
import sys
from pathlib import Path

import numpy as np
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
    / "position_bootstrap_test.json"
)

MAX_SENTENCES = 60
MAX_TOKENS = 64
TOP_K = 3

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 2026

METRICS = ("rouge1", "rouge2", "rougeL")
ROUGE = rouge_scorer.RougeScorer(
    list(METRICS),
    use_stemmer=True,
)


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
        raise ValueError(f"No valid records in {path}")

    return records


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
            ids, dtype=torch.long, device=device
        )

    token_mask = token_ids != 0
    sentence_mask = torch.ones(
        (1, sentence_count),
        dtype=torch.bool,
        device=device,
    )

    if sentence_count > 1:
        positions = torch.linspace(
            0.0, 1.0, sentence_count,
            dtype=torch.float32, device=device,
        ).unsqueeze(0)
    else:
        positions = torch.zeros(
            (1, sentence_count),
            dtype=torch.float32,
            device=device,
        )

    return token_ids, token_mask, sentence_mask, positions


# ============================================================
# CHECKPOINTS
# ============================================================

def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


def load_original_model(checkpoint, vocab_size, device):
    dim = checkpoint.get("embedding_dim", 256)

    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=dim,
    ).to(device)

    scorer = PBANNTextScorer(
        embedding_dim=dim,
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
    dim = checkpoint.get("embedding_dim", 256)

    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=dim,
    ).to(device)

    scorer = PBANNPositionTextScorer(
        embedding_dim=dim,
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
# PER-ARTICLE METRICS
# ============================================================

def make_prediction(sentences, probabilities):
    ranked = sorted(
        range(len(sentences)),
        key=lambda i: probabilities[i],
        reverse=True,
    )[:TOP_K]

    selected = sorted(ranked)
    return " ".join(sentences[i] for i in selected)


def calculate_f1(prediction, reference):
    scores = ROUGE.score(reference, prediction)
    return np.array(
        [scores[m].fmeasure for m in METRICS],
        dtype=np.float64,
    )


def evaluate_model(records, vocab, encoder, scorer, device, use_position):
    all_scores = []

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            sentences = record["sentences"]

            token_ids, token_mask, sentence_mask, positions = (
                prepare_inputs(sentences, vocab, device)
            )

            embeddings = encoder(token_ids, token_mask)

            if use_position:
                logits = scorer(
                    embeddings, sentence_mask, positions
                )
            else:
                logits = scorer(embeddings, sentence_mask)

            probabilities = torch.sigmoid(
                logits[0]
            ).cpu().tolist()

            prediction = make_prediction(
                sentences, probabilities
            )

            all_scores.append(
                calculate_f1(prediction, record["reference"])
            )

            if index % 100 == 0 or index == len(records):
                print(f"  Scored {index}/{len(records)} articles")

    return np.stack(all_scores, axis=0)


def evaluate_lead3(records):
    rows = []

    for record in records:
        prediction = " ".join(record["sentences"][:TOP_K])
        rows.append(
            calculate_f1(prediction, record["reference"])
        )

    return np.stack(rows, axis=0)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(candidate, baseline, rng):
    """
    Resample article indices jointly for both methods.

    Returns the observed mean difference, 95% percentile CI,
    and bootstrap probability that the difference is positive.
    """
    if candidate.shape != baseline.shape:
        raise ValueError("Paired score arrays must have equal shapes.")

    differences = candidate - baseline
    article_count, metric_count = differences.shape

    observed = differences.mean(axis=0)
    bootstrap_means = np.empty(
        (N_BOOTSTRAP, metric_count),
        dtype=np.float64,
    )

    # Batch resampling to keep memory use moderate.
    batch_size = 250

    for start in range(0, N_BOOTSTRAP, batch_size):
        end = min(start + batch_size, N_BOOTSTRAP)

        indices = rng.integers(
            0,
            article_count,
            size=(end - start, article_count),
        )

        sampled = differences[indices]
        bootstrap_means[start:end] = sampled.mean(axis=1)

    lower = np.percentile(bootstrap_means, 2.5, axis=0)
    upper = np.percentile(bootstrap_means, 97.5, axis=0)
    probability_positive = (
        bootstrap_means > 0
    ).mean(axis=0)

    results = {}

    for i, metric in enumerate(METRICS):
        results[metric] = {
            "observed_mean_f1_difference": float(observed[i]),
            "ci95_lower": float(lower[i]),
            "ci95_upper": float(upper[i]),
            "bootstrap_fraction_difference_positive": float(
                probability_positive[i]
            ),
            "ci_excludes_zero": bool(
                lower[i] > 0 or upper[i] < 0
            ),
        }

    return results


def summarize_mean_f1(scores):
    return {
        metric: float(scores[:, i].mean())
        for i, metric in enumerate(METRICS)
    }


def print_comparison(name, result):
    print(f"\nPosition-aware PB-ANN minus {name}")

    for metric in METRICS:
        item = result[metric]
        print(
            f"  {metric.upper():7s} "
            f"delta={item['observed_mean_f1_difference']:+.4f} "
            f"95% CI=[{item['ci95_lower']:+.4f}, "
            f"{item['ci95_upper']:+.4f}] "
            f"bootstrap P(delta > 0)="
            f"{item['bootstrap_fraction_difference_positive']:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for path in (
        TEST_PATH,
        VOCAB_PATH,
        ORIGINAL_CHECKPOINT,
        POSITION_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    print("=" * 68)
    print("PAIRED BOOTSTRAP: HELD-OUT TEST SET")
    print("=" * 68)
    print("Device:", device)
    print("Bootstrap resamples:", f"{N_BOOTSTRAP:,}")

    vocab = Vocabulary.load(VOCAB_PATH)
    records = load_records(TEST_PATH)

    original_data = load_checkpoint(ORIGINAL_CHECKPOINT, device)
    position_data = load_checkpoint(POSITION_CHECKPOINT, device)

    if original_data["vocab_size"] != len(vocab):
        raise ValueError("Original checkpoint vocabulary mismatch.")

    if position_data["vocab_size"] != len(vocab):
        raise ValueError("Position checkpoint vocabulary mismatch.")

    original_encoder, original_scorer = load_original_model(
        original_data, len(vocab), device
    )

    position_encoder, position_scorer = load_position_model(
        position_data, len(vocab), device
    )

    print(f"Test articles: {len(records)}")

    print("\nScoring original PB-ANN...")
    original_scores = evaluate_model(
        records, vocab,
        original_encoder, original_scorer,
        device, use_position=False,
    )

    print("\nScoring position-aware PB-ANN...")
    position_scores = evaluate_model(
        records, vocab,
        position_encoder, position_scorer,
        device, use_position=True,
    )

    print("\nScoring Lead-3...")
    lead3_scores = evaluate_lead3(records)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    original_comparison = paired_bootstrap(
        position_scores, original_scores, rng
    )
    lead3_comparison = paired_bootstrap(
        position_scores, lead3_scores, rng
    )

    output = {
        "test_articles": len(records),
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metric": "mean per-article ROUGE F1",
        "selection": "top-3 source sentences",
        "mean_f1": {
            "original_pbann": summarize_mean_f1(original_scores),
            "position_aware_pbann": summarize_mean_f1(position_scores),
            "lead3": summarize_mean_f1(lead3_scores),
        },
        "paired_comparisons": {
            "position_minus_original": original_comparison,
            "position_minus_lead3": lead3_comparison,
        },
        "interpretation_note": (
            "A 95% paired bootstrap percentile interval excluding zero "
            "indicates evidence of a difference under this resampling "
            "procedure; it is not a guarantee of generalization to "
            "other datasets."
        ),
    }

    print("\n" + "=" * 68)
    print("PAIRED BOOTSTRAP RESULTS")
    print("=" * 68)

    print_comparison("original PB-ANN", original_comparison)
    print_comparison("Lead-3", lead3_comparison)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\nSaved:", OUTPUT_PATH)
    print("No models were retrained or modified.")


if __name__ == "__main__":
    main()
