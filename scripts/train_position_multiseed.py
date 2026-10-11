
import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT — MUST COME BEFORE PROJECT IMPORTS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import random
import inspect

import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer

from src.dataset import Vocabulary, SummarizationDataset, create_dataloader
from src.text_encoder import TextEncoder
from src.pbann_position import PBANNPositionTextScorer
from scripts.train_position_ablation import PBANNNoPositionTextScorer


# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2026]

TRAIN_PATH = ROOT / "data/processed/train.jsonl"
VAL_PATH = ROOT / "data/processed/validation.jsonl"
VOCAB_PATH = ROOT / "models/vocab.json"

OUTPUT_DIR = ROOT / "evaluation_results/multiseed"
CHECKPOINT_DIR = ROOT / "models/multiseed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

EMBEDDING_DIM = 256
NUM_NEURONS = 512
STEPS = 6
SPARSITY = 0.8
POSITION_DIM = 32

MAX_SENTENCES = 60
MAX_TOKENS = 64
SUMMARY_SENTENCES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA
# ============================================================

def read_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            sentences = record["sentences"][:MAX_SENTENCES]
            labels = record["labels"][:MAX_SENTENCES]

            if len(sentences) != len(labels):
                raise ValueError(
                    f"Sentence/label mismatch in {path}, "
                    f"line {line_number}"
                )

            records.append({
                "sentences": sentences,
                "labels": labels,
            })

    return records


def make_model(model_type, vocab_size):
    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
    )

    scorer_class = (
        PBANNPositionTextScorer
        if model_type == "position"
        else PBANNNoPositionTextScorer
    )

    parameters = inspect.signature(scorer_class.__init__).parameters

    candidates = {
        "embedding_dim": EMBEDDING_DIM,
        "num_neurons": NUM_NEURONS,
        "steps": STEPS,
        "num_steps": STEPS,
        "sparsity": SPARSITY,
        "position_dim": POSITION_DIM,
        "pos_dim": POSITION_DIM,
    }

    kwargs = {
        key: value
        for key, value in candidates.items()
        if key in parameters
    }

    scorer = scorer_class(**kwargs)

    return encoder.to(DEVICE), scorer.to(DEVICE)


# ============================================================
# MODEL FORWARD
# ============================================================

def get_logits(encoder, scorer, batch):
    token_ids = batch["token_ids"].to(DEVICE)
    token_mask = batch["token_mask"].to(DEVICE)
    sentence_mask = batch["sentence_mask"].to(DEVICE)

    sentence_embeddings = encoder(token_ids, token_mask)
    logits = scorer(sentence_embeddings, sentence_mask)

    return logits


# ============================================================
# ROUGE
# ============================================================

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def calculate_rouge(prediction, reference):
    scores = ROUGE.score(reference, prediction)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def evaluate(encoder, scorer, loader, records):
    encoder.eval()
    scorer.eval()

    all_scores = []
    record_index = 0

    with torch.no_grad():
        for batch in loader:
            logits = get_logits(encoder, scorer, batch)

            for i in range(logits.size(0)):
                record = records[record_index]
                record_index += 1

                sentences = record["sentences"]
                labels = record["labels"]
                valid_count = min(len(sentences), len(labels))

                if valid_count == 0:
                    continue

                count = min(SUMMARY_SENTENCES, valid_count)

                selected = torch.topk(
                    logits[i, :valid_count],
                    k=count,
                ).indices.tolist()

                # Return selected sentences in article order.
                selected.sort()

                prediction = " ".join(
                    sentences[j] for j in selected
                )

                reference = " ".join(
                    sentences[j]
                    for j, label in enumerate(labels[:valid_count])
                    if int(label) == 1
                )

                if not reference.strip():
                    continue

                all_scores.append(
                    calculate_rouge(prediction, reference)
                )

    if not all_scores:
        raise RuntimeError("No valid examples were evaluated.")

    return {
        metric: float(np.mean([
            row[metric] for row in all_scores
        ]))
        for metric in ("rouge1", "rouge2", "rougeL")
    }


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one(model_type, seed, vocab):
    # Reset seed before creating either model so the comparison
    # uses the same random seed for both variants.
    set_seed(seed)

    train_dataset = SummarizationDataset(
        TRAIN_PATH,
        vocab,
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
    )

    val_dataset = SummarizationDataset(
        VAL_PATH,
        vocab,
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
    )

    # Separate, explicitly seeded DataLoader shuffle generator.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Ensure the generator is actually used for training shuffle.
    train_loader.generator = generator

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    encoder, scorer = make_model(model_type, len(vocab))

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_rouge = -1.0
    best_metrics = None
    best_epoch = 0

    checkpoint_path = (
        CHECKPOINT_DIR / f"{model_type}_seed_{seed}_best.pt"
    )

    print(f"\n{'=' * 65}")
    print(f"Model: {model_type} | Seed: {seed}")
    print(f"{'=' * 65}")

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        scorer.train()

        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            logits = get_logits(encoder, scorer, batch)

            labels = batch["labels"].to(DEVICE)
            mask = batch["sentence_mask"].to(DEVICE).float()

            per_item_loss = criterion(logits, labels)
            loss = (
                (per_item_loss * mask).sum()
                / mask.sum().clamp(min=1.0)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(scorer.parameters()),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / max(batches, 1)

        val_metrics = evaluate(
            encoder,
            scorer,
            val_loader,
            read_records(VAL_PATH),
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"R1={val_metrics['rouge1']:.4f} | "
            f"R2={val_metrics['rouge2']:.4f} | "
            f"RL={val_metrics['rougeL']:.4f}"
        )

        if val_metrics["rougeL"] > best_val_rouge:
            best_val_rouge = val_metrics["rougeL"]
            best_metrics = val_metrics
            best_epoch = epoch

            torch.save({
                "seed": seed,
                "model_type": model_type,
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "scorer_state_dict": scorer.state_dict(),
                "validation_metrics": val_metrics,
            }, checkpoint_path)

    print(
        f"Best epoch: {best_epoch} | "
        f"Best validation ROUGE-L: {best_val_rouge:.4f}"
    )

    return {
        "model": model_type,
        "seed": seed,
        "best_epoch": best_epoch,
        **best_metrics,
        "checkpoint": str(checkpoint_path),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    print("=" * 70)
    print("PB-ANN POSITION FEATURE MULTI-SEED EXPERIMENT")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print("Held-out test set: NOT USED")

    for path in (TRAIN_PATH, VAL_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    vocab = Vocabulary.load(VOCAB_PATH)
    results = []

    for seed in SEEDS:
        for model_type in ("no_position", "position"):
            result = train_one(model_type, seed, vocab)
            results.append(result)

            raw_path = OUTPUT_DIR / "multiseed_raw.json"
            with raw_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)

    summary = {}

    for model_type in ("no_position", "position"):
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

    summary["experiment"] = {
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "test_set_used": False,
    }

    summary_path = OUTPUT_DIR / "multiseed_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 70)
    print("MULTI-SEED VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} {'Metric':<10} {'Mean':>10} {'Std':>10}")

    for model_type in ("no_position", "position"):
        for metric in ("rouge1", "rouge2", "rougeL"):
            values = summary[model_type][metric]

            print(
                f"{model_type:<18} "
                f"{metric:<10} "
                f"{values['mean']:>10.4f} "
                f"{values['std']:>10.4f}"
            )

    print(f"\nSaved summary: {summary_path}")
    print("Held-out test set was not used.")


if __name__ == "__main__":
    main()
