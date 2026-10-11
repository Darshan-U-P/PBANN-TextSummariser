
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Project paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    SummarizationDataset,
    Vocabulary,
    create_dataloader,
)
from src.text_encoder import TextEncoder
from src.pbann import PBANNTextScorer


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "validation.jsonl"
VOCAB_PATH = MODEL_DIR / "vocab.json"

# Preserve the original checkpoint.
CHECKPOINT_PATH = MODEL_DIR / "pbann_weighted.pt"

SEED = 42

BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

MAX_SENTENCES = 60
MAX_TOKENS = 64

EMBEDDING_DIM = 256
NUM_NEURONS = 512
RECURRENT_STEPS = 6
SPARSITY = 0.8

MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0

# Used for diagnostic metrics, not for the training loss.
DECISION_THRESHOLD = 0.5

# Prevent an excessively large positive-class weight.
MAX_POS_WEIGHT = 15.0


# ============================================================
# Reproducibility and device
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(logits, labels, mask):
    """Calculate metrics over valid sentence positions."""
    predictions = (
        (torch.sigmoid(logits) >= DECISION_THRESHOLD) & mask
    )
    targets = (labels >= 0.5) & mask

    true_positive = (predictions & targets).sum().item()
    false_positive = (
        predictions & ~targets & mask
    ).sum().item()
    false_negative = (
        ~predictions & targets & mask
    ).sum().item()

    precision = true_positive / max(
        true_positive + false_positive, 1
    )
    recall = true_positive / max(
        true_positive + false_negative, 1
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    valid_count = mask.sum().item()
    predicted_positive_rate = (
        predictions.sum().item() / max(valid_count, 1)
    )
    actual_positive_rate = (
        targets.sum().item() / max(valid_count, 1)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": predicted_positive_rate,
        "actual_positive_rate": actual_positive_rate,
    }


# ============================================================
# Training and validation
# ============================================================

def run_epoch(
    encoder,
    scorer,
    loader,
    optimizer,
    criterion,
    device,
    training,
):
    encoder.train(training)
    scorer.train(training)

    total_loss = 0.0
    total_valid_sentences = 0

    all_predictions = []
    all_targets = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for batch_index, batch in enumerate(loader, start=1):
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            sentence_embeddings = encoder(
                token_ids,
                token_mask,
            )

            logits = scorer(
                sentence_embeddings,
                sentence_mask,
            )

            # Only real sentence positions contribute to loss.
            valid_logits = logits[sentence_mask]
            valid_labels = labels[sentence_mask]

            loss = criterion(valid_logits, valid_labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_index}"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters())
                    + list(scorer.parameters()),
                    MAX_GRAD_NORM,
                )

                optimizer.step()

            valid_count = valid_labels.numel()
            total_loss += loss.item() * valid_count
            total_valid_sentences += valid_count

            with torch.no_grad():
                predictions = (
                    torch.sigmoid(valid_logits)
                    >= DECISION_THRESHOLD
                )

                all_predictions.append(
                    predictions.detach().cpu()
                )
                all_targets.append(
                    (valid_labels >= 0.5).detach().cpu()
                )

            if training and batch_index % 100 == 0:
                print(
                    f"  Batch {batch_index}/{len(loader)}"
                    f" | Loss: {loss.item():.4f}"
                )

    if total_valid_sentences == 0:
        raise ValueError("No valid sentence positions found.")

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    true_positive = (predictions & targets).sum().item()
    false_positive = (predictions & ~targets).sum().item()
    false_negative = (~predictions & targets).sum().item()

    precision = true_positive / max(
        true_positive + false_positive, 1
    )
    recall = true_positive / max(
        true_positive + false_negative, 1
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    predicted_positive_rate = (
        predictions.float().mean().item()
    )
    actual_positive_rate = targets.float().mean().item()

    return {
        "loss": total_loss / total_valid_sentences,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": predicted_positive_rate,
        "actual_positive_rate": actual_positive_rate,
    }


def print_metrics(name, metrics):
    print(f"\n{name}:")
    print(f"  Loss:                   {metrics['loss']:.4f}")
    print(f"  Precision:              {metrics['precision']:.4f}")
    print(f"  Recall:                 {metrics['recall']:.4f}")
    print(f"  F1:                     {metrics['f1']:.4f}")
    print(
        "  Predicted positive rate:"
        f" {metrics['predicted_positive_rate']:.4f}"
    )
    print(
        "  Actual positive rate:   "
        f"{metrics['actual_positive_rate']:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = get_device()

    print("=" * 65)
    print("PB-ANN Text Summariser — Weighted Training")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    for path in (TRAIN_PATH, VAL_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Vocabulary and datasets
    # --------------------------------------------------------

    vocab = Vocabulary.load(VOCAB_PATH)
    print("Vocabulary size:", len(vocab))

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

    train_loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------------------------
    # Calculate class imbalance from actual training records
    # --------------------------------------------------------

    positive_count = sum(
        sum(example["labels"])
        for example in train_dataset.examples
    )

    total_count = sum(
        len(example["labels"])
        for example in train_dataset.examples
    )

    negative_count = total_count - positive_count

    if positive_count == 0:
        raise ValueError("Training dataset has no positive labels.")

    pos_weight_value = min(
        negative_count / positive_count,
        MAX_POS_WEIGHT,
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print("\nClass balance:")
    print(f"  Positive labels: {positive_count:,}")
    print(f"  Negative labels: {negative_count:,}")
    print(f"  Positive-class weight: {pos_weight_value:.2f}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    encoder = TextEncoder(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    scorer = PBANNTextScorer(
        embedding_dim=EMBEDDING_DIM,
        num_neurons=NUM_NEURONS,
        steps=RECURRENT_STEPS,
        sparsity=SPARSITY,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in list(encoder.parameters())
        + list(scorer.parameters())
        if parameter.requires_grad
    )

    print("\nTrainable parameters:", f"{trainable_parameters:,}")

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(scorer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_metrics = run_epoch(
            encoder,
            scorer,
            train_loader,
            optimizer,
            criterion,
            device,
            training=True,
        )

        val_metrics = run_epoch(
            encoder,
            scorer,
            val_loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

        print_metrics("Training", train_metrics)
        print_metrics("Validation", val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "vocab_size": len(vocab),
                    "embedding_dim": EMBEDDING_DIM,
                    "num_neurons": NUM_NEURONS,
                    "recurrent_steps": RECURRENT_STEPS,
                    "sparsity": SPARSITY,
                    "max_sentences": MAX_SENTENCES,
                    "max_tokens": MAX_TOKENS,
                    "decision_threshold": DECISION_THRESHOLD,
                    "positive_class_weight": pos_weight_value,
                    "encoder_state_dict": encoder.state_dict(),
                    "scorer_state_dict": scorer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_val_loss,
                },
                CHECKPOINT_PATH,
            )

            print("\nSaved best checkpoint:", CHECKPOINT_PATH)

    print("\nTraining run complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
