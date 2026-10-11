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
if str(ROOT) not in sys.path:
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

# Keep the original baseline checkpoint untouched.
BASELINE_CHECKPOINT = MODEL_DIR / "pbann_weighted.pt"

# Phase 2 produces a separate checkpoint.
CHECKPOINT_PATH = MODEL_DIR / "pbann_phase2_best.pt"

SEED = 42

BATCH_SIZE = 2
EPOCHS = 5

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

# Diagnostic metrics use this threshold.
# Threshold tuning remains a separate validation step.
DECISION_THRESHOLD = 0.5

MAX_POS_WEIGHT = 15.0

# Learning-rate scheduling.
LR_FACTOR = 0.5
LR_PATIENCE = 1
MIN_LEARNING_RATE = 1e-6

# Stop after this many consecutive epochs without
# a new best validation loss.
EARLY_STOPPING_PATIENCE = 2

# Require at least this much validation-loss improvement.
MIN_DELTA = 1e-4

PROGRESS_INTERVAL = 100


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

def calculate_binary_metrics(predictions, targets):
    true_positive = (
        predictions & targets
    ).sum().item()

    false_positive = (
        predictions & ~targets
    ).sum().item()

    false_negative = (
        ~predictions & targets
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

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": (
            predictions.float().mean().item()
        ),
        "actual_positive_rate": (
            targets.float().mean().item()
        ),
    }


def print_metrics(name, metrics):
    print(f"\n{name}:")
    print(f"  Loss:                    {metrics['loss']:.4f}")
    print(f"  Precision:               {metrics['precision']:.4f}")
    print(f"  Recall:                  {metrics['recall']:.4f}")
    print(f"  F1:                      {metrics['f1']:.4f}")
    print(
        "  Predicted positive rate: "
        f"{metrics['predicted_positive_rate']:.4f}"
    )
    print(
        "  Actual positive rate:    "
        f"{metrics['actual_positive_rate']:.4f}"
    )


# ============================================================
# One training or validation epoch
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
        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            # Encode the article sentences.
            sentence_embeddings = encoder(
                token_ids,
                token_mask,
            )

            # Predict sentence-importance logits.
            logits = scorer(
                sentence_embeddings,
                sentence_mask,
            )

            # Exclude padded sentence positions.
            valid_logits = logits[sentence_mask]
            valid_labels = labels[sentence_mask]

            loss = criterion(
                valid_logits,
                valid_labels,
            )

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

                targets = valid_labels >= 0.5

                all_predictions.append(
                    predictions.detach().cpu()
                )

                all_targets.append(
                    targets.detach().cpu()
                )

            if (
                training
                and batch_index % PROGRESS_INTERVAL == 0
            ):
                print(
                    f"  Batch {batch_index}/{len(loader)}"
                    f" | Loss: {loss.item():.4f}"
                )

    if total_valid_sentences == 0:
        raise ValueError(
            "No valid sentence positions were found."
        )

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    metrics = calculate_binary_metrics(
        predictions,
        targets,
    )

    metrics["loss"] = (
        total_loss / total_valid_sentences
    )

    return metrics


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    path,
    epoch,
    encoder,
    scorer,
    optimizer,
    scheduler,
    vocab_size,
    pos_weight_value,
    validation_metrics,
    best_val_loss,
):
    checkpoint = {
        "epoch": epoch,
        "vocab_size": vocab_size,
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
        "scheduler_state_dict": scheduler.state_dict(),

        "validation_loss": validation_metrics["loss"],
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "best_validation_loss": best_val_loss,
    }

    torch.save(checkpoint, path)


# ============================================================
# Main training procedure
# ============================================================

def main():
    set_seed(SEED)
    device = get_device()

    print("=" * 68)
    print("PB-ANN Text Summariser — Phase 2 Training")
    print("=" * 68)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Maximum epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Early-stopping patience:", EARLY_STOPPING_PATIENCE)

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    for path in (
        TRAIN_PATH,
        VAL_PATH,
        VOCAB_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"Phase 2 checkpoint already exists:\n"
            f"{CHECKPOINT_PATH}\n"
            "Rename or back up that file before starting "
            "a new experiment. The existing checkpoint "
            "will not be overwritten automatically."
        )

    # --------------------------------------------------------
    # Load vocabulary and datasets
    # --------------------------------------------------------

    vocab = Vocabulary.load(VOCAB_PATH)
    vocab_size = len(vocab)

    print("\nVocabulary size:", vocab_size)

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
    # Calculate class imbalance
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

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "Training labels must contain both positive "
            "and negative examples."
        )

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
    # Initialize model from scratch
    # --------------------------------------------------------

    encoder = TextEncoder(
        vocab_size=vocab_size,
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
        for parameter in (
            list(encoder.parameters())
            + list(scorer.parameters())
        )
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        f"{trainable_parameters:,}",
    )

    # --------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(scorer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    # Reduce the learning rate when validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        threshold=MIN_DELTA,
        min_lr=MIN_LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Multi-epoch training
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 68)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("=" * 68)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.7g}")

        train_metrics = run_epoch(
            encoder=encoder,
            scorer=scorer,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            training=True,
        )

        val_metrics = run_epoch(
            encoder=encoder,
            scorer=scorer,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            training=False,
        )

        print_metrics(
            "Training",
            train_metrics,
        )

        print_metrics(
            "Validation",
            val_metrics,
        )

        # Step scheduler using validation loss.
        scheduler.step(val_metrics["loss"])

        next_lr = optimizer.param_groups[0]["lr"]

        if next_lr < current_lr:
            print(
                f"\nLearning rate reduced: "
                f"{current_lr:.7g} -> {next_lr:.7g}"
            )

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "validation_loss": val_metrics["loss"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
        })

        # Save a checkpoint only when validation loss
        # improves by at least MIN_DELTA.
        improved = (
            val_metrics["loss"] < best_val_loss - MIN_DELTA
        )

        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=CHECKPOINT_PATH,
                epoch=epoch,
                encoder=encoder,
                scorer=scorer,
                optimizer=optimizer,
                scheduler=scheduler,
                vocab_size=vocab_size,
                pos_weight_value=pos_weight_value,
                validation_metrics=val_metrics,
                best_val_loss=best_val_loss,
            )

            print(
                "\nSaved new best checkpoint:",
                CHECKPOINT_PATH,
            )

        else:
            epochs_without_improvement += 1

            print(
                "\nValidation loss did not improve."
                f" Consecutive epochs without improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "\nEarly stopping: validation loss has not "
                "improved for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("PHASE 2 TRAINING COMPLETE")
    print("=" * 68)

    print(f"Epochs completed: {len(history)}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")

    print("\nEpoch history:")
    print(
        f"{'Epoch':>5} "
        f"{'Train loss':>12} "
        f"{'Val loss':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>9} "
        f"{'F1':>9}"
    )

    for item in history:
        print(
            f"{item['epoch']:>5} "
            f"{item['train_loss']:>12.4f} "
            f"{item['validation_loss']:>10.4f} "
            f"{item['validation_precision']:>11.4f} "
            f"{item['validation_recall']:>9.4f} "
            f"{item['validation_f1']:>9.4f}"
        )

    print(
        "\nThe original pbann_weighted.pt checkpoint "
        "was not modified."
    )

    print(
        "Next: evaluate the Phase 2 checkpoint using "
        "threshold tuning and ROUGE on held-out test data."
    )


if __name__ == "__main__":
    main()