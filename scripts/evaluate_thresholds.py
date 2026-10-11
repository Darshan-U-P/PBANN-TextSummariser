import sys
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
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

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

CHECKPOINT_PATH = MODEL_DIR / "pbann_weighted.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"

BATCH_SIZE = 2
MAX_SENTENCES = 60
MAX_TOKENS = 64

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU unavailable; using CPU.")

    return device


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(path, device):
    """
    Load a checkpoint created by this project.
    """

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    required_keys = [
        "encoder_state_dict",
        "scorer_state_dict",
    ]

    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(
                f"Checkpoint is missing '{key}'. "
                f"Check the checkpoint file: {path}"
            )

    return checkpoint


# ============================================================
# Collect validation predictions
# ============================================================

@torch.inference_mode()
def collect_validation_predictions(
    encoder,
    scorer,
    dataloader,
    device,
):
    """
    Run inference on the validation set.

    Padded sentences are excluded from the collected
    probabilities and labels.
    """

    encoder.eval()
    scorer.eval()

    all_probabilities = []
    all_labels = []

    total_batches = len(dataloader)

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        token_ids = batch["token_ids"].to(device)
        token_mask = batch["token_mask"].to(device)
        sentence_mask = batch["sentence_mask"].to(device)
        labels = batch["labels"].to(device)

        # Encode sentence tokens.
        sentence_embeddings = encoder(
            token_ids,
            token_mask,
        )

        # Calculate sentence importance logits.
        logits = scorer(
            sentence_embeddings,
            sentence_mask,
        )

        # Convert logits to probabilities.
        probabilities = torch.sigmoid(logits)

        # Remove padded sentence positions.
        valid_probabilities = probabilities[sentence_mask]
        valid_labels = labels[sentence_mask]

        all_probabilities.append(
            valid_probabilities.detach().cpu()
        )

        all_labels.append(
            valid_labels.detach().cpu()
        )

        if (
            batch_index % 50 == 0
            or batch_index == total_batches
        ):
            print(
                f"\rEvaluated batches: "
                f"{batch_index}/{total_batches}",
                end="",
                flush=True,
            )

    print()

    if not all_labels:
        raise RuntimeError(
            "No validation predictions were collected."
        )

    probabilities = torch.cat(
        all_probabilities
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy().astype(int)

    return probabilities, labels


# ============================================================
# Calculate threshold metrics
# ============================================================

def evaluate_threshold(
    probabilities,
    labels,
    threshold,
):
    """
    Evaluate sentence-level binary classification
    at a specified probability threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_rate": float(predictions.mean()),
        "actual_rate": float(labels.mean()),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 68)
    print("PB-ANN Text Summariser — Threshold Evaluation")
    print("=" * 68)

    # --------------------------------------------------------
    # 1. Check required files
    # --------------------------------------------------------

    required_files = [
        CHECKPOINT_PATH,
        VOCAB_PATH,
        VALIDATION_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # 2. Select device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # 3. Load trained checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    print(f"Checkpoint: {CHECKPOINT_PATH.name}")

    # --------------------------------------------------------
    # 4. Load vocabulary correctly
    # --------------------------------------------------------

    # IMPORTANT:
    # Vocabulary.load() is a class method.
    # It returns the populated Vocabulary instance.
    vocab = Vocabulary.load(VOCAB_PATH)

    vocab_size = len(vocab)

    print(f"Loaded vocabulary size: {vocab_size}")

    expected_vocab_size = (
        checkpoint["encoder_state_dict"]["embedding.weight"]
        .shape[0]
    )

    # Check vocabulary size against the trained embedding.
    if vocab_size != expected_vocab_size:
        raise ValueError(
            "Vocabulary/checkpoint mismatch.\n"
            f"Vocabulary entries: {vocab_size}\n"
            f"Checkpoint embedding rows: {expected_vocab_size}\n"
            "Use the vocabulary saved alongside the trained model."
        )

    # --------------------------------------------------------
    # 5. Load validation dataset
    # --------------------------------------------------------

    validation_dataset = SummarizationDataset(
        jsonl_path=VALIDATION_PATH,
        vocab=vocab,
        max_sentences=MAX_SENTENCES,
        max_tokens=MAX_TOKENS,
    )

    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Validation examples: "
        f"{len(validation_dataset)}"
    )
    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    # --------------------------------------------------------
    # 6. Read model configuration from checkpoint
    # --------------------------------------------------------

    embedding_dim = checkpoint.get(
        "embedding_dim",
        256,
    )

    num_neurons = checkpoint.get(
        "num_neurons",
        512,
    )

    recurrent_steps = checkpoint.get(
        "recurrent_steps",
        6,
    )

    sparsity = checkpoint.get(
        "sparsity",
        0.8,
    )

    # --------------------------------------------------------
    # 7. Initialize models
    # --------------------------------------------------------

    encoder = TextEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    ).to(device)

    scorer = PBANNTextScorer(
        embedding_dim=embedding_dim,
        num_neurons=num_neurons,
        steps=recurrent_steps,
        sparsity=sparsity,
    ).to(device)

    # --------------------------------------------------------
    # 8. Restore trained parameters
    # --------------------------------------------------------

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    scorer.load_state_dict(
        checkpoint["scorer_state_dict"]
    )

    print("Encoder weights loaded successfully.")
    print("PB-ANN scorer weights loaded successfully.")

    # --------------------------------------------------------
    # 9. Run validation inference once
    # --------------------------------------------------------

    print("\nCollecting validation predictions...")

    probabilities, labels = (
        collect_validation_predictions(
            encoder=encoder,
            scorer=scorer,
            dataloader=validation_loader,
            device=device,
        )
    )

    print(
        f"\nValidation sentences: {len(labels):,}"
    )

    print(
        f"Positive labels: {int(labels.sum()):,}"
    )

    print(
        f"Actual positive rate: "
        f"{labels.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 10. Evaluate all thresholds
    # --------------------------------------------------------

    results = []

    for threshold in THRESHOLDS:
        result = evaluate_threshold(
            probabilities=probabilities,
            labels=labels,
            threshold=threshold,
        )

        results.append(result)

    print("\n" + "=" * 88)
    print("THRESHOLD COMPARISON")
    print("=" * 88)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'Predicted %':>13} "
        f"{'Actual %':>11}"
    )

    print("-" * 88)

    for result in results:
        print(
            f"{result['threshold']:>10.2f} "
            f"{result['precision']:>11.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} "
            f"{result['predicted_rate'] * 100:>12.2f}% "
            f"{result['actual_rate'] * 100:>10.2f}%"
        )

    # --------------------------------------------------------
    # 11. Find best threshold by validation F1
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda result: result["f1"],
    )

    print("\n" + "=" * 68)
    print("BEST THRESHOLD BY VALIDATION F1")
    print("=" * 68)

    print(
        f"Threshold:        "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Precision:        "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall:           "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1 score:         "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"Predicted positive rate: "
        f"{best_result['predicted_rate'] * 100:.2f}%"
    )

    print("\nEvaluation complete.")

    print(
        "This threshold maximizes sentence-level F1 "
        "among the thresholds tested."
    )

    print(
        "It may not maximize ROUGE or summary quality. "
        "Generated summaries must be evaluated separately."
    )


if __name__ == "__main__":
    main()
