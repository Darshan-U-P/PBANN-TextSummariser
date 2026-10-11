
import sys
from pathlib import Path

import torch

# ============================================================
# Project paths and imports
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    SummarizationDataset,
    build_vocabulary,
    create_dataloader,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "validation.jsonl"

BATCH_SIZE = 2
MAX_SENTENCES = 60
MAX_TOKENS = 64
MIN_FREQUENCY = 2
MAX_VOCAB_SIZE = 50_000

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Validate input files
# ============================================================

def main():
    print("=" * 60)
    print("PB-ANN Dataset Pipeline Test")
    print("=" * 60)

    for path in (TRAIN_PATH, VAL_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {path}"
            )

    # --------------------------------------------------------
    # Build vocabulary from training data only
    # --------------------------------------------------------

    print("\nBuilding vocabulary from training data...")

    vocab = build_vocabulary(
        TRAIN_PATH,
        min_frequency=MIN_FREQUENCY,
        max_size=MAX_VOCAB_SIZE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vocab.save(MODEL_DIR / "vocab.json")

    print("Vocabulary size:", len(vocab))
    print("Vocabulary saved:", MODEL_DIR / "vocab.json")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

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

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty.")

    # --------------------------------------------------------
    # Load one batch
    # --------------------------------------------------------

    loader = create_dataloader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    token_ids = batch["token_ids"].to(device)
    token_mask = batch["token_mask"].to(device)
    sentence_mask = batch["sentence_mask"].to(device)
    labels = batch["labels"].to(device)

    # --------------------------------------------------------
    # Display and validate tensor shapes
    # --------------------------------------------------------

    print("\nDataset information")
    print("-" * 40)
    print("Device:", device)
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    print("\nBatch tensor shapes")
    print("-" * 40)
    print("Token IDs:", tuple(token_ids.shape))
    print("Token mask:", tuple(token_mask.shape))
    print("Sentence mask:", tuple(sentence_mask.shape))
    print("Labels:", tuple(labels.shape))

    print("\nBatch statistics")
    print("-" * 40)
    print("Valid sentences:", int(sentence_mask.sum().item()))
    print("Positive labels:", int(labels.sum().item()))
    print("Padded sentences:", int((~sentence_mask).sum().item()))

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert token_ids.ndim == 3, (
        "Token IDs must have shape [batch, sentences, tokens]."
    )

    assert token_mask.shape == token_ids.shape, (
        "Token mask must match token IDs."
    )

    assert sentence_mask.shape == labels.shape, (
        "Sentence mask must match labels."
    )

    assert token_ids.dtype == torch.long
    assert token_mask.dtype == torch.bool
    assert sentence_mask.dtype == torch.bool
    assert labels.dtype == torch.float32

    assert torch.all(
        labels[~sentence_mask] == 0
    ).item(), "Padded sentences must have zero labels."

    assert torch.all(
        token_mask <= sentence_mask.unsqueeze(-1)
    ).item(), "Padded sentences must not contain valid tokens."

    assert torch.all(
        (labels == 0) | (labels == 1)
    ).item(), "Labels must be binary."

    print("\nDataset test passed.")


if __name__ == "__main__":
    main()
