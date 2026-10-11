
import json
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SummarizationDataset, Vocabulary, create_dataloader
from src.text_encoder import TextEncoder
from src.pbann import PBANNTextScorer


# ============================================================
# CONFIGURATION — separate experiment, no existing checkpoints overwritten
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "evaluation_results"

TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "validation.jsonl"
VOCAB_PATH = MODEL_DIR / "vocab.json"

EXPERIMENT_DIR = MODEL_DIR / "ranking_loss_experiment"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
BEST_CHECKPOINT = EXPERIMENT_DIR / "pbann_best_ranking.pt"
HISTORY_PATH = OUTPUT_DIR / "ranking_loss_training.json"

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
MAX_POS_WEIGHT = 15.0
MAX_SUMMARY_SENTENCES = 3

# Combined objective:
# total_loss = BCE_loss + RANKING_WEIGHT * pairwise_ranking_loss
RANKING_WEIGHT = 0.2
RANKING_MARGIN = 0.5

EARLY_STOPPING_PATIENCE = 2
MIN_ROUGE_IMPROVEMENT = 1e-4
PROGRESS_INTERVAL = 100

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# VALIDATION RECORDS AND ROUGE
# ============================================================

def load_rouge_records(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            sentences = item.get("sentences", [])
            reference = item.get("summary", "")

            if not isinstance(sentences, list):
                raise ValueError(
                    f"Invalid sentences at {path}, line {line_number}"
                )

            if not isinstance(reference, str) or not reference.strip():
                continue

            sentences = [
                " ".join(str(sentence).split())
                for sentence in sentences[:MAX_SENTENCES]
                if str(sentence).strip()
            ]

            if sentences:
                records.append({
                    "sentences": sentences,
                    "reference": " ".join(reference.split()),
                })

    if not records:
        raise ValueError(f"No valid records found: {path}")

    return records


def calculate_rouge(prediction, reference):
    scores = ROUGE.score(reference, prediction)
    return {
        key: scores[key].fmeasure
        for key in ("rouge1", "rouge2", "rougeL")
    }


def mean_rouge(rows):
    return {
        key: mean(row[key] for row in rows)
        for key in ("rouge1", "rouge2", "rougeL")
    }


def evaluate_top3(encoder, scorer, records, vocab, device):
    encoder.eval()
    scorer.eval()
    all_scores = []

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            sentences = record["sentences"]
            encoded = [
                vocab.encode(sentence, max_tokens=MAX_TOKENS)
                for sentence in sentences
            ]

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
                (1, sentence_count),
                dtype=torch.bool,
                device=device,
            )

            embeddings = encoder(token_ids, token_mask)
            logits = scorer(embeddings, sentence_mask)
            probabilities = torch.sigmoid(logits[0]).cpu().tolist()

            ranked = sorted(
                range(sentence_count),
                key=lambda i: probabilities[i],
                reverse=True,
            )[:MAX_SUMMARY_SENTENCES]

            # Keep selected sentences in their original article order.
            selected = sorted(ranked)
            prediction = " ".join(sentences[i] for i in selected)

            all_scores.append(
                calculate_rouge(prediction, record["reference"])
            )

            if index % PROGRESS_INTERVAL == 0 or index == len(records):
                print(f"  Validation ROUGE: {index}/{len(records)}")

    return mean_rouge(all_scores)


# ============================================================
# PAIRWISE RANKING LOSS
# ============================================================

def pairwise_ranking_loss(logits, labels, sentence_mask):
    """
    For each article, encourage every positive-label sentence to score
    above every negative-label sentence by RANKING_MARGIN.

    Padding is excluded using sentence_mask.
    Articles without both positive and negative labels are skipped.
    """

    article_losses = []

    for batch_index in range(logits.size(0)):
        valid = sentence_mask[batch_index].bool()
        article_logits = logits[batch_index][valid]
        article_labels = labels[batch_index][valid]

        positive_scores = article_logits[article_labels > 0.5]
        negative_scores = article_logits[article_labels <= 0.5]

        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            continue

        # Each matrix entry compares one positive with one negative.
        differences = (
            positive_scores.unsqueeze(1)
            - negative_scores.unsqueeze(0)
        )

        # Soft margin-ranking objective.
        pair_losses = F.softplus(RANKING_MARGIN - differences)
        article_losses.append(pair_losses.mean())

    if not article_losses:
        # Differentiable zero connected to the logits.
        return logits.sum() * 0.0

    return torch.stack(article_losses).mean()


# ============================================================
# TRAINING AND VALIDATION LOSS
# ============================================================

def run_loss_epoch(
    encoder,
    scorer,
    loader,
    optimizer,
    bce_criterion,
    device,
    training,
):
    encoder.train(training)
    scorer.train(training)

    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    batch_count = 0

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch_index, batch in enumerate(loader, start=1):
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            embeddings = encoder(token_ids, token_mask)
            logits = scorer(embeddings, sentence_mask)

            # BCE only over real sentences, never padded positions.
            valid_logits = logits[sentence_mask]
            valid_labels = labels[sentence_mask]
            bce_loss = bce_criterion(valid_logits, valid_labels)

            rank_loss = pairwise_ranking_loss(
                logits, labels, sentence_mask
            )

            loss = bce_loss + RANKING_WEIGHT * rank_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_index}"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(scorer.parameters()),
                    MAX_GRAD_NORM,
                )
                optimizer.step()

            total_loss += loss.item()
            total_bce += bce_loss.item()
            total_rank += rank_loss.item()
            batch_count += 1

            if training and batch_index % PROGRESS_INTERVAL == 0:
                print(
                    f"  Batch {batch_index}/{len(loader)}"
                    f" | total={loss.item():.4f}"
                    f" | BCE={bce_loss.item():.4f}"
                    f" | rank={rank_loss.item():.4f}"
                )

    denominator = max(batch_count, 1)
    return {
        "total_loss": total_loss / denominator,
        "bce_loss": total_bce / denominator,
        "ranking_loss": total_rank / denominator,
    }


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    path,
    epoch,
    encoder,
    scorer,
    optimizer,
    vocab_size,
    validation_loss,
    rouge_metrics,
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
        "decision_threshold": None,
        "selection_strategy": "top_k",
        "selection_k": MAX_SUMMARY_SENTENCES,
        "validation_loss": validation_loss,
        "validation_rouge1_f1": rouge_metrics["rouge1"],
        "validation_rouge2_f1": rouge_metrics["rouge2"],
        "validation_rougeL_f1": rouge_metrics["rougeL"],
        "training_objective": "bce_plus_pairwise_ranking",
        "ranking_weight": RANKING_WEIGHT,
        "ranking_margin": RANKING_MARGIN,
        "encoder_state_dict": encoder.state_dict(),
        "scorer_state_dict": scorer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)
    device = get_device()

    for path in (TRAIN_PATH, VAL_PATH, VOCAB_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Refuse to overwrite any previous experiment.
    if EXPERIMENT_DIR.exists():
        raise FileExistsError(
            f"Experiment directory already exists:\n{EXPERIMENT_DIR}\n"
            "Rename or back it up before starting another run."
        )

    CHECKPOINT_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PB-ANN — PAIRWISE RANKING EXPERIMENT")
    print("=" * 68)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    vocab = Vocabulary.load(VOCAB_PATH)
    vocab_size = len(vocab)

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

    val_records = load_rouge_records(VAL_PATH)

    positive_count = 0
    total_count = 0

    for example in train_dataset.examples:
        labels = example["labels"]
        positive_count += sum(labels)
        total_count += len(labels)

    negative_count = total_count - positive_count

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training requires positive and negative labels.")

    pos_weight_value = min(
        negative_count / positive_count,
        MAX_POS_WEIGHT,
    )
    pos_weight = torch.tensor(
        [pos_weight_value], dtype=torch.float32, device=device
    )

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

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(scorer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_rouge = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_records):,}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Positive-class weight: {pos_weight_value:.3f}")
    print(f"Ranking loss weight: {RANKING_WEIGHT}")
    print(f"Ranking margin: {RANKING_MARGIN}")

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "=" * 68)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 68)

        train_metrics = run_loss_epoch(
            encoder,
            scorer,
            train_loader,
            optimizer,
            bce_criterion,
            device,
            training=True,
        )

        val_metrics = run_loss_epoch(
            encoder,
            scorer,
            val_loader,
            optimizer,
            bce_criterion,
            device,
            training=False,
        )

        print(
            f"Train total={train_metrics['total_loss']:.4f} "
            f"BCE={train_metrics['bce_loss']:.4f} "
            f"rank={train_metrics['ranking_loss']:.4f}"
        )
        print(
            f"Val total={val_metrics['total_loss']:.4f} "
            f"BCE={val_metrics['bce_loss']:.4f} "
            f"rank={val_metrics['ranking_loss']:.4f}"
        )

        print("\nEvaluating validation ROUGE with Top-3...")
        rouge_metrics = evaluate_top3(
            encoder, scorer, val_records, vocab, device
        )

        print(
            f"ROUGE-1={rouge_metrics['rouge1']:.4f} "
            f"ROUGE-2={rouge_metrics['rouge2']:.4f} "
            f"ROUGE-L={rouge_metrics['rougeL']:.4f}"
        )

        epoch_checkpoint = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"
        save_checkpoint(
            epoch_checkpoint,
            epoch,
            encoder,
            scorer,
            optimizer,
            vocab_size,
            val_metrics["total_loss"],
            rouge_metrics,
        )
        print("Saved:", epoch_checkpoint)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "validation_rouge": rouge_metrics,
            "checkpoint": str(epoch_checkpoint),
        })

        current_score = rouge_metrics["rougeL"]

        if current_score > best_rouge + MIN_ROUGE_IMPROVEMENT:
            best_rouge = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                epoch,
                encoder,
                scorer,
                optimizer,
                vocab_size,
                val_metrics["total_loss"],
                rouge_metrics,
            )
            print("New best checkpoint:", BEST_CHECKPOINT)
        else:
            epochs_without_improvement += 1
            print(
                f"No sufficient ROUGE-L improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        HISTORY_PATH.write_text(
            json.dumps({
                "selection_metric": "validation_rougeL_f1",
                "objective": "bce_plus_pairwise_ranking",
                "ranking_weight": RANKING_WEIGHT,
                "ranking_margin": RANKING_MARGIN,
                "best_epoch": best_epoch,
                "best_validation_rougeL_f1": best_rouge,
                "epochs": history,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping based on validation ROUGE-L.")
            break

    print("\n" + "=" * 68)
    print("RANKING EXPERIMENT COMPLETE")
    print("=" * 68)
    print("Epochs completed:", len(history))
    print("Best epoch:", best_epoch)
    print(f"Best validation ROUGE-L F1: {best_rouge:.4f}")
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("History:", HISTORY_PATH)
    print("\nExisting checkpoints were not modified.")


if __name__ == "__main__":
    main()
