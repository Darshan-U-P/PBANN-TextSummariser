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
sys.path.insert(0, str(ROOT))

from src.dataset import SummarizationDataset, Vocabulary, create_dataloader
from src.text_encoder import TextEncoder
from src.pbann_position import PBANNPositionTextScorer


# ------------------------------------------------------------
# Configuration: match the position-aware experiment
# ------------------------------------------------------------

TRAIN_PATH = ROOT / "data/processed/train.jsonl"
VAL_PATH = ROOT / "data/processed/validation.jsonl"
VOCAB_PATH = ROOT / "models/vocab.json"

EXPERIMENT_DIR = ROOT / "models/position_ablation"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
BEST_CHECKPOINT = EXPERIMENT_DIR / "pbann_no_position_best.pt"
HISTORY_PATH = ROOT / "evaluation_results/position_ablation_training.json"
COMPARISON_PATH = ROOT / "evaluation_results/position_ablation_comparison.json"

POSITION_CHECKPOINT = (
    ROOT / "models/position_experiment/pbann_position_best.pt"
)

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
POSITION_DIM = 32

MAX_GRAD_NORM = 1.0
MAX_POS_WEIGHT = 15.0
EARLY_STOPPING_PATIENCE = 2
MIN_ROUGE_IMPROVEMENT = 1e-4
PROGRESS_INTERVAL = 100
TOP_K = 3

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


# ------------------------------------------------------------
# PB-ANN ablation: no positional features
# ------------------------------------------------------------

class PBANNNoPositionTextScorer(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_neurons=512,
        steps=6,
        leak=0.2,
        inhibition_strength=0.1,
        sparsity=0.8,
        dropout=0.05,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_neurons = num_neurons
        self.steps = steps
        self.leak = leak
        self.inhibition_strength = inhibition_strength

        # No position encoder. Text embeddings enter directly.
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_neurons),
        )

        self.thresholds = nn.Parameter(torch.zeros(num_neurons))
        self.recurrent_weight = nn.Parameter(
            torch.empty(num_neurons, num_neurons)
        )
        nn.init.xavier_uniform_(self.recurrent_weight, gain=0.4)

        mask = (
            torch.rand(num_neurons, num_neurons) >= sparsity
        ).float()
        self.register_buffer("recurrent_mask", mask)

        self.rec_norm = nn.LayerNorm(num_neurons)
        self.rec_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(num_neurons)
        self.sentence_head = nn.Linear(num_neurons, 1)

    def forward(self, sentence_embeddings, sentence_mask=None):
        batch, count, dim = sentence_embeddings.shape

        if dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, got {dim}"
            )

        if sentence_mask is None:
            sentence_mask = torch.ones(
                batch, count, dtype=torch.bool,
                device=sentence_embeddings.device,
            )
        else:
            sentence_mask = sentence_mask.bool()

        stimulation = self.encoder(sentence_embeddings)
        potentials = torch.zeros_like(stimulation)
        weights = self.recurrent_weight * self.recurrent_mask

        for _ in range(self.steps):
            recurrent = F.linear(potentials, weights)
            recurrent = self.rec_norm(recurrent)

            potentials = (
                (1.0 - self.leak) * potentials
                + recurrent
                + stimulation
            )

            firing = F.relu(potentials - self.thresholds)
            firing_mean = firing.mean(dim=-1, keepdim=True)

            potentials = (
                firing - self.inhibition_strength * firing_mean
            )
            potentials = self.rec_dropout(potentials)

        logits = self.sentence_head(
            self.out_norm(potentials)
        ).squeeze(-1)

        return logits.masked_fill(~sentence_mask, -1e4)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path):
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            sentences = [
                " ".join(str(s).split())
                for s in record.get("sentences", [])[:MAX_SENTENCES]
                if str(s).strip()
            ]
            reference = record.get("summary", "")

            if sentences and isinstance(reference, str) and reference.strip():
                records.append({
                    "sentences": sentences,
                    "reference": " ".join(reference.split()),
                })

    if not records:
        raise ValueError(f"No usable records in {path}")

    return records


def rouge_scores(prediction, reference):
    result = ROUGE.score(reference, prediction)
    return {key: result[key].fmeasure for key in ("rouge1", "rouge2", "rougeL")}


def mean_scores(items):
    return {
        key: mean(item[key] for item in items)
        for key in ("rouge1", "rouge2", "rougeL")
    }


def encode_sentences(sentences, vocab, device):
    encoded = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    width = max(max((len(ids) for ids in encoded), default=0), 1)
    ids = torch.zeros(
        (1, len(encoded), width), dtype=torch.long, device=device
    )

    for i, sequence in enumerate(encoded):
        if sequence:
            ids[0, i, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )

    token_mask = ids != 0
    sentence_mask = torch.ones(
        (1, len(encoded)), dtype=torch.bool, device=device
    )
    return ids, token_mask, sentence_mask


def evaluate(encoder, scorer, records, vocab, device, with_position):
    encoder.eval()
    scorer.eval()
    results = []

    with torch.inference_mode():
        for number, record in enumerate(records, 1):
            sentences = record["sentences"]
            ids, token_mask, sentence_mask = encode_sentences(
                sentences, vocab, device
            )
            embeddings = encoder(ids, token_mask)

            if with_position:
                n = len(sentences)
                positions = (
                    torch.linspace(
                        0, 1, n, device=device,
                        dtype=embeddings.dtype,
                    ).unsqueeze(0)
                    if n > 1
                    else torch.zeros(
                        (1, n), device=device,
                        dtype=embeddings.dtype,
                    )
                )
                logits = scorer(embeddings, sentence_mask, positions)
            else:
                logits = scorer(embeddings, sentence_mask)

            selected = torch.topk(
                logits[0], k=min(TOP_K, len(sentences))
            ).indices.tolist()
            selected.sort()

            prediction = " ".join(sentences[i] for i in selected)
            results.append(rouge_scores(prediction, record["reference"]))

            if number % 100 == 0 or number == len(records):
                print(f"  Evaluation: {number}/{len(records)}")

    return mean_scores(results)


def evaluate_lead3(records):
    return mean_scores([
        rouge_scores(
            " ".join(record["sentences"][:TOP_K]),
            record["reference"],
        )
        for record in records
    ])


def run_epoch(encoder, scorer, loader, optimizer, criterion, device, training):
    encoder.train(training)
    scorer.train(training)

    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad if training else torch.no_grad

    with context():
        for batch_number, batch in enumerate(loader, 1):
            ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            sentence_mask = batch["sentence_mask"].to(device)
            labels = batch["labels"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            embeddings = encoder(ids, token_mask)
            logits = scorer(embeddings, sentence_mask)
            loss = criterion(logits[sentence_mask], labels[sentence_mask])

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at batch {batch_number}"
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(scorer.parameters()),
                    MAX_GRAD_NORM,
                )
                optimizer.step()

            count = labels[sentence_mask].numel()
            total_loss += loss.item() * count
            total_items += count

            if training and batch_number % PROGRESS_INTERVAL == 0:
                print(f"  Batch {batch_number}/{len(loader)} loss={loss.item():.4f}")

    return total_loss / max(total_items, 1)


def save_checkpoint(path, epoch, encoder, scorer, optimizer, val_loss, metrics):
    torch.save({
        "experiment": "pbann_without_position_features",
        "epoch": epoch,
        "vocab_size": len(Vocabulary.load(VOCAB_PATH)),
        "embedding_dim": EMBEDDING_DIM,
        "num_neurons": NUM_NEURONS,
        "recurrent_steps": RECURRENT_STEPS,
        "sparsity": SPARSITY,
        "max_sentences": MAX_SENTENCES,
        "max_tokens": MAX_TOKENS,
        "selection_strategy": "top_k",
        "selection_k": TOP_K,
        "validation_loss": val_loss,
        "validation_rouge1_f1": metrics["rouge1"],
        "validation_rouge2_f1": metrics["rouge2"],
        "validation_rougeL_f1": metrics["rougeL"],
        "encoder_state_dict": encoder.state_dict(),
        "scorer_state_dict": scorer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for path in (TRAIN_PATH, VAL_PATH, VOCAB_PATH, POSITION_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    if EXPERIMENT_DIR.exists():
        raise FileExistsError(
            f"{EXPERIMENT_DIR} already exists. Rename it before rerunning; "
            "this script will not overwrite prior experiment results."
        )

    CHECKPOINT_DIR.mkdir(parents=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("PB-ANN POSITION-FEATURE ABLATION")
    print("=" * 68)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    vocab = Vocabulary.load(VOCAB_PATH)
    train_ds = SummarizationDataset(
        TRAIN_PATH, vocab, max_sentences=MAX_SENTENCES, max_tokens=MAX_TOKENS
    )
    val_ds = SummarizationDataset(
        VAL_PATH, vocab, max_sentences=MAX_SENTENCES, max_tokens=MAX_TOKENS
    )
    train_loader = create_dataloader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    val_records = load_records(VAL_PATH)

    positives = sum(sum(x["labels"]) for x in train_ds.examples)
    total = sum(len(x["labels"]) for x in train_ds.examples)
    negatives = total - positives
    pos_weight_value = min(negatives / max(positives, 1), MAX_POS_WEIGHT)
    pos_weight = torch.tensor([pos_weight_value], device=device)

    encoder = TextEncoder(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM
    ).to(device)
    scorer = PBANNNoPositionTextScorer(
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_rouge = float("-inf")
    best_epoch = 0
    no_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'=' * 68}\nEPOCH {epoch}/{EPOCHS}\n{'=' * 68}")

        train_loss = run_epoch(
            encoder, scorer, train_loader, optimizer, criterion, device, True
        )
        val_loss = run_epoch(
            encoder, scorer, val_loader, optimizer, criterion, device, False
        )

        print(f"Train loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")
        metrics = evaluate(
            encoder, scorer, val_records, vocab, device, with_position=False
        )
        print(
            f"ROUGE-1={metrics['rouge1']:.4f} "
            f"ROUGE-2={metrics['rouge2']:.4f} "
            f"ROUGE-L={metrics['rougeL']:.4f}"
        )

        epoch_path = CHECKPOINT_DIR / f"epoch_{epoch:02d}.pt"
        save_checkpoint(
            epoch_path, epoch, encoder, scorer, optimizer, val_loss, metrics
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_rouge1_f1": metrics["rouge1"],
            "validation_rouge2_f1": metrics["rouge2"],
            "validation_rougeL_f1": metrics["rougeL"],
        })

        if metrics["rougeL"] > best_rouge + MIN_ROUGE_IMPROVEMENT:
            best_rouge = metrics["rougeL"]
            best_epoch = epoch
            no_improvement = 0
            save_checkpoint(
                BEST_CHECKPOINT, epoch, encoder, scorer, optimizer, val_loss, metrics
            )
            print("Saved best ablation checkpoint.")
        else:
            no_improvement += 1
            print(f"No improvement: {no_improvement}/{EARLY_STOPPING_PATIENCE}")

        HISTORY_PATH.write_text(json.dumps({
            "experiment": "pbann_without_position_features",
            "seed": SEED,
            "best_epoch": best_epoch,
            "best_validation_rougeL_f1": best_rouge,
            "epochs": history,
        }, indent=2), encoding="utf-8")

        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    # Compare best ablation checkpoint against the existing position model.
    best = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    encoder.load_state_dict(best["encoder_state_dict"])
    scorer.load_state_dict(best["scorer_state_dict"])
    no_position_metrics = evaluate(
        encoder, scorer, val_records, vocab, device, with_position=False
    )

    position_encoder = TextEncoder(
        vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM
    ).to(device)
    position_scorer = PBANNPositionTextScorer(
        embedding_dim=EMBEDDING_DIM,
        num_neurons=NUM_NEURONS,
        steps=RECURRENT_STEPS,
        sparsity=SPARSITY,
        position_dim=POSITION_DIM,
    ).to(device)

    checkpoint = torch.load(
        POSITION_CHECKPOINT, map_location=device, weights_only=False
    )
    position_encoder.load_state_dict(checkpoint["encoder_state_dict"])
    position_scorer.load_state_dict(checkpoint["scorer_state_dict"])
    position_metrics = evaluate(
        position_encoder, position_scorer, val_records, vocab, device,
        with_position=True,
    )
    lead3_metrics = evaluate_lead3(val_records)

    comparison = {
        "validation_articles": len(val_records),
        "selection": "top-3, restored to original sentence order",
        "metrics": {
            "pbann_no_position": no_position_metrics,
            "pbann_position_aware": position_metrics,
            "lead3": lead3_metrics,
        },
        "best_ablation_epoch": best_epoch,
        "note": "Validation comparison only. The test set was not used for tuning.",
    }
    COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    print("\nFINAL VALIDATION COMPARISON")
    print(f"{'Model':28s} {'ROUGE-1':>9s} {'ROUGE-2':>9s} {'ROUGE-L':>9s}")
    for name, values in comparison["metrics"].items():
        print(
            f"{name:28s} {values['rouge1']:9.4f} "
            f"{values['rouge2']:9.4f} {values['rougeL']:9.4f}"
        )

    print("\nBest epoch:", best_epoch)
    print("Checkpoint:", BEST_CHECKPOINT)
    print("History:", HISTORY_PATH)
    print("Comparison:", COMPARISON_PATH)
    print("Existing model checkpoints were not modified.")


if __name__ == "__main__":
    main()