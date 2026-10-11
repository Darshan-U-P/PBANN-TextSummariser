
import json
import sys
from pathlib import Path
from statistics import mean

import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary
from src.text_encoder import TextEncoder
from src.pbann_position import PBANNPositionTextScorer
from scripts.train_position_ablation import PBANNNoPositionTextScorer


TEST_PATH = ROOT / "data/processed/test.jsonl"
VOCAB_PATH = ROOT / "models/vocab.json"

ABLATION_CHECKPOINT = (
    ROOT / "models/position_ablation/pbann_no_position_best.pt"
)
POSITION_CHECKPOINT = (
    ROOT / "models/position_experiment/pbann_position_best.pt"
)

OUTPUT_PATH = ROOT / "evaluation_results/position_ablation_test.json"

MAX_SENTENCES = 60
MAX_TOKENS = 64
EMBEDDING_DIM = 256
NUM_NEURONS = 512
RECURRENT_STEPS = 6
SPARSITY = 0.8
POSITION_DIM = 32
TOP_K = 3

ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_records(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sentences = [
                " ".join(str(s).split())
                for s in item.get("sentences", [])[:MAX_SENTENCES]
                if str(s).strip()
            ]
            summary = item.get("summary", "")
            if sentences and isinstance(summary, str) and summary.strip():
                records.append({
                    "sentences": sentences,
                    "reference": " ".join(summary.split()),
                })
    if not records:
        raise ValueError(f"No valid records in {path}")
    return records


def score_pair(prediction, reference):
    scores = ROUGE.score(reference, prediction)
    return {
        key: scores[key].fmeasure
        for key in ("rouge1", "rouge2", "rougeL")
    }


def mean_scores(rows):
    return {
        key: mean(row[key] for row in rows)
        for key in ("rouge1", "rouge2", "rougeL")
    }


def encode(sentences, vocab, device):
    sequences = [
        vocab.encode(s, max_tokens=MAX_TOKENS) for s in sentences
    ]
    width = max(max((len(s) for s in sequences), default=0), 1)

    ids = torch.zeros(
        (1, len(sequences), width),
        dtype=torch.long,
        device=device,
    )
    for i, sequence in enumerate(sequences):
        if sequence:
            ids[0, i, :len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )

    token_mask = ids != 0
    sentence_mask = torch.ones(
        (1, len(sequences)), dtype=torch.bool, device=device
    )
    return ids, token_mask, sentence_mask


def predict_model(encoder, scorer, record, vocab, device, with_position):
    sentences = record["sentences"]
    ids, token_mask, sentence_mask = encode(sentences, vocab, device)

    with torch.inference_mode():
        embeddings = encoder(ids, token_mask)

        if with_position:
            n = len(sentences)
            positions = (
                torch.linspace(
                    0.0, 1.0, n, device=device,
                    dtype=embeddings.dtype,
                ).unsqueeze(0)
                if n > 1
                else torch.zeros(
                    (1, n), device=device, dtype=embeddings.dtype
                )
            )
            logits = scorer(embeddings, sentence_mask, positions)
        else:
            logits = scorer(embeddings, sentence_mask)

        chosen = torch.topk(
            logits[0], k=min(TOP_K, len(sentences))
        ).indices.tolist()

    chosen.sort()
    prediction = " ".join(sentences[i] for i in chosen)
    return prediction


def load_model(checkpoint_path, vocab_size, device, with_position):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    encoder = TextEncoder(
        vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM
    ).to(device)

    if with_position:
        scorer = PBANNPositionTextScorer(
            embedding_dim=EMBEDDING_DIM,
            num_neurons=NUM_NEURONS,
            steps=RECURRENT_STEPS,
            sparsity=SPARSITY,
            position_dim=POSITION_DIM,
        ).to(device)
    else:
        scorer = PBANNNoPositionTextScorer(
            embedding_dim=EMBEDDING_DIM,
            num_neurons=NUM_NEURONS,
            steps=RECURRENT_STEPS,
            sparsity=SPARSITY,
        ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    scorer.load_state_dict(checkpoint["scorer_state_dict"])
    encoder.eval()
    scorer.eval()
    return encoder, scorer


def main():
    for path in (
        TEST_PATH, VOCAB_PATH,
        ABLATION_CHECKPOINT, POSITION_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(VOCAB_PATH)
    records = load_records(TEST_PATH)

    print("=" * 68)
    print("POSITION ABLATION — HELD-OUT TEST EVALUATION")
    print("=" * 68)
    print("Device:", device)
    print("Test articles:", len(records))

    no_pos_encoder, no_pos_scorer = load_model(
        ABLATION_CHECKPOINT, len(vocab), device, with_position=False
    )
    pos_encoder, pos_scorer = load_model(
        POSITION_CHECKPOINT, len(vocab), device, with_position=True
    )

    names = ("pbann_no_position", "pbann_position_aware", "lead3")
    per_article = {name: [] for name in names}

    for index, record in enumerate(records, start=1):
        no_pos_prediction = predict_model(
            no_pos_encoder, no_pos_scorer, record, vocab, device, False
        )
        pos_prediction = predict_model(
            pos_encoder, pos_scorer, record, vocab, device, True
        )
        lead_prediction = " ".join(record["sentences"][:TOP_K])

        per_article["pbann_no_position"].append(
            score_pair(no_pos_prediction, record["reference"])
        )
        per_article["pbann_position_aware"].append(
            score_pair(pos_prediction, record["reference"])
        )
        per_article["lead3"].append(
            score_pair(lead_prediction, record["reference"])
        )

        if index % 100 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}")

    means = {
        name: mean_scores(rows) for name, rows in per_article.items()
    }

    print("\nTEST RESULTS — ROUGE F1")
    print(f"{'Model':28s} {'ROUGE-1':>9s} {'ROUGE-2':>9s} {'ROUGE-L':>9s}")
    for name, metrics in means.items():
        print(
            f"{name:28s} "
            f"{metrics['rouge1']:9.4f} "
            f"{metrics['rouge2']:9.4f} "
            f"{metrics['rougeL']:9.4f}"
        )

    output = {
        "split": "test",
        "articles": len(records),
        "selection": "top-3 sentences in original article order",
        "mean_rouge_f1": means,
        "per_article_rouge_f1": per_article,
        "note": (
            "The test set was used for evaluation only. Do not use these "
            "results to choose epochs or tune hyperparameters."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    print("\nSaved:", OUTPUT_PATH)
    print("No model checkpoints were modified.")


if __name__ == "__main__":
    main()
