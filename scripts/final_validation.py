
from __future__ import annotations

import inspect
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary, collate_summaries
from scripts.train_position_multiseed import make_model, get_logits


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
VOCAB_PATH = ROOT / "models" / "vocab.json"
CHECKPOINT_DIR = ROOT / "models" / "multiseed"
OUTPUT_DIR = ROOT / "evaluation_results" / "final_validation"

SEEDS = [42, 123, 2026]
SAMPLE_COUNT = 5
SUMMARY_SENTENCES = 3
RANDOM_SEED = 20261012
MAX_SENTENCES = 60
MAX_TOKENS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS = ("rouge1", "rouge2", "rougeL")


# ============================================================
# DATA AND METRICS
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                record = json.loads(line)
                if "sentences" not in record:
                    raise ValueError(
                        f"Record at line {line_number} has no 'sentences' field."
                    )
                if "summary" not in record:
                    raise ValueError(
                        f"Record at line {line_number} has no 'summary' field."
                    )
                records.append(record)

    return records


def count_words(text: str) -> int:
    return len(text.split())


def mean_metric(scores: list[dict], metric: str) -> float:
    if not scores:
        return 0.0
    return float(np.mean([score[metric].fmeasure for score in scores]))


def select_top_sentences(sentences, logits, limit=3):
    """Select the highest-scoring sentences, retaining source order."""
    values = torch.as_tensor(logits).detach().float().cpu().flatten()
    count = min(len(sentences), len(values))

    if count == 0:
        return []

    ranked_indices = sorted(
        range(count),
        key=lambda index: float(values[index]),
        reverse=True,
    )
    selected_indices = sorted(ranked_indices[:limit])

    return [sentences[index] for index in selected_indices]


# ============================================================
# MODEL FACTORY / INFERENCE ADAPTERS
# ============================================================

def call_make_model(vocab_size: int, variant: str, seed: int):
    """
    Call the project's existing model factory using recognized parameter
    names. Stop if required parameters cannot be identified safely.
    """
    signature = inspect.signature(make_model)

    possible_values = {
        "vocab_size": vocab_size,
        "vocab": vocab_size,
        "variant": variant,
        "model_type": variant,
        "mode": variant,
        "use_position": variant == "position",
        "position": variant == "position",
        "device": DEVICE,
        "seed": seed,
        "random_seed": seed,
    }

    kwargs = {}
    missing = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in possible_values:
            kwargs[name] = possible_values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Cannot call make_model{signature}. "
            f"Unrecognized required parameters: {missing}. "
            "Adapt call_make_model() to your local training script."
        )

    return make_model(**kwargs)


def call_get_logits(model_or_pair, batch):
    """
    Call get_logits with recognized arguments.
    The returned value is normalized by extract_logits().
    """
    signature = inspect.signature(get_logits)

    possible_values = {
        "model": model_or_pair,
        "net": model_or_pair,
        "encoder": (
            model_or_pair[0]
            if isinstance(model_or_pair, (tuple, list))
            else getattr(model_or_pair, "encoder", None)
        ),
        "scorer": (
            model_or_pair[1]
            if isinstance(model_or_pair, (tuple, list))
            else getattr(model_or_pair, "scorer", None)
        ),
        "batch": batch,
        "data": batch,
        "inputs": batch,
        "device": DEVICE,
        "target_device": DEVICE,
    }

    kwargs = {}
    missing = []

    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in possible_values and possible_values[name] is not None:
            kwargs[name] = possible_values[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Cannot call get_logits{signature}. "
            f"Unrecognized required parameters: {missing}. "
            "Adapt call_get_logits() to your local training script."
        )

    return get_logits(**kwargs)


def extract_logits(result):
    """Normalize common output formats into a tensor."""
    if isinstance(result, dict):
        for key in ("logits", "scores", "sentence_logits", "output"):
            if key in result:
                result = result[key]
                break
        else:
            raise TypeError(
                f"Unknown get_logits() dictionary keys: {list(result.keys())}"
            )

    if isinstance(result, (tuple, list)):
        if not result:
            raise ValueError("get_logits() returned an empty sequence.")
        result = result[0]

    if not isinstance(result, torch.Tensor):
        result = torch.as_tensor(result)

    return result.detach().float().cpu()


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint_model(seed: int, vocab_size: int):
    """
    Load the checkpoint format used by this project:
      encoder_state_dict
      scorer_state_dict
    """
    checkpoint_path = CHECKPOINT_DIR / f"position_seed_{seed}_best.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}."
        )

    required_keys = ("encoder_state_dict", "scorer_state_dict")
    missing = [key for key in required_keys if key not in checkpoint]

    if missing:
        raise KeyError(
            f"Checkpoint is missing {missing}. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    if checkpoint.get("model_type") not in (None, "position"):
        raise ValueError(
            f"Expected a position-aware checkpoint, got "
            f"model_type={checkpoint.get('model_type')!r}."
        )

    model = call_make_model(vocab_size, "position", seed)

    # The factory may return (encoder, scorer).
    if isinstance(model, (tuple, list)):
        if len(model) != 2:
            raise RuntimeError(
                "make_model() returned a tuple/list that does not contain "
                "exactly (encoder, scorer)."
            )

        encoder, scorer = model
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        scorer.load_state_dict(checkpoint["scorer_state_dict"])

        encoder.to(DEVICE).eval()
        scorer.to(DEVICE).eval()

        return encoder, scorer

    # Or it may return a wrapper containing those modules.
    if hasattr(model, "encoder") and hasattr(model, "scorer"):
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.scorer.load_state_dict(checkpoint["scorer_state_dict"])

        model.to(DEVICE).eval()
        return model

    # Do not guess how to combine two state dictionaries.
    raise RuntimeError(
        "make_model() must return either (encoder, scorer) or a wrapper "
        "with .encoder and .scorer attributes. "
        f"Returned type: {type(model).__name__}. "
        "Inspect make_model() in scripts/train_position_multiseed.py."
    )


# ============================================================
# INFERENCE
# ============================================================

def predict_record(model, record: dict, vocab: Vocabulary):
    """
    Infer sentence scores without using reference summaries or labels.
    """
    sentences = record["sentences"][:MAX_SENTENCES]

    if not sentences:
        return [], []

    encoded_sentences = [
        vocab.encode(sentence, max_tokens=MAX_TOKENS)
        for sentence in sentences
    ]

    item = {
        "id": str(record.get("id", "unknown")),
        "token_ids": encoded_sentences,
        "labels": [0] * len(encoded_sentences),
    }

    batch = collate_summaries([item])
    batch = {
        key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }

    with torch.inference_mode():
        raw_logits = call_get_logits(model, batch)
        logits = extract_logits(raw_logits)

    # Remove only singleton batch dimensions.
    while logits.ndim > 1 and logits.shape[0] == 1:
        logits = logits.squeeze(0)

    logits = logits.flatten()
    usable_count = min(len(sentences), logits.numel())
    sentences = sentences[:usable_count]
    logits = logits[:usable_count]

    selected_sentences = select_top_sentences(
        sentences,
        logits,
        SUMMARY_SENTENCES,
    )

    return selected_sentences, logits.tolist()


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(TEST_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    if not records:
        raise RuntimeError("The test set is empty.")

    scorer = rouge_scorer.RougeScorer(
        list(METRICS),
        use_stemmer=True,
    )

    print("=" * 76)
    print("PB-ANN TEXT SUMMARISER — FINAL HELD-OUT TEST EVALUATION")
    print("=" * 76)
    print(f"Device: {DEVICE}")
    print(f"Test articles: {len(records)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Seeds: {SEEDS}")
    print("Training: NOT performed")
    print("Checkpoint modification: NONE")
    print()

    sample_rng = random.Random(RANDOM_SEED)
    sample_indices = sorted(
        sample_rng.sample(
            range(len(records)),
            min(SAMPLE_COUNT, len(records)),
        )
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "test_file": str(TEST_PATH),
        "test_articles": len(records),
        "seeds": SEEDS,
        "sample_checks": [],
        "metrics": {},
        "limitations": [
            "ROUGE measures lexical overlap with reference summaries, not factual correctness.",
            "Training sentence labels were heuristic, not manually annotated.",
            "Lead-3 is a baseline; implementation of extractive summarization does not require beating it.",
            "Do not use the held-out test results to tune the model.",
        ],
    }

    all_seed_scores = {}
    sample_outputs_by_seed = {}

    # --------------------------------------------------------
    # Evaluate each saved position-aware model.
    # --------------------------------------------------------

    for seed in SEEDS:
        print(f"\nLoading position-aware checkpoint, seed {seed}...")
        model = load_checkpoint_model(seed, len(vocab))

        article_scores = []
        sample_outputs = {}

        for index, record in enumerate(records):
            reference = record.get("summary", "").strip()

            selected_sentences, _ = predict_record(model, record, vocab)
            prediction = " ".join(selected_sentences)

            article_scores.append(scorer.score(reference, prediction))

            if index in sample_indices:
                source_sentences = record["sentences"][:MAX_SENTENCES]
                source_text = " ".join(source_sentences)

                source_words = count_words(source_text)
                summary_words = count_words(prediction)

                copied = all(
                    sentence in source_sentences
                    for sentence in selected_sentences
                )
                shorter = summary_words < source_words

                compression = (
                    100.0 * (source_words - summary_words) / source_words
                    if source_words > 0
                    else 0.0
                )

                sample_outputs[index] = {
                    "id": record.get("id", str(index)),
                    "reference_summary": reference,
                    "source_sentences": source_sentences,
                    "predicted_sentences": selected_sentences,
                    "summary": prediction,
                    "source_word_count": source_words,
                    "summary_word_count": summary_words,
                    "compression_percent": round(compression, 2),
                    "sentences_copied_from_source": copied,
                    "shorter_than_article": shorter,
                    "checks_passed": copied and shorter and bool(prediction),
                }

        all_seed_scores[seed] = article_scores
        sample_outputs_by_seed[seed] = sample_outputs

        print(f"Completed seed {seed}: {len(article_scores)} articles")
        for metric in METRICS:
            print(f"  {metric}: {mean_metric(article_scores, metric):.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Evaluate Lead-3 on exactly the same test examples.
    # --------------------------------------------------------

    lead3_scores = []

    for record in records:
        sentences = record["sentences"][:MAX_SENTENCES]
        reference = record.get("summary", "").strip()
        prediction = " ".join(sentences[:SUMMARY_SENTENCES])
        lead3_scores.append(scorer.score(reference, prediction))

    report["metrics"]["lead3"] = {
        metric: mean_metric(lead3_scores, metric)
        for metric in METRICS
    }

    report["metrics"]["position_pbann_by_seed"] = {
        str(seed): {
            metric: mean_metric(all_seed_scores[seed], metric)
            for metric in METRICS
        }
        for seed in SEEDS
    }

    report["metrics"]["position_pbann_mean_across_seeds"] = {
        metric: float(np.mean([
            mean_metric(all_seed_scores[seed], metric)
            for seed in SEEDS
        ]))
        for metric in METRICS
    }

    report["metrics"]["position_pbann_minus_lead3"] = {
        metric: (
            report["metrics"]["position_pbann_mean_across_seeds"][metric]
            - report["metrics"]["lead3"][metric]
        )
        for metric in METRICS
    }

    # --------------------------------------------------------
    # Display held-out examples from seed 42.
    # --------------------------------------------------------

    primary_seed = SEEDS[0]

    print("\n" + "=" * 76)
    print(f"UNSEEN TEST ARTICLE EXAMPLES — SEED {primary_seed}")
    print("=" * 76)

    for index in sample_indices:
        item = sample_outputs_by_seed[primary_seed][index]
        report["sample_checks"].append(item)

        print(f"\nArticle ID: {item['id']}")
        print(f"Original words: {item['source_word_count']}")
        print(f"Summary words:  {item['summary_word_count']}")
        print(f"Compression:    {item['compression_percent']}%")
        print(f"Copied from source: {item['sentences_copied_from_source']}")
        print(f"Shorter than article: {item['shorter_than_article']}")
        print("\nSUMMARY:")
        print(item["summary"] or "[EMPTY SUMMARY]")
        print("\nREFERENCE SUMMARY:")
        print(item["reference_summary"] or "[NO REFERENCE SUMMARY]")
        print("-" * 76)

    # --------------------------------------------------------
    # Final metric comparison.
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("FINAL TEST METRICS")
    print("=" * 76)
    print(f"{'Metric':<12}{'Lead-3':>12}{'PB-ANN mean':>16}{'Difference':>16}")

    for metric in METRICS:
        lead = report["metrics"]["lead3"][metric]
        pbann = report["metrics"]["position_pbann_mean_across_seeds"][metric]
        difference = pbann - lead

        print(
            f"{metric:<12}{lead:>12.4f}{pbann:>16.4f}{difference:>+16.4f}"
        )

    passed = sum(
        item["checks_passed"]
        for item in report["sample_checks"]
    )
    total = len(report["sample_checks"])

    report["sample_checks_passed"] = passed
    report["sample_checks_total"] = total

    differences = report["metrics"]["position_pbann_minus_lead3"]

    if all(value > 0 for value in differences.values()):
        conclusion = (
            "PB-ANN has higher mean ROUGE on all three metrics than Lead-3 "
            "on this test set. This does not, by itself, establish statistical "
            "significance."
        )
    elif all(value < 0 for value in differences.values()):
        conclusion = (
            "Lead-3 has higher mean ROUGE on all three metrics on this test set."
        )
    else:
        conclusion = (
            "Results are mixed across ROUGE metrics. The evaluation does not "
            "show a consistent overall winner over Lead-3."
        )

    report["conclusion"] = conclusion

    json_path = OUTPUT_DIR / "final_test_report.json"
    markdown_path = OUTPUT_DIR / "final_test_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    lines = [
        "# PB-ANN Text Summariser — Final Test Report",
        "",
        f"- Evaluation time (UTC): {report['created_utc']}",
        f"- Device: {DEVICE}",
        f"- Test articles: {len(records)}",
        f"- Seeds: {SEEDS}",
        f"- Sample checks passed: {passed}/{total}",
        "",
        "## ROUGE results",
        "",
        "| Metric | Lead-3 | PB-ANN mean across seeds | Difference |",
        "|---|---:|---:|---:|",
    ]

    for metric in METRICS:
        lead = report["metrics"]["lead3"][metric]
        pbann = report["metrics"]["position_pbann_mean_across_seeds"][metric]
        difference = pbann - lead

        lines.append(
            f"| {metric} | {lead:.4f} | {pbann:.4f} | {difference:+.4f} |"
        )

    lines += [
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "## Sample output checks",
        "",
        "| Article ID | Source words | Summary words | Compression | Copied source sentences | Shorter |",
        "|---|---:|---:|---:|---|---|",
    ]

    for item in report["sample_checks"]:
        lines.append(
            f"| {item['id']} | {item['source_word_count']} | "
            f"{item['summary_word_count']} | {item['compression_percent']:.2f}% | "
            f"{item['sentences_copied_from_source']} | "
            f"{item['shorter_than_article']} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        *[f"- {note}" for note in report["limitations"]],
        "",
    ]

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 76)
    print("FINAL CONCLUSION")
    print("=" * 76)
    print(conclusion)
    print(f"\nSample checks passed: {passed}/{total}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print("\nNo training was performed and no checkpoint was modified.")


if __name__ == "__main__":
    main()
