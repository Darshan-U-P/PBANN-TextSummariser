
import json
import re
from pathlib import Path

from nltk import sent_tokenize
from rouge_score import rouge_scorer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABEL_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ("train", "validation", "test")

MAX_SELECTED_SENTENCES = 3
MIN_SENTENCE_WORDS = 4
MIN_ROUGE_IMPROVEMENT = 0.005

PROGRESS_INTERVAL = 1000


# ============================================================
# Text preprocessing
# ============================================================

def split_article(article):
    """Split an article into non-empty sentences."""
    return [
        sentence.strip()
        for sentence in sent_tokenize(article)
        if sentence.strip()
    ]


def normalize(text):
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def is_valid_candidate(sentence):
    """
    Reject obvious fragments and low-information candidates.

    This is a heuristic filter, not a semantic relevance detector.
    """
    words = re.findall(
        r"\b[\w'-]+\b",
        sentence,
        flags=re.UNICODE,
    )

    if len(words) < MIN_SENTENCE_WORDS:
        return False

    alphabetic_words = [
        word
        for word in words
        if any(char.isalpha() for char in word)
    ]

    if len(alphabetic_words) < 3:
        return False

    letters = sum(char.isalpha() for char in sentence)

    if letters / max(len(sentence), 1) < 0.5:
        return False

    return True


# ============================================================
# Sentence label generation
# ============================================================

def select_oracle_sentences(sentences, reference, scorer):
    """
    Generate binary sentence-selection labels using greedy ROUGE-1.

    A candidate is selected only if it improves the current
    ROUGE-1 F-measure by at least MIN_ROUGE_IMPROVEMENT.

    Returns:
        labels: list[int], one binary label per source sentence.

    Note:
        These are heuristic oracle labels, not human annotations.
    """
    labels = [0] * len(sentences)

    reference = normalize(reference)

    if not reference:
        return labels

    candidates = [
        index
        for index, sentence in enumerate(sentences)
        if is_valid_candidate(sentence)
    ]

    if not candidates:
        return labels

    selected = set()
    current_summary = ""

    current_score = scorer.score(
        reference,
        current_summary,
    )["rouge1"].fmeasure

    for _ in range(min(MAX_SELECTED_SENTENCES, len(candidates))):
        best_index = None
        best_score = current_score

        for index in candidates:
            if index in selected:
                continue

            proposed_summary = normalize(
                f"{current_summary} {sentences[index]}"
            )

            score = scorer.score(
                reference,
                proposed_summary,
            )["rouge1"].fmeasure

            if score > best_score:
                best_score = score
                best_index = index

        # No candidate improves the current summary.
        if best_index is None:
            break

        improvement = best_score - current_score

        # Stop when the improvement is negligible.
        if improvement < MIN_ROUGE_IMPROVEMENT:
            break

        selected.add(best_index)
        labels[best_index] = 1

        current_summary = normalize(
            f"{current_summary} {sentences[best_index]}"
        )

        current_score = best_score

    return labels


# ============================================================
# Process one dataset split
# ============================================================

def process_split(split_name, scorer):
    input_path = RAW_DIR / f"{split_name}.jsonl"
    output_path = LABEL_DIR / f"{split_name}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    positive_labels = 0
    zero_label_examples = 0

    with (
        input_path.open("r", encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path}, "
                    f"line {line_number}: {exc}"
                ) from exc

            article = record.get("article", "").strip()
            reference = record.get("summary", "").strip()

            if not article or not reference:
                skipped += 1
                continue

            sentences = split_article(article)

            if not sentences:
                skipped += 1
                continue

            labels = select_oracle_sentences(
                sentences,
                reference,
                scorer,
            )

            selected_count = sum(labels)

            if selected_count == 0:
                zero_label_examples += 1

            output_record = {
                "id": record.get("id", str(line_number)),
                "sentences": sentences,
                "labels": labels,
                "summary": reference,
                "language": record.get("language", "en"),
                "source": record.get("source", "unknown"),
            }

            destination.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                ) + "\n"
            )

            processed += 1
            positive_labels += selected_count

            if processed % PROGRESS_INTERVAL == 0:
                print(
                    f"{split_name}: processed {processed:,}"
                )

    average_selected = (
        positive_labels / processed if processed else 0.0
    )

    print(f"\n{split_name} results")
    print("-" * 45)
    print(f"Examples processed:       {processed:,}")
    print(f"Examples skipped:         {skipped:,}")
    print(f"Positive sentence labels: {positive_labels:,}")
    print(f"Zero-label examples:      {zero_label_examples:,}")
    print(f"Average selected/example: {average_selected:.2f}")
    print(f"Output: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("PB-ANN sentence-label generation")
    print(f"Input directory:  {RAW_DIR}")
    print(f"Output directory: {LABEL_DIR}")
    print(f"Maximum selections/article: {MAX_SELECTED_SENTENCES}")
    print(f"Minimum ROUGE improvement: {MIN_ROUGE_IMPROVEMENT}")
    print()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1"],
        use_stemmer=True,
    )

    for split_name in SPLITS:
        process_split(split_name, scorer)

    print("\nAll dataset splits processed.")
    print(
        "Reminder: ROUGE-derived labels are heuristic targets "
        "and require quality inspection."
    )


if __name__ == "__main__":
    main()
