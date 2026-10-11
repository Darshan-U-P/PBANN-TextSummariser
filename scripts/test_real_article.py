from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import Vocabulary
from scripts.final_validation import DEVICE, load_checkpoint_model, predict_record

# ============================================================
# CONFIGURATION
# ============================================================

ARTICLE_PATH = ROOT / "data" / "real_article.txt"
VOCAB_PATH = ROOT / "models" / "vocab.json"
OUTPUT_PATH = (
    ROOT / "evaluation_results" / "final_validation" / "real_article_result.json"
)

SEED = 42
TOP_K_TO_PRINT = 10
SUMMARY_SENTENCES = 3
MAX_SENTENCES = 60


# ============================================================
# TEXT CLEANING AND SENTENCE SPLITTING
# ============================================================

def clean_article_text(text: str) -> str:
    """Normalize whitespace and remove a known leading webpage artefact."""
    text = text.replace("\ufeff", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Remove the AP timestamp prefix seen in the previous article extraction.
    # This pattern is anchored to the beginning to avoid changing article prose.
    text = re.sub(
        r"^\s*\d{1,2},\s+[A-Za-z]+\s+\d{4},\s*"
        r"\d{1,2}:\d{2}\s*(?:AM|PM)\s*(?:ET|EST|EDT|UTC)?\s*"
        r"\(AP\)\s*\.{0,3}\s*(?:\.\.\.\s*)?\(?Show more\)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split article text into sentences.
    Use NLTK when installed and tokenizer data is available; otherwise use a
    regular-expression fallback so no NLTK download is required.
    """
    try:
        import nltk

        try:
            parts = nltk.sent_tokenize(text)
        except LookupError:
            parts = re.split(
                r'(?<=[.!?])\s+(?=[A-Z0-9“"\[])',
                text,
            )
    except ImportError:
        parts = re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9“"\[])',
            text,
        )

    cleaned: list[str] = []

    for sentence in parts:
        sentence = re.sub(r"\s+", " ", sentence).strip(" \t\n-•")
        if not sentence:
            continue

        # Remove standalone webpage controls, not ordinary article mentions.
        if re.fullmatch(
            r"\(?\s*(show more|read more|listen|share|print)\s*\)?[.!]?",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue

        # Ignore very short fragments that are unlikely to be complete sentences.
        if len(sentence.split()) < 5:
            continue

        cleaned.append(sentence)

    return cleaned


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not ARTICLE_PATH.is_file():
        raise FileNotFoundError(
            f"Article file not found: {ARTICLE_PATH}\n"
            "Make sure data/real_article.txt exists in the project directory."
        )

    article = ARTICLE_PATH.read_text(encoding="utf-8").strip()

    if not article:
        raise ValueError(f"Article file is empty: {ARTICLE_PATH}")

    cleaned_text = clean_article_text(article)
    all_sentences = split_sentences(cleaned_text)

    if len(all_sentences) < SUMMARY_SENTENCES:
        raise RuntimeError(
            f"Only {len(all_sentences)} usable sentences were found in "
            f"{ARTICLE_PATH}. Inspect the file or provide more article text."
        )

    # The existing model adapter supports up to MAX_SENTENCES for this test.
    sentences = all_sentences[:MAX_SENTENCES]

    vocab = Vocabulary.load(VOCAB_PATH)
    model = load_checkpoint_model(SEED, len(vocab))

    record = {
        "id": "real_article",
        "sentences": sentences,
    }

    _, raw_scores = predict_record(model, record, vocab)

    usable = min(len(sentences), len(raw_scores))
    sentences = sentences[:usable]
    scores = [float(score) for score in raw_scores[:usable]]

    if usable < SUMMARY_SENTENCES:
        raise RuntimeError(
            f"The model returned scores for only {usable} sentences; "
            f"{SUMMARY_SENTENCES} are required."
        )

    # Rank by model score, then restore source order in the final summary.
    ranked_indices = sorted(
        range(usable),
        key=lambda index: scores[index],
        reverse=True,
    )
    chosen_indices = sorted(ranked_indices[:SUMMARY_SENTENCES])
    summary = " ".join(sentences[index] for index in chosen_indices)

    source_words = len(" ".join(sentences).split())
    summary_words = len(summary.split())
    compression = (
        100.0 * (source_words - summary_words) / source_words
        if source_words
        else 0.0
    )

    # Structural validation.
    copied = all(
        sentences[index] in sentences
        for index in chosen_indices
    )
    shorter = summary_words < source_words

    # Lead-3 baseline uses the same cleaned input.
    lead3 = " ".join(sentences[:SUMMARY_SENTENCES])

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("=" * 76)
    print("PB-ANN REAL ARTICLE TEST — FILE INPUT + TOP-10 RANKING")
    print("=" * 76)
    print(f"Article file: {ARTICLE_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint seed: {SEED}")
    print(f"Sentences extracted: {len(all_sentences)}")
    print(f"Sentences evaluated: {len(sentences)}")
    print(f"Input words evaluated: {source_words}")
    print()

    print("-" * 76)
    print(f"TOP {min(TOP_K_TO_PRINT, usable)} SENTENCES BY MODEL SCORE")
    print("-" * 76)

    for rank, index in enumerate(
        ranked_indices[:TOP_K_TO_PRINT],
        start=1,
    ):
        selected_mark = "  [SELECTED]" if index in chosen_indices else ""
        print(
            f"{rank:>2}. Source sentence #{index + 1} | "
            f"Score: {scores[index]:.4f}{selected_mark}"
        )
        print(f"    {sentences[index]}")
        print()

    print("-" * 76)
    print("PB-ANN EXTRACTIVE SUMMARY (SOURCE ORDER)")
    print("-" * 76)
    print(summary)
    print()

    print("-" * 76)
    print("LEAD-3 BASELINE")
    print("-" * 76)
    print(lead3)
    print()

    print("-" * 76)
    print("STRUCTURAL CHECKS")
    print("-" * 76)
    print(f"Source words: {source_words}")
    print(f"Summary words: {summary_words}")
    print(f"Compression: {compression:.2f}%")
    print(f"All selected sentences copied from source: {copied}")
    print(f"Summary shorter than input: {shorter}")

    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "article_path": str(ARTICLE_PATH),
        "device": str(DEVICE),
        "checkpoint_seed": SEED,
        "sentences_extracted": len(all_sentences),
        "input_sentence_count": len(sentences),
        "input_word_count": source_words,
        "summary_sentence_count": len(chosen_indices),
        "summary_word_count": summary_words,
        "compression_percent": round(compression, 2),
        "selected_sentence_indices_1_based": [
            index + 1 for index in chosen_indices
        ],
        "selected_sentences": [
            sentences[index] for index in chosen_indices
        ],
        "summary": summary,
        "top_ranked_sentences": [
            {
                "rank": rank,
                "source_sentence_index_1_based": index + 1,
                "score": scores[index],
                "selected": index in chosen_indices,
                "sentence": sentences[index],
            }
            for rank, index in enumerate(
                ranked_indices[:TOP_K_TO_PRINT],
                start=1,
            )
        ],
        "lead3_baseline": lead3,
        "structural_checks": {
            "selected_sentences_copied_from_source": copied,
            "summary_shorter_than_input": shorter,
        },
        "note": (
            "Scores are raw model logits, not probabilities. "
            f"Only the first {MAX_SENTENCES} sentences are evaluated. "
            "Lead-3 uses the same cleaned input."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved result: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
