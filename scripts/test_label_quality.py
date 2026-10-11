
import json
from pathlib import Path

from rouge_score import rouge_scorer

from generate_labels import (
    select_oracle_sentences,
    split_article,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "train.jsonl"

scorer = rouge_scorer.RougeScorer(
    ["rouge1"],
    use_stemmer=True,
)

with DATA_PATH.open("r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index >= 5:
            break

        example = json.loads(line)
        sentences = split_article(example["article"])

        labels = select_oracle_sentences(
            sentences,
            example["summary"],
            scorer,
        )

        print("\n" + "=" * 70)
        print("REFERENCE:")
        print(example["summary"])
        print("\nSELECTED SENTENCES:")

        for sentence, label in zip(sentences, labels):
            if label:
                print("-", sentence)

        print("Selected:", sum(labels))
