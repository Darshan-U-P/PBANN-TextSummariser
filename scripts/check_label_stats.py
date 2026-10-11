
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "processed" / "train.jsonl"

selected_counts = Counter()
examples = 0
empty_examples = 0
total_sentences = 0

with PATH.open("r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        labels = record["labels"]

        examples += 1
        selected_counts[sum(labels)] += 1
        total_sentences += len(labels)

        if sum(labels) == 0:
            empty_examples += 1

print("Examples:", examples)
print("Total source sentences:", total_sentences)
print("Average sentences/article:", round(total_sentences / max(examples, 1), 2))
print("Examples with zero selected sentences:", empty_examples)
print("Selected sentences per article:", dict(sorted(selected_counts.items())))
