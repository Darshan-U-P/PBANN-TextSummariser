
import json
from pathlib import Path

from nltk import sent_tokenize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "train.jsonl"

with DATA_PATH.open("r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        if index >= 3:
            break

        example = json.loads(line)

        print("\n" + "=" * 80)
        print("EXAMPLE:", example["id"])
        print("\nREFERENCE SUMMARY:")
        print(example["summary"])

        print("\nSELECTED SENTENCES:")
        selected_count = 0

        for sentence, label in zip(
            example["sentences"], example["labels"]
        ):
            if label == 1:
                selected_count += 1
                print(f"[SELECTED] {sentence}")

        print(f"\nSelected: {selected_count}")
        print("Total source sentences:", len(example["sentences"]))
