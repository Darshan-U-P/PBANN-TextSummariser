
import json
from pathlib import Path

from datasets import load_dataset


# Project root is the parent of this script's directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_NAME = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"

TRAIN_LIMIT = 10_000
VALIDATION_LIMIT = 500
TEST_LIMIT = 500
SEED = 42


def save_jsonl(dataset, output_path):
    """Write article-summary examples as UTF-8 JSON Lines."""
    count = 0

    with output_path.open("w", encoding="utf-8") as file:
        for example in dataset:
            article = example["article"].strip()
            summary = example["highlights"].strip()

            if not article or not summary:
                continue

            record = {
                "id": str(example["id"]),
                "article": article,
                "summary": summary,
                "language": "en",
                "source": DATASET_NAME,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            count += 1

    return count


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CNN/DailyMail...")
    print("This may take time on the first run.")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    # Shuffle deterministically so repeated runs are reproducible.
    train = dataset["train"].shuffle(seed=SEED)
    validation = dataset["validation"]
    test = dataset["test"]

    splits = [
        ("train", train.select(range(min(TRAIN_LIMIT, len(train))))),
        (
            "validation",
            validation.select(
                range(min(VALIDATION_LIMIT, len(validation)))
            ),
        ),
        (
            "test",
            test.select(range(min(TEST_LIMIT, len(test)))),
        ),
    ]

    for split_name, split_data in splits:
        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        count = save_jsonl(split_data, output_path)

        print(f"{split_name}: {count:,} examples")
        print(f"Saved: {output_path}")

    print("\nDataset preparation complete.")
    print("Note: CNN/DailyMail is English-only.")


if __name__ == "__main__":
    main()
