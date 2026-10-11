
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "models" / "vocab.json"
DATA_DIR = ROOT / "data" / "processed"

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

def main():
    with VOCAB_PATH.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"Vocabulary size: {len(vocab):,}")
    print("-" * 60)

    overall_total = 0
    overall_unknown = 0

    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.jsonl"

        if not path.exists():
            print(f"{split}: FILE NOT FOUND")
            continue

        total = 0
        unknown = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)

                for sentence in record["sentences"][:60]:
                    tokens = tokenize(sentence)[:64]
                    total += len(tokens)
                    unknown += sum(
                        token not in vocab for token in tokens
                    )

        rate = unknown / max(total, 1)
        print(
            f"{split}: {unknown:,}/{total:,} "
            f"unknown tokens ({rate:.2%})"
        )

        overall_total += total
        overall_unknown += unknown

    print("-" * 60)
    print(
        f"OVERALL: {overall_unknown:,}/{overall_total:,} "
        f"unknown tokens "
        f"({overall_unknown / max(overall_total, 1):.2%})"
    )

if __name__ == "__main__":
    main()
