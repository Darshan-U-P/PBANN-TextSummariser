
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    """
    Unicode-aware tokenizer.
    Preserves words from English and Indic scripts,
    along with punctuation.
    """
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    """Vocabulary constructed only from training data."""

    def __init__(self, min_frequency=2, max_size=50000):
        self.min_frequency = min_frequency
        self.max_size = max_size

        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

    def build(self, sentences):
        counts = Counter()

        for sentence in sentences:
            counts.update(tokenize(sentence))

        for token, count in counts.most_common():
            if count < self.min_frequency:
                break

            if token in self.token_to_id:
                continue

            if len(self.token_to_id) >= self.max_size:
                break

            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text, max_tokens=64):
        tokens = tokenize(text)[:max_tokens]

        if not tokens:
            return [self.token_to_id[UNK_TOKEN]]

        return [
            self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])
            for token in tokens
        ]

    def __len__(self):
        return len(self.token_to_id)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.token_to_id,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as file:
            token_to_id = json.load(file)

        vocab = cls()
        vocab.token_to_id = {
            token: int(index)
            for token, index in token_to_id.items()
        }
        return vocab


class SummarizationDataset(Dataset):
    """Load preprocessed article sentences and binary labels."""

    def __init__(
        self,
        jsonl_path,
        vocab,
        max_sentences=60,
        max_tokens=64,
    ):
        self.vocab = vocab
        self.max_sentences = max_sentences
        self.max_tokens = max_tokens
        self.examples = []

        path = Path(jsonl_path)

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                record = json.loads(line)
                sentences = record["sentences"]
                labels = record["labels"]

                if len(sentences) != len(labels):
                    raise ValueError(
                        f"Sentence/label mismatch in {path}, "
                        f"line {line_number}"
                    )

                # Keep sentence-label alignment while truncating.
                sentences = sentences[:max_sentences]
                labels = labels[:max_sentences]

                encoded_sentences = [
                    vocab.encode(
                        sentence,
                        max_tokens=max_tokens,
                    )
                    for sentence in sentences
                ]

                self.examples.append({
                    "id": record.get("id", str(line_number)),
                    "token_ids": encoded_sentences,
                    "labels": [int(label) for label in labels],
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_summaries(batch):
    """
    Pad a batch to [B, S, T].

    Returns:
        token_ids:     [B, S, T]
        token_mask:    [B, S, T]
        sentence_mask: [B, S]
        labels:        [B, S]
        ids:           list[str]
    """
    batch_size = len(batch)

    max_sentences = max(
        len(item["token_ids"]) for item in batch
    )
    max_tokens = max(
        len(sentence)
        for item in batch
        for sentence in item["token_ids"]
    )

    token_ids = torch.zeros(
        batch_size,
        max_sentences,
        max_tokens,
        dtype=torch.long,
    )

    labels = torch.zeros(
        batch_size,
        max_sentences,
        dtype=torch.float32,
    )

    sentence_mask = torch.zeros(
        batch_size,
        max_sentences,
        dtype=torch.bool,
    )

    for batch_index, item in enumerate(batch):
        for sentence_index, ids in enumerate(item["token_ids"]):
            token_ids[
                batch_index, sentence_index, :len(ids)
            ] = torch.tensor(ids, dtype=torch.long)

            labels[batch_index, sentence_index] = (
                item["labels"][sentence_index]
            )

            sentence_mask[batch_index, sentence_index] = True

    token_mask = token_ids != 0

    return {
        "token_ids": token_ids,
        "token_mask": token_mask,
        "sentence_mask": sentence_mask,
        "labels": labels,
        "ids": [item["id"] for item in batch],
    }


def build_vocabulary(
    train_jsonl,
    min_frequency=2,
    max_size=50000,
):
    """Build vocabulary from training sentences only."""
    path = Path(train_jsonl)
    vocab = Vocabulary(
        min_frequency=min_frequency,
        max_size=max_size,
    )

    def training_sentences():
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                yield from record["sentences"]

    vocab.build(training_sentences())
    return vocab


def create_dataloader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_summaries,
        pin_memory=torch.cuda.is_available(),
    )
