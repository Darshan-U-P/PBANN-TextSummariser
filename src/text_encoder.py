
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def split_sentences(text):
    """Split an article into sentences."""
    return [
        sentence.strip()
        for sentence in nltk.sent_tokenize(text)
        if sentence.strip()
    ]


def tokenize(text):
    """Basic word and punctuation tokenizer."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocabulary:
    """Vocabulary built from the training corpus."""

    def __init__(self, min_frequency=1):
        self.min_frequency = min_frequency
        self.token_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }

    def build(self, texts):
        counts = Counter()

        for text in texts:
            counts.update(tokenize(text))

        for word, count in counts.most_common():
            if count >= self.min_frequency and word not in self.token_to_id:
                self.token_to_id[word] = len(self.token_to_id)

    def encode(self, text, max_tokens=64):
        tokens = tokenize(text)[:max_tokens]

        if not tokens:
            tokens = [UNK_TOKEN]

        return [
            self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])
            for token in tokens
        ]

    def __len__(self):
        return len(self.token_to_id)


class TextEncoder(nn.Module):
    """
    Trainable text encoder.

    Input:
        token_ids: [batch, sentences, tokens]
        token_mask: [batch, sentences, tokens]
                   True for real tokens.

    Output:
        sentence_embeddings: [batch, sentences, embedding_dim]
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, token_ids, token_mask=None):
        # token_ids: [B, S, T]
        embeddings = self.embedding(token_ids)

        if token_mask is None:
            token_mask = token_ids != 0
        else:
            token_mask = token_mask.bool()

        # Masked mean pooling over words in each sentence
        mask = token_mask.unsqueeze(-1).to(embeddings.dtype)

        summed = (embeddings * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp(min=1.0)

        sentence_vectors = summed / counts

        # Learn a transformation into PB-ANN input space
        sentence_vectors = self.projection(sentence_vectors)

        # Keep completely padded sentences at zero
        valid_sentences = token_mask.any(dim=2)
        sentence_vectors = sentence_vectors * valid_sentences.unsqueeze(-1)

        return sentence_vectors
