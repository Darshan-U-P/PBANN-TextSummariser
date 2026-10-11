
import torch

from src.text_encoder import (
    Vocabulary,
    TextEncoder,
    split_sentences,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

articles = [
    "Artificial intelligence helps doctors analyse medical images. "
    "Researchers are developing new AI systems."
]

vocab = Vocabulary()
vocab.build(articles)

sentences = split_sentences(articles[0])
encoded = [vocab.encode(sentence) for sentence in sentences]

max_tokens = max(map(len, encoded))

token_ids = torch.zeros(
    1, len(encoded), max_tokens, dtype=torch.long
)

for i, ids in enumerate(encoded):
    token_ids[0, i, :len(ids)] = torch.tensor(ids)

token_ids = token_ids.to(device)
token_mask = token_ids != 0

encoder = TextEncoder(
    vocab_size=len(vocab),
    embedding_dim=256,
).to(device)

output = encoder(token_ids, token_mask)

loss = output.square().mean()
loss.backward()

print("Device:", device)
print("Vocabulary size:", len(vocab))
print("Sentences:", len(sentences))
print("Embedding shape:", tuple(output.shape))
print(
    "Embedding gradients:",
    encoder.embedding.weight.grad is not None,
)
print("Test passed.")
