
import torch

from src.text_encoder import Vocabulary, TextEncoder, split_sentences
from src.pbann import PBANNTextScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

article = (
    "Artificial intelligence helps doctors analyse medical images. "
    "Hospitals use AI to support clinical decisions. "
    "Researchers are developing safer AI systems."
)

# 1. Split the article into sentences
sentences = split_sentences(article)

# 2. Build a vocabulary for this smoke test
vocab = Vocabulary()
vocab.build(sentences)

encoded = [vocab.encode(sentence) for sentence in sentences]
max_tokens = max(len(ids) for ids in encoded)

token_ids = torch.zeros(
    1, len(sentences), max_tokens, dtype=torch.long
)

for i, ids in enumerate(encoded):
    token_ids[0, i, :len(ids)] = torch.tensor(ids)

token_ids = token_ids.to(device)
token_mask = token_ids != 0
sentence_mask = token_mask.any(dim=-1)

# 3. Initialize both models
encoder = TextEncoder(
    vocab_size=len(vocab),
    embedding_dim=256,
).to(device)

pbann = PBANNTextScorer(
    embedding_dim=256,
    num_neurons=512,
    steps=6,
    sparsity=0.8,
).to(device)

# 4. Convert sentences into embeddings
sentence_embeddings = encoder(token_ids, token_mask)

# 5. Predict sentence importance scores
scores = pbann(sentence_embeddings, sentence_mask)

# 6. Test end-to-end gradient flow
loss = scores[sentence_mask].square().mean()
loss.backward()

print("Device:", device)
print("Number of sentences:", len(sentences))
print("Sentence embedding shape:", tuple(sentence_embeddings.shape))
print("Importance scores:", scores.detach().cpu().tolist())
print(
    "Encoder gradients:",
    encoder.embedding.weight.grad is not None,
)
print(
    "PB-ANN recurrent gradients:",
    pbann.recurrent_weight.grad is not None,
)
print("All gradients finite:", (
    torch.isfinite(encoder.embedding.weight.grad).all()
    and torch.isfinite(pbann.recurrent_weight.grad).all()
).item())
print("Pipeline test passed.")
