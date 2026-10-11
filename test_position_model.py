import torch
from src.pbann_position import PBANNPositionTextScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PBANNPositionTextScorer(
    embedding_dim=256,
    num_neurons=512,
    steps=6,
    sparsity=0.8,
).to(device)

batch_size = 2
num_sentences = 10

embeddings = torch.randn(
    batch_size, num_sentences, 256, device=device
)
mask = torch.ones(
    batch_size, num_sentences, dtype=torch.bool, device=device
)
positions = torch.linspace(
    0, 1, num_sentences, device=device
).unsqueeze(0).expand(batch_size, -1)

with torch.no_grad():
    logits = model(embeddings, mask, positions)
    scores = torch.sigmoid(logits)

print("Device:", device)
print("Logits shape:", tuple(logits.shape))
print("Scores shape:", tuple(scores.shape))
print("All scores finite:", torch.isfinite(scores).all().item())
print("Model test passed.")
