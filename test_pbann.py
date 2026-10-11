
import torch
from src.pbann import PBANNTextScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PBANNTextScorer(
    embedding_dim=256,
    num_neurons=512,
    steps=6,
    sparsity=0.8,
).to(device)

embeddings = torch.randn(2, 5, 256, device=device)

mask = torch.tensor(
    [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ],
    device=device,
)

scores = model(embeddings, mask)
loss = scores[mask].square().mean()
loss.backward()

print("Device:", device)
print("Output shape:", tuple(scores.shape))
print("Gradient exists:", model.recurrent_weight.grad is not None)
print(
    "Gradient finite:",
    torch.isfinite(model.recurrent_weight.grad).all().item(),
)
print("Test passed.")
