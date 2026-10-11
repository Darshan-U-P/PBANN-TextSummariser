import torch
import torch.nn as nn
import torch.nn.functional as F


class PBANNPositionTextScorer(nn.Module):
    """
    Position-aware PB-ANN for extractive text summarisation.

    Inputs:
        sentence_embeddings: [batch, num_sentences, embedding_dim]
        sentence_mask:       [batch, num_sentences], True = valid
        sentence_positions:  [batch, num_sentences], normalized to [0, 1]

    Output:
        importance_logits:   [batch, num_sentences]
    """

    def __init__(
        self,
        embedding_dim=256,
        num_neurons=512,
        steps=6,
        leak=0.2,
        inhibition_strength=0.1,
        sparsity=0.8,
        dropout=0.05,
        position_dim=32,
    ):
        super().__init__()

        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")
        if position_dim < 1:
            raise ValueError("position_dim must be positive.")
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        self.embedding_dim = embedding_dim
        self.num_neurons = num_neurons
        self.steps = steps
        self.leak = leak
        self.inhibition_strength = inhibition_strength

        # Encode normalized sentence position.
        self.position_encoder = nn.Sequential(
            nn.Linear(2, position_dim),
            nn.GELU(),
            nn.Linear(position_dim, embedding_dim),
        )

        # Combine sentence text with its position representation.
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_neurons),
        )

        # Trainable PB-ANN thresholds.
        self.thresholds = nn.Parameter(torch.zeros(num_neurons))

        # Trainable recurrent weights.
        self.recurrent_weight = nn.Parameter(
            torch.empty(num_neurons, num_neurons)
        )
        nn.init.xavier_uniform_(self.recurrent_weight, gain=0.4)

        # Fixed sparse connectivity mask.
        mask = (
            torch.rand(num_neurons, num_neurons) >= sparsity
        ).float()
        self.register_buffer("recurrent_mask", mask)

        self.rec_norm = nn.LayerNorm(num_neurons)
        self.rec_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(num_neurons)
        self.sentence_head = nn.Linear(num_neurons, 1)

    def forward(
        self,
        sentence_embeddings,
        sentence_mask=None,
        sentence_positions=None,
    ):
        batch_size, num_sentences, embedding_dim = (
            sentence_embeddings.shape
        )
        device = sentence_embeddings.device

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {embedding_dim}."
            )

        if sentence_mask is None:
            sentence_mask = torch.ones(
                batch_size,
                num_sentences,
                dtype=torch.bool,
                device=device,
            )
        else:
            sentence_mask = sentence_mask.to(
                device=device, dtype=torch.bool
            )

        # Default to evenly spaced normalized sentence positions.
        if sentence_positions is None:
            base_positions = torch.linspace(
                0.0,
                1.0,
                num_sentences,
                device=device,
                dtype=sentence_embeddings.dtype,
            )
            sentence_positions = base_positions.unsqueeze(0).expand(
                batch_size, -1
            )
        else:
            sentence_positions = sentence_positions.to(
                device=device, dtype=sentence_embeddings.dtype
            )

        if sentence_positions.shape != (batch_size, num_sentences):
            raise ValueError(
                "sentence_positions must have shape "
                "[batch, num_sentences]."
            )

        # Encode both position and distance from the article centre.
        position_features = torch.stack(
            [
                sentence_positions,
                torch.abs(sentence_positions - 0.5) * 2.0,
            ],
            dim=-1,
        )

        position_embeddings = self.position_encoder(position_features)

        # Add position information to the text representation.
        combined_embeddings = (
            sentence_embeddings + position_embeddings
        )

        # Convert combined text-position representation into stimulation.
        stimulation = self.encoder(combined_embeddings)

        # Initial neural state.
        potentials = torch.zeros_like(stimulation)

        effective_weights = (
            self.recurrent_weight * self.recurrent_mask
        )

        # PB-ANN recurrent dynamics.
        for _ in range(self.steps):
            recurrent = F.linear(potentials, effective_weights)
            recurrent = self.rec_norm(recurrent)

            potentials = (
                (1.0 - self.leak) * potentials
                + recurrent
                + stimulation
            )

            firing = F.relu(potentials - self.thresholds)

            firing_mean = firing.mean(dim=-1, keepdim=True)
            potentials = (
                firing
                - self.inhibition_strength * firing_mean
            )

            potentials = self.rec_dropout(potentials)

        states = self.out_norm(potentials)
        logits = self.sentence_head(states).squeeze(-1)

        # Keep padding sentences out of selection and loss.
        logits = logits.masked_fill(~sentence_mask, -1e4)

        return logits
