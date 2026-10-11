
import torch
import torch.nn as nn
import torch.nn.functional as F


class PBANNTextScorer(nn.Module):
    """
    PB-ANN v6 adapted for extractive text summarisation.

    Input:
        sentence_embeddings: [batch, num_sentences, embedding_dim]
        sentence_mask:       [batch, num_sentences]
                             True for valid sentences.

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
    ):
        super().__init__()

        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1).")

        self.num_neurons = num_neurons
        self.steps = steps
        self.leak = leak
        self.inhibition_strength = inhibition_strength

        # Text representation -> PB-ANN stimulation
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, num_neurons),
        )

        # PB-ANN v6: trainable thresholds
        self.thresholds = nn.Parameter(torch.zeros(num_neurons))

        # PB-ANN v6: trainable recurrent weights
        self.recurrent_weight = nn.Parameter(
            torch.empty(num_neurons, num_neurons)
        )
        nn.init.xavier_uniform_(self.recurrent_weight, gain=0.4)

        # Fixed sparse connectivity mask
        mask = (
            torch.rand(num_neurons, num_neurons) >= sparsity
        ).float()
        self.register_buffer("recurrent_mask", mask)

        self.rec_norm = nn.LayerNorm(num_neurons)
        self.rec_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(num_neurons)

        # Predict one importance logit per sentence
        self.sentence_head = nn.Linear(num_neurons, 1)

    def forward(self, sentence_embeddings, sentence_mask=None):
        batch_size, num_sentences, _ = sentence_embeddings.shape

        if sentence_mask is None:
            sentence_mask = torch.ones(
                batch_size,
                num_sentences,
                dtype=torch.bool,
                device=sentence_embeddings.device,
            )
        else:
            sentence_mask = sentence_mask.bool()

        # Encode each sentence independently into PB stimulation
        stimulation = self.encoder(sentence_embeddings)

        # Initial PB neuron state
        potentials = torch.zeros_like(stimulation)

        # Recurrent PB-ANN dynamics
        effective_weights = (
            self.recurrent_weight * self.recurrent_mask
        )

        for _ in range(self.steps):
            recurrent = F.linear(potentials, effective_weights)
            recurrent = self.rec_norm(recurrent)

            # Leaky integration + external stimulation
            potentials = (
                (1.0 - self.leak) * potentials
                + recurrent
                + stimulation
            )

            # Threshold firing
            firing = F.relu(potentials - self.thresholds)

            # Lateral inhibition within each sentence's neural state
            firing_mean = firing.mean(dim=-1, keepdim=True)
            potentials = (
                firing
                - self.inhibition_strength * firing_mean
            )

            potentials = self.rec_dropout(potentials)

        # Final neural state -> sentence importance
        states = self.out_norm(potentials)
        logits = self.sentence_head(states).squeeze(-1)

        # Exclude padding sentences from downstream scoring/loss
        logits = logits.masked_fill(~sentence_mask, -1e4)

        return logits
