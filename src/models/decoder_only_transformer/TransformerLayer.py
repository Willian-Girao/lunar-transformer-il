from math import sqrt
import torch.nn as nn
import torch
from src.models.decoder_only_transformer.MultiHeadedAttention import MultiHeadedAttention
from src.models.decoder_only_transformer.FeedForward import FeedForward

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadedAttention(config)
        self.feedforward = FeedForward(config)

    def forward(self, x, padding_mask=None, noise_seed=None):
        # Apply normalization and then copy input into Q, K, and V.
        normd_x = self.layer_norm_1(x)
        # Apply attention with a skip connection.
        x = x + self.attention(normd_x, padding_mask, noise_seed)
        # Apply the feed-forward layer with a skip connection.
        x = x + self.feedforward(self.layer_norm_2(x), noise_seed)
        return x