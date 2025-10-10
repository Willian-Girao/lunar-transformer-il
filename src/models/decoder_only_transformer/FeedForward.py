import torch.nn as nn
import torch
from src.models.noisy_layers.NoisyLinear import NoisyLinear

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)
        
        self.linear_1 = NoisyLinear(size_in=config.embedding_dim, size_out=config.intermediate_dim, config=config)
        self.linear_2 = NoisyLinear(size_in=config.intermediate_dim, size_out=config.embedding_dim, config=config)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, noise_seed=None):
        x = self.linear_1(x, noise_seed)
        x = self.gelu(x)
        x = self.linear_2(x, noise_seed)
        x = self.dropout(x)
        return x