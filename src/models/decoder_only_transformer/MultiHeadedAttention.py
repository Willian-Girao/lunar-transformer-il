import torch
import torch.nn.functional as F
from math import sqrt
import torch.nn as nn
from src.models.noisy_layers.NoisyLinear import NoisyLinear

# --- Multi-headed Attention ---

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)

        embed_dim = config.embedding_dim
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [MaskedAttentionHead(config, embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = NoisyLinear(size_in=embed_dim, size_out=embed_dim, config=config)

    def forward(self, hidden_states, padding_mask=None, noise_seed=None):
        x = torch.cat([h(hidden_states, padding_mask, noise_seed) for h in self.heads], dim=-1)
        x = self.output_linear(x, noise_seed)
        return x
    
# --- Attention Head ---
    
class MaskedAttentionHead(nn.Module):
    def __init__(self, config, embed_dim, head_dim):
        super().__init__()

        # By default, `torch.arange`, `torch.zeros_like`, and `torch.ones_like` create CPU tensors, so we
        # need to know the device to move them there in the forward pass.
        self.device = config.device

        self.Q = NoisyLinear(embed_dim, head_dim, config=config)
        self.K = NoisyLinear(embed_dim, head_dim, config=config)
        self.V = NoisyLinear(embed_dim, head_dim, config=config)

    def forward(self, hidden_state, padding_mask=None, noise_seed=None):
        """
        hidden_state: [batch, seq_len, embed_dim]
        padding_mask: [batch, seq_len], 1 for real tokens, 0 for padded tokens
        """
        batch_size, seq_len, _ = hidden_state.size()

        # causal mask to prevent attending to future tokens.
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).unsqueeze(0)

        # padding mask to prevent attending to padded tokens.
        if padding_mask is not None:
            # `.unsqueeze(1)` gives us `padding_mask` with shape [batch, 1, seq_len] while
            # `.expand(-1, seq_len, -1)` gives us `padding_mask` with shape [batch, seq_len, seq_len]
            # so that every timestep has access to the mask for all positions.
            mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1) * causal_mask
        else:
            mask = causal_mask

        attn_embeds = scaled_dot_product_attention(
            self.Q(hidden_state, noise_seed), self.K(hidden_state, noise_seed), self.V(hidden_state, noise_seed), mask)
        
        return attn_embeds
    
# --- Similarity Function ---

def scaled_dot_product_attention(query, key, value, mask=None):
        dim_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)