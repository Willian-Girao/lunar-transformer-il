import torch.nn as nn
import torch
from src.models.decoder_only_transformer.LanderEmbedding import LanderEmbedding
from src.models.decoder_only_transformer.TransformerLayer import TransformerLayer
from src.models.noisy_layers.NoisyLinear import NoisyLinear

class DecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)

        self.embedding_layer = LanderEmbedding(config)
        self.decoder_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.depth)]
        )
        self.lin_readout = NoisyLinear(size_in=config.embedding_dim, size_out=config.nb_actions, config=config)

    def forward(self, states_seq, actions_seq, padding_mask=None, noise_seed=None):
        # env. state space -> model state space.
        x = self.embedding_layer(states_seq, actions_seq)

        # feed through decoder layers.
        for dl in self.decoder_layers:
            x = dl(x, padding_mask, noise_seed)

        # Feed through a linear readout. Since we have a masked attention
        # that prevents peeking into the future, the last element embedding
        # has contextual information about all elements to the left in the
        # sequence.
        return self.lin_readout(x[:, -1, :]) # embedding of last element on the seq.