# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import math

class NoisyLinear(nn.Module):
    """
    """
    def __init__(self, size_in, size_out, config):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out
        self.noise_scale = getattr(config, 'noise_scale', 1.0)

        # Use a local generator for reproducible weight initialization.
        generator = torch.Generator()
        generator.manual_seed(config.seed)

        # Match nn.Linear initialization exactly.
        self.weights = nn.Parameter(torch.empty(size_out, size_in))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5), generator=generator)

        # `nn.Linear` by default initializes bias using the same uniform distribution as weights.
        bound = 1 / math.sqrt(size_in)
        self.bias = nn.Parameter(torch.empty(size_out))
        nn.init.uniform_(self.bias, -bound, bound, generator=generator)

        # Noise function.
        if config.noise_type == 'uniform':
            self.noise_fn = torch.rand
        elif config.noise_type == 'normal':
            self.noise_fn = torch.randn
            self.noise_std = config.noise_std
            self.noise_mean = config.noise_mean
        else:
            self.noise_fn = None

    def forward(self, x, noise_seed=None):
        """
        Args:
         x: Tensor of shape (batch_size, size_in) or (batch_size, seq_len, size_in).
        """
        # Compute weighted sum (pre-activation before bias).
        pre_activation = torch.matmul(x, self.weights.t())

        # Add noise *before* bias (i.e., to the pre-activation).
        if self.training and self.noise_fn is not None and noise_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(noise_seed)

            # Noise per sample × per neuron.
            noise = self.noise_fn(x.size(0), self.size_out, generator=gen) * self.noise_scale

            # Adjust for mean/std if normal.
            if self.noise_fn == torch.randn:
                noise = noise * self.noise_std + self.noise_mean
            pre_activation = pre_activation + noise

        return pre_activation + self.bias