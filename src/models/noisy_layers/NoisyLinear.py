# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class NoisyLinear(nn.Module):
    """
    """
    def __init__(self, size_in, size_out, config):
        super().__init__()

        self.size_in, self.size_out = size_in, size_out
        self.noise_scale = config.noise_scale

        # Use a local generator for reproducible weight initialization
        generator = torch.Generator()
        generator.manual_seed(config.seed)

        # Initialize weights with kaiming_uniform using the generator
        weights = torch.empty(size_out, size_in)
        nn.init.kaiming_uniform_(weights, mode='fan_in', nonlinearity='linear')
        self.weights = nn.Parameter(weights)

        # Bias
        bias = torch.zeros(size_out)
        self.bias = nn.Parameter(bias)

        # Noise function
        if getattr(config, 'noise_type', 'uniform') == 'uniform':
            self.noise_fn = torch.rand
        elif getattr(config, 'noise_type', 'uniform') == 'normal':
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
        if self.training and self.noise_fn is not None and noise_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(noise_seed)
            unit_noise = self.noise_fn(x.size(0), self.size_out, generator=gen) * self.noise_scale

            # Adjust for mean/std if normal
            if self.noise_fn == torch.randn:
                unit_noise = unit_noise * self.noise_std + self.noise_mean

            w_times_x = torch.add(torch.matmul(x, self.weights.t()), unit_noise)
        else:
            w_times_x = torch.matmul(x, self.weights.t())

        return torch.add(w_times_x, self.bias)