import torch
import torch.nn as nn
from .config import Config

class WeightSampler(nn.Module):
    """
    Implements continuous weight sampling using implicit neural representations.
    """
    def __init__(self, config):
        super(WeightSampler, self).__init__()
        self.config = config
        self.weight_generator = self._build_weight_generator()

    def _build_weight_generator(self):
        """
        Build the weight generator model.
        """
        model = nn.Sequential(
            nn.Linear(self.config.latent_size, self.config.input_size * self.config.output_size),
            nn.Tanh()
        )
        return model

    def forward(self, latent_code):
        """
        Generate weights for the given latent code.
        """
        flat_weights = self.weight_generator(latent_code)
        weights = flat_weights.view(-1, self.config.input_size, self.config.output_size)
        return weights