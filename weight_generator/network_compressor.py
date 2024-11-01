
import torch
import torch.nn as nn
from .weight_sampler import WeightSampler

class NetworkCompressor:
    """
    Compresses a neural network while maintaining performance.
    """
    def __init__(self, config, full_model, compression_factor=0.25):
        self.config = config
        self.full_model = full_model
        self.compression_factor = compression_factor
        self.compressed_model = self._build_compressed_model()
        self.weight_sampler = WeightSampler(config)

    def _build_compressed_model(self):
        """
        Build the compressed model architecture.
        """
        compressed_channels = [int(c * self.compression_factor) for c in self.config.channels_per_layer]
        compressed_model = nn.Sequential(
            *[nn.Conv2d(in_channels, out_channels, 3, padding=1)
              for in_channels, out_channels in zip(compressed_channels[:-1], compressed_channels[1:])]
        )
        return compressed_model

    def compress_model(self):
        """
        Compress the full model and generate weights for the compressed model.
        """
        with torch.no_grad():
            compressed_weights = []
            for param in self.full_model.parameters():
                compressed_weights.append(self.weight_sampler(param.data.flatten()))
            for i, (param, compressed_weight) in enumerate(zip(self.compressed_model.parameters(), compressed_weights)):
                param.data = compressed_weight
        return self.compressed_model

    def train_compressed_model(self, train_loader):
        """
        Train the compressed model to ensure it maintains accuracy after compression.
        """
        optimizer = torch.optim.Adam(self.compressed_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.compressed_model.train()
        
        for epoch in range(self.config.num_epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.compressed_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
