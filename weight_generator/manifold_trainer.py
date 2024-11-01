
import torch
import torch.nn as nn
from .weight_sampler import WeightSampler

class WeightManifoldTrainer:
    """
    Trains a weight manifold for generating weights.
    """
    def __init__(self, config):
        self.config = config
        self.model = nn.Sequential(
            *[nn.Linear(config.latent_size, config.latent_size) for _ in range(config.num_layers)]
        )
        self.weight_sampler = WeightSampler(config)

    def train_manifold(self, dataset):
        """
        Train the weight manifold on the provided dataset.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(self.config.num_epochs):
            for data, target in dataset:
                output = self.model(data)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def validate_manifold(self, validation_data):
        """
        Validate the trained manifold.
        """
        self.model.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for data, target in validation_data:
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(validation_data)
