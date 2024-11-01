import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from weight_generator.config import Config


# Define the full model
class FullModel(nn.Module):
    def __init__(self, config):
        super(FullModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, config.channels_per_layer[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.channels_per_layer[1], config.channels_per_layer[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.channels_per_layer[2], config.channels_per_layer[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(config.channels_per_layer[3] * 16 * 16, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten before feeding into classifier
        x = self.classifier(x)
        return x


# Training loop
def train(model, device, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(
                    f'Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.6f}')


if __name__ == "__main__":
    # Configuration
    config = Config()
    config.validate_config()

    # Device setup (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Model, criterion, optimizer setup
    model = FullModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, device, train_loader, optimizer, criterion, config.num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), './full_model.pth')