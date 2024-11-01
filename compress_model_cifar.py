import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from weight_generator.config import Config
from weight_generator.network_compressor import NetworkCompressor
from full_train_cifar import FullModel

if __name__ == "__main__":
    # Configuration setup
    config = Config()
    config.validate_config()

    # Load trained full model with error handling
    if not os.path.exists('./full_model.pth'):
        raise FileNotFoundError("Model weights not found at './full_model.pth'")
    full_model = FullModel(config)
    full_model.load_state_dict(torch.load('./full_model.pth', map_location=torch.device('cpu')))

    # Compress the model
    compressor = NetworkCompressor(config, full_model, compression_factor=0.5)  # 50% compression
    compressed_model = compressor.compress_model()

    # Device setup (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compressed_model = compressed_model.to(device)

    # Prepare data loader for training the compressed model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Train the compressed model
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    compressed_model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = compressed_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.6f}')

    # Save the compressed model
    torch.save(compressed_model.state_dict(), './compressed_model.pth')
