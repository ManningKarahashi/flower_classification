import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

def get_dataloaders(data_directory="./flower_data", batch_size = 32, train_split = 0.9):
    # Define transformations for your images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),                                              # Resize images to 128x128
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
        transforms.ToTensor(),                                                      # Convert images to PyTorch tensors
        transforms.Normalize([0.5, 0.5, 0.5], 
                             [0.5, 0.5, 0.5]) 
        ])                                                                          # Normalize to [-1, 1] range for 3 RGB channels

    dataset = datasets.ImageFolder(root = data_directory, transform = transform)

    # Split dataset into training and testing sets
    train_size = int(train_split * len(dataset))                                    # 90% for training
    test_size = len(dataset) - train_size                                           # 10% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])    # Splits the dataset randomly into training and testing sets

    targets = [dataset.targets[i] for i in train_dataset.indices]                   # Get the target probabilities for training set only
    counts = Counter(targets)                                                       # Count occurrences of each class in training
    class_counts = torch.tensor([
        counts[i] for i in range(len(dataset.classes))], dtype = torch.float)       # Create tensor of the counts for each class
    class_weights = 1.0 / class_counts                                              # Inverse frequency weighting

    train_targets = torch.tensor(targets)
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )

    train_loader = DataLoader(                                                      # Centralize loaders in dataset.py because data pipline is static
                                                                                    # If data pinpine is configurable per run, define loaders seperately in test.py/train.py
        train_dataset,
        batch_size,
        sampler = sampler,
        # num_workers = 12,                                                         # Shift load to CPU threads for faster performance
        # pin_memory = True                                                         # Speeds up transfer to specified device
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size,
        shuffle = True,
        # num_workers = 12,
        # pin_memory = True
        )

    return train_loader, test_loader, dataset.classes