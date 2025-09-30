import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter


class Flower(nn.Module):
    def __init__(self):
        super(Flower, self).__init__()
        self.convolutionlayer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)         # 3 RGB channels instead of 1 for grayscale, 32 filters/outputs, each filter looks at 3x3 patches at a time, size of input is preserved due to padding
        self.batchnormalization1 = nn.BatchNorm2d(32)                               # Normalizes activations for the 32 channels from conv1, stabilizes training
        self.pool = nn.MaxPool2d(2, 2)                                              # Takes the maximum value of each non-overlapping 2x2 patch, halves height and width of image to reduce computation and speed up training

        # Second conv block
        self.convolutionlayer2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)        # 32 input channels, 64 more complex features
        self.batchnormalization2 = nn.BatchNorm2d(64)

        # Third conv block
        self.convolutionlayer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)       # 64 input channels, 128 more complex features
        self.batchnormalization3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.gap = nn.AdaptiveAvgPool2d((1,1))                                      # Global average pooling
        self.fullyconnected1 = nn.Linear(128, 128)                                  # Connects all 128 neurons to 128 neurons in the next layer
        self.fullyconnected2 = nn.Linear(128, 15)                                   # Connects all 128 neurons to 15 output classes (daisy, dandelion, rose, sunflower, tulip)   

    def forward(self, x):
        x = self.pool(F.relu(self.batchnormalization1(self.convolutionlayer1(x))))  # conv1 + BN + pool
        x = self.pool(F.relu(self.batchnormalization2(self.convolutionlayer2(x))))  # conv2 + BN + pool
        x = self.pool(F.relu(self.batchnormalization3(self.convolutionlayer3(x))))  # conv3 + BN + pool
        x = self.gap(x)
        x = torch.flatten(x, 1)                                                     # Flatten into vectors
        x = F.relu(self.fullyconnected1(x))                                         # ReLU activation layer after first fully connected layer
        return F.log_softmax(self.fullyconnected2(x), dim = 1)                      # log_softmax for NLLLoss

    
# Visualization function to display image and its predicted class probabilities    
def view_classify(image, probabilities):
    probabilities = probabilities.cpu().data.numpy().squeeze()                      # Move probabilities to CPU and convert to numpy array
    fig, (ax1, ax2) = plt.subplots(figsize = (6, 9), ncols = 2)

    img = image.permute(1, 2, 0).numpy()                                            # Convert from tensor image to numpy array
    img = (img * 0.5) + 0.5                                                         # Unnormalize image to see colors

    ax1.imshow(img)
    ax1.axis('off')
    num_classes = len(probabilities)
    ax2.barh(np.arange(num_classes), probabilities)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(num_classes))
    ax2.set_yticklabels(dataset.classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}" if torch.cuda.is_available() else "")

    # Define transformations for your images
    transform = transforms.Compose([
        transforms.Resize((96, 96)),                                                # Resize images to 96x96
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
        transforms.ToTensor(),                                                      # Convert images to PyTorch tensors
        transforms.Normalize([0.5, 0.5, 0.5], 
                             [0.5, 0.5, 0.5]) 
        ])                                                                          # Normalize to [-1, 1] range for 3 RGB channels

    dataset = datasets.ImageFolder(root = "./flower_data", transform = transform)     # load dataset from folder structure

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))                                            # 80% for training
    test_size = len(dataset) - train_size                                           # 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])    # Splits the dataset randomly into training and testing sets
    
    targets = [dataset.targets[i] for i in train_dataset.indices]                   # Get the target probabilities for training set only
    counts = Counter(targets)                                                       # Count occurrences of each class in training
    class_counts = torch.tensor([
        counts[i] for i in range(len(dataset.classes))], dtype = torch.float)         # Create tensor of the counts for each class
    class_weights = 1.0 / class_counts                                              # Inverse frequency weighting
    
    train_targets = torch.tensor(targets)                                           # Convert targets to tensor
    sample_weights = class_weights[train_targets]                                   # Assign weights to each sample based on its class
    sampler = torch.utils.data.WeightedRandomSampler(
        weights = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size = 32,
        sampler = sampler,
        num_workers = 12,                                                           # Shift load to CPU threads for faster performance
        pin_memory = True                                                           # Speeds up transfer to GPU
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size = 32,
        shuffle = True,
        num_workers = 12,
        pin_memory = True
        )

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = Flower().to(device)                                        # Move model to GPU
    cost_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)                    # Adam optimizer, acts in place of gradeitn descent, faster convergence  
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = 15, gamma = 0.5)                                     # Reduces learning rate by factor of 0.5 every 15 epochs, helps with convergence due to smaller dataset

    # Train model
    epochs = 40                                                                     # loop over training set 40 times
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)                   # Move data to GPU
            optimizer.zero_grad()                                                   # Zero the gradients before running the backward pass
            output = model(images)                                                  # Forward pass: compute predicted outputs by passing inputs to the model
            cost = cost_function(output, labels)                                    # Calculate the cost of this run
            cost.backward()                                                         # Backward pass: compute gradient of the cost with respect to model parameters
            optimizer.step()                                                        # Perform a single optimization step (parameter update)

        scheduler.step()                                                            # Adjust the learning rate
        print(f"Epoch {epoch+1}/{epochs}, Cost: {cost.item():.4f}")                 # This is all the same as MNIST except for scheduler step

    # Evaluate the model on the test dataset
    model.eval()                                                                    # Set the model to evaluation mode
    all_preds = []                                                                  # Store all predictions generated by model
    all_labels = []                                                                 # Store all true labels of flowers
    with torch.no_grad():                                                           # No need to track gradients for validation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)                   # Move data to GPU
            outputs = model(images)                                                 # Get predictions
            _, predicted = torch.max(outputs, 1)                                    # Get the index of the max log-probability
            all_preds.extend(predicted.cpu().numpy())                               # Move predictions to CPU and convert to numpy array
            all_labels.extend(labels.cpu().numpy())                                 # Move labels to CPU and convert to numpy array

    print(classification_report(
        all_labels, all_preds, target_names = dataset.classes))                     # Print precision, recall, f1-score for each class

    # Visualize some predictions
    images, _ = next(iter(test_loader))                                             # Get a batch of test images
    for i in range(min(20, len(images))):                                           # Display first 20 images
        image = images[i].unsqueeze(0).to(device)                                   # Add batch dimension and move to device
        with torch.no_grad():                                                       # No need to track gradients for validation
            log_probabilities = model(image)
        probabilities = torch.exp(log_probabilities)                                # Convert log probabilities to normal probabilities
        view_classify(images[i].cpu(), probabilities.cpu())                         # Move image and probabilities to CPU for visualization

    print("Model parameters are on:", next(model.parameters()).device)