import torch
import torch.nn as nn
from model import Flower
from dataset import get_dataloaders
import os


if __name__ == "__main__":
    device = (
        torch.device("cuda") if torch.cuda.is_available()                           # Use CUDA if available
        else torch.device("mps") if torch.backends.mps.is_available()               # Use MPS for Apple Silicon GPU if CUDA is not available
        else torch.device("cpu")                                                    # Fallback to CPU if neither CUDA nor MPS is available
    )
    print(f"Using device: {device}")

    train_loader, test_loader, classes = get_dataloaders()                          # Load test and training datasets

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = Flower().to(device)                                                     # Move model to specified device
    cost_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)                    # Adam optimizer, acts in place of gradeint descent, faster convergence 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = 15, gamma = 0.5                                      # Reduces learning rate by factor of 0.5 every 15 epochs, helps with convergence due to smaller dataset
        )

    epochs = 40
    for epoch in range(epochs):
        model.train()                                                               # Set model to training mode
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)                   # Move data to specified device
            optimizer.zero_grad()                                                   # Zero the gradients before running the backward pass
            output = model(images)                                                  # Forward pass: compute predicted outputs by passing inputs to the model
            cost = cost_function(output, labels)                                    # Calculate the cost of this run
            cost.backward()                                                         # Backward pass: compute gradient of the cost with respect to model parameters
            optimizer.step()                                                        # Perform a single optimization step
        scheduler.step()                                                            # Adjust the learning rate
        print(f"Epoch {epoch+1}/{epochs}, Cost: {cost.item():.4f}")                 # This is all the same as MNIST except for scheduler step


    os.makedirs("model", exist_ok = True)                                             # Ensures folder exists
    save_path = os.path.join("model", "flower_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")