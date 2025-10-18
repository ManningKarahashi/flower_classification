import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from model import Flower
from dataset import get_dataloaders

def view_classify(image, probabilities, classes):
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
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = (
        torch.device("cuda") if torch.cuda.is_available()                           # Use CUDA if available
        else torch.device("mps") if torch.backends.mps.is_available()               # Use MPS for Apple Silicon GPU if CUDA is not available
        else torch.device("cpu")                                                    # Fallback to CPU if neither CUDA nor MPS is available
    )
    print(f"Using device: {device}")

    train_loader, test_loader, classes = get_dataloaders()                          # Load test and training datasets

    model = Flower().to(device)
    model.load_state_dict(torch.load("flower_model.pth", map_location = device))

    # Evaluate the model on the test dataset
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():                                                           # No need to track gradients for validation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)                   # Move data to specified device
            outputs = model(images)                                                 # Get predictions
            _, predicted = torch.max(outputs, 1)                                    # Get the index of the max log-probability
            all_preds.extend(predicted.cpu().numpy())                               # Move predictions to CPU and convert to numpy array
            all_labels.extend(labels.cpu().numpy())                                 # Move labels to CPU and convert to numpy array

    print(classification_report(all_labels, all_preds, target_names = classes))

    # Visualize some predictions
    images, _ = next(iter(test_loader))                                             # Get a batch of test images
    for i in range(min(20, len(images))):                                           # Display first 20 images
        image = images[i].unsqueeze(0).to(device)                                   # Add batch dimension and move to device
        with torch.no_grad():                                                       # No need to track gradients for validation
            log_probabilities = model(image)
        probabilities = torch.exp(log_probabilities)                                # Convert log probabilities to normal probabilities
        view_classify(images[i].cpu(), probabilities.cpu(), classes)                # Move image and probabilities to CPU for visualization