import torch
import torch.nn as nn
import torch.nn.functional as F

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