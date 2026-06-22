import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # Conv layer 1: 1 input channel (grayscale), 16 filters, 3x3 kernel, padding=1
        # Output shape: 16 x 28 x 28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        
        # Conv layer 2: 16 input channels, 32 filters, 3x3 kernel, padding=1
        # Output shape: 32 x 28 x 28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Max pool: 2x2 kernel, stride 2.
        # After conv1 + pool: 16 x 14 x 14
        # After conv2 + pool: 32 x 7 x 7
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer 1
        # Input features: 32 filters * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)
        
        # Output layer: 10 classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Input x shape: [batch_size, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))  # Output: [batch_size, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # Output: [batch_size, 32, 7, 7]
        
        # Flatten
        x = x.view(-1, 32 * 7 * 7)            # Output: [batch_size, 1568]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Returns raw logits. CrossEntropyLoss in PyTorch handles Softmax.

    def get_activations(self, x):
        """
        Helper method to extract activations of conv1 and conv2 for visualization.
        Returns:
            act1: tensor of shape [1, 16, 28, 28]
            act2: tensor of shape [1, 32, 14, 14]
        """
        act1 = F.relu(self.conv1(x))
        # Note: we get act2 by pooling act1 first, then passing through conv2 + relu
        act1_pooled = self.pool(act1)
        act2 = F.relu(self.conv2(act1_pooled))
        return act1, act2
