import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AudioResNet18(nn.Module):
    """
    Lightweight ResNet18 adapted for Mel-spectrograms.
    """
    def __init__(self, pretrained=True):
        super(AudioResNet18, self).__init__()
        
        # Load standard ResNet18
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)
            
        # We will feed 1-channel spectrograms, but ResNet expects 3 channels.
        # We can either modify the first conv layer or just repeat the channels in forward().
        # Repeating is simpler and works well with pre-trained weights.
        
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        """
        x: (batch_size, 1, n_mels, time)
        Returns: (batch_size, 1) logits
        """
        # Repeat the 1 channel to 3 channels (R, G, B) to match ImageNet expectations
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        logits = self.resnet(x)
        return logits
