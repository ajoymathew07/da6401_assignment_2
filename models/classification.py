"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    # AdaptiveAvgPool -> Flatten -> FC(4096) -> BN -> ReLU -> Dropout
    # FC(4096) -> BN -> ReLU -> Dropout -> FC(num_classes)

    # BN is placed after the FC linear transform and before the ReLU activation.

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """

        super().__init__()

        def maybe_bn(num_features: int) -> nn.Module:
            return nn.BatchNorm1d(num_features) if use_bn else nn.Identity()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7)) #collapses spatial dimensions to 7 *7

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            # nn.BatchNorm1d(4096),
            maybe_bn(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            maybe_bn(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes)
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.

        features = self.encoder(x, return_features=False)

        pooled = self.avg_pool(features)

        flat = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flat)

        return logits