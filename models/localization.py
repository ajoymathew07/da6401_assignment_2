"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    IMAGE_SIZE = 224
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()

        # VGG11Encoder (shared)
        # Flatten [B, 25088]
        # FC(4096) -> BN -> ReLu -> Dropout
        # FC(1024) -> BN -> ReLu -> Dropout
        # FC(4) -> Sigmoid * IMAGE_SIZE

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(1024, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        # TODO: Implement forward pass.
        features = self.encoder(x, return_features=False)
        pooled = self.avgpool(features)
        flat = pooled.view(pooled.size(0), -1)
        raw = self.regressor(flat)

        bbox = torch.sigmoid(raw) * self.IMAGE_SIZE
        return bbox

    def load_encoder_weights(self, classifier_checkpoint_path: str, device : str = "cpu") -> None:
        """Load encoder weights from a VGG11 classifier checkpoint.

        Args:
            classifier_checkpoint_path: Path to the VGG11 classifier checkpoint.
            device: Device to load the checkpoint on (default: "cpu").
        """
        ckpt = torch.load(classifier_checkpoint_path, map_location=device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=True)

        if missing:
            print(f"Warning: Missing keys in encoder state dict: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in encoder state dict: {unexpected}")

        print(f"Successfully loaded encoder weights from {classifier_checkpoint_path}")
