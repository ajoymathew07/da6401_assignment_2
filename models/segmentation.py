"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.bottleneck_drop = CustomDropout(p= dropout_p)

        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)

        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = _dec_block(512 + 512, 256)

        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = _dec_block(256 + 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = _dec_block(128 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = _dec_block(64 + 64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.

        bottleneck , skips = self.encoder(x, return_features=True)

        d = self.bottleneck_drop(bottleneck)

        d = self.up5(d)
        d = torch.cat([d, skips["block5"]], dim=1)
        d = self.dec5(d)

        d = self.up4(d)
        d = torch.cat([d, skips["block4"]], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, skips["block3"]], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, skips["block2"]], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, skips["block1"]], dim=1)
        d = self.dec1(d)

        return self.head(d)


    def load_encoder_weights(self, classifier_checkpoint_path:str, device:str = "cpu") -> None:
        
        ckpt = torch.load(classifier_checkpoint_path, map_location=device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state.items() if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        print(f"Loaded encoder weights from {classifier_checkpoint_path}")