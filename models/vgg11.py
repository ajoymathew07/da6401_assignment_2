"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        def conv_bn_relu(in_ch, out_ch, kernel = 3, padding = 1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias = False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # 224 * 224 -> 112 * 112        
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 112 * 112 -> 56 * 56
        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 56 * 56 -> 28 * 28
        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 28 * 28 -> 14 * 14
        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 14 * 14 -> 7 * 7 (two conv layers)

        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.

        # To capture features BEFORE pooling, we manually split each block into 
        # conv and pooling parts.

        f1 = self.block1[0](x)  # conv
        p1 = self.block1[1](f1)  # pool

        f2 = self.block2[0](p1)
        p2 = self.block2[1](f2)

        f3 = self.block3[:-1](p2)
        p3 = self.block3[-1](f3)

        f4 = self.block4[:-1](p3)
        p4 = self.block4[-1](f4)

        f5 = self.block5[:-1](p4)
        p5 = self.block5[-1](f5)

        if return_features:
            features = {
                "block1": f1,
                "block2": f2,
                "block3": f3,
                "block4": f4,
                "block5": f5
            }

            return p5, features
        
        return p5
    
VGG11 = VGG11Encoder