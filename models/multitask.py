"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
 
from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .segmentation import _dec_block
 
 
# ── Inline head definitions ───────────────────────────────────────────────
# We re-declare the heads here rather than importing the full task models,
# because the multitask model owns a SINGLE shared encoder — importing
# VGG11Classifier would bring a second, separate encoder with it.
 
class _ClassificationHead(nn.Module):
    """FC head identical to VGG11Classifier's classifier Sequential."""
    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
 
    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(bottleneck)            # [B, 512, 7, 7]
        x = x.view(x.size(0), -1)               # [B, 25088]
        return self.fc(x)                        # [B, num_classes]
 
 
class _LocalizationHead(nn.Module):
    """Regression head identical to VGG11Localizer's regressor Sequential."""
    IMAGE_SIZE = 224
 
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )
 
    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(bottleneck)            # [B, 512, 7, 7]
        x = x.view(x.size(0), -1)               # [B, 25088]
        raw = self.regressor(x)                  # [B, 4]
        return torch.sigmoid(raw) * self.IMAGE_SIZE  # [B, 4] in pixel space
 
 
class _SegmentationHead(nn.Module):
    """U-Net decoder identical to VGG11UNet (decoder + head only, no encoder)."""
    def __init__(self, num_classes: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.bottleneck_drop = CustomDropout(p=dropout_p)
 
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)
 
        self.up4  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = _dec_block(512 + 512, 256)
 
        self.up3  = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = _dec_block(256 + 256, 128)
 
        self.up2  = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = _dec_block(128 + 128,  64)
 
        self.up1  = nn.ConvTranspose2d(64,   64, kernel_size=2, stride=2)
        self.dec1 = _dec_block( 64 +  64,  64)
 
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
 
    def forward(self, bottleneck: torch.Tensor, skips: dict) -> torch.Tensor:
        d = self.bottleneck_drop(bottleneck)
 
        d = self.up5(d);  d = torch.cat([d, skips["block5"]], dim=1);  d = self.dec5(d)
        d = self.up4(d);  d = torch.cat([d, skips["block4"]], dim=1);  d = self.dec4(d)
        d = self.up3(d);  d = torch.cat([d, skips["block3"]], dim=1);  d = self.dec3(d)
        d = self.up2(d);  d = torch.cat([d, skips["block2"]], dim=1);  d = self.dec2(d)
        d = self.up1(d);  d = torch.cat([d, skips["block1"]], dim=1);  d = self.dec1(d)
 
        return self.head(d)                      # [B, num_classes, 224, 224]
 
 

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        import gdown
#         https://drive.google.com/file/d/1aD-PFsrIDWMqFMd8QOBzuCEhQ1w4HN-9/view?usp=sharing

# https://drive.google.com/file/d/1z5HxRX3Y4Ik3Qb6ryqj-9Z6RIviiKLol/view?usp=sharing
# https://drive.google.com/file/d/1B4UKyuN5i-KOg8o3tB4BZCtwhuNCs5pe/view?usp=sharing
        gdown.download(id="1aD-PFsrIDWMqFMd8QOBzuCEhQ1w4HN-9", output=classifier_path, quiet=False)
        gdown.download(id="1z5HxRX3Y4Ik3Qb6ryqj-9Z6RIviiKLol", output=localizer_path, quiet=False)
        gdown.download(id="1B4UKyuN5i-KOg8o3tB4BZCtwhuNCs5pe", output=unet_path, quiet=False)

        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.cls_head = _ClassificationHead(num_classes=num_breeds)
        self.loc_head = _LocalizationHead()
        self.seg_head = _SegmentationHead(num_classes=seg_classes)
 
        # ── Load weights from individual task checkpoints ─────────────────
        self._load_all_weights(classifier_path, localizer_path, unet_path)

    def _load_all_weights(
        self,
        classifier_path: str,
        localizer_path: str,
        unet_path: str,
    ) -> None:
        """Load encoder + each head from saved task checkpoints.
 
        Strategy:
            - Encoder weights: taken from classifier.pth (most general,
              trained on the full classification objective).
            - Classification head: from classifier.pth  (keys: "classifier.*")
            - Localization head:   from localizer.pth   (keys: "regressor.*")
            - Segmentation decoder: from unet.pth        (keys: "up*/dec*/head*")
        """
        device = "cpu"  # load to CPU first; move to GPU via .to(device) externally
 
        def _load(path):
            ckpt = torch.load(path, map_location=device)
            return ckpt["state_dict"] if "state_dict" in ckpt else ckpt
 
        # ── Classifier checkpoint -> encoder + cls_head ───────────────────
        cls_state = _load(classifier_path)
 
        # Encoder
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in cls_state.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc_state, strict=True)
        print(f"  Encoder loaded from {classifier_path}")
 
        # Classification head: keys are "classifier.0.weight", etc.
        # Our _ClassificationHead stores them under "fc.*"
        cls_head_state = {k.replace("classifier.", "fc."): v
                          for k, v in cls_state.items() if k.startswith("classifier.")}
        # avg_pool has no weights, so only fc keys matter
        missing, unexpected = self.cls_head.load_state_dict(cls_head_state, strict=False)
        print(f"  Classification head loaded | missing={missing} unexpected={unexpected}")
 
        # ── Localizer checkpoint -> loc_head ──────────────────────────────
        loc_state = _load(localizer_path)
        # Keys: "regressor.*", "avg_pool.*" — map directly
        loc_head_state = {k: v for k, v in loc_state.items()
                          if k.startswith("regressor.") or k.startswith("avg_pool.")}
        missing, unexpected = self.loc_head.load_state_dict(loc_head_state, strict=False)
        print(f"  Localization head loaded   | missing={missing} unexpected={unexpected}")
 
        # ── U-Net checkpoint -> seg_head ──────────────────────────────────
        unet_state = _load(unet_path)
        # U-Net checkpoint has keys like "encoder.*", "bottleneck_drop.*",
        # "up5.*", "dec5.*", ... , "head.*".
        # We only need the decoder keys (not the encoder — we already loaded it).
        seg_head_state = {k: v for k, v in unet_state.items()
                          if not k.startswith("encoder.")}
        missing, unexpected = self.seg_head.load_state_dict(seg_head_state, strict=False)
        print(f"  Segmentation head loaded   | missing={missing} unexpected={unexpected}")
    
    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.
        bottleneck, skips = self.encoder(x, return_features=True) # bottleneck: [B, 512, 7, 7]
        # skips: {"block1": [B,64,224,224], ..., "block5": [B,512,14,14]}

        cls_logits = self.cls_head(bottleneck)           # [B, num_breeds]
        bbox       = self.loc_head(bottleneck)           # [B, 4]
        seg_logits = self.seg_head(bottleneck, skips)    # [B, seg_classes, H, W]

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg_logits,
        }
