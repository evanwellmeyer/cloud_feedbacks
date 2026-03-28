"""
CNN for predicting global-mean net cloud feedback from mean-state CRE maps.

Input:  (batch, 2, 144, 192) — [SW_CRE, LW_CRE] on HadGEM 1.25°×1.875° grid
Output: (batch,)             — scalar delta Net CRE (W/m²)

Padding convention
------------------
All convolutions use GeoPad2d instead of standard padding:
  - Circular along longitude (dim=-1): the map wraps east-west with no seam
  - Reflection along latitude (dim=-2): the poles reflect naturally
Conv2d layers are then called with padding=0 so no implicit padding is added.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoPad2d(nn.Module):
    """
    Geophysically-aware 2D padding for global lat-lon maps.

    Apply BEFORE every Conv2d (with padding=0 on the conv itself).

    Longitude (last dim)           : circular — seamless date-line wrap
    Latitude  (second-to-last dim) : reflect  — natural pole boundary
    """
    def __init__(self, pad: int):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular pad east-west edges first so the reflection sees the
        # already-wrapped corners, avoiding corner artefacts.
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="reflect")
        return x


class ConvBlock(nn.Module):
    """GeoPad → Conv2d → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, pad: int = 2):
        super().__init__()
        self.geo_pad = GeoPad2d(pad)
        self.conv    = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=0)
        self.bn      = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(self.geo_pad(x))))


class CloudFeedbackCNN(nn.Module):
    """
    Default architecture:

        ConvBlock(2  → 32,  k=5)   # regional cloud patterns
        ConvBlock(32 → 64,  k=5)   # broader structures
        ConvBlock(64 → 128, k=5)
        ConvBlock(128→ 128, k=3)   # fine-tuning
        AdaptiveAvgPool2d(1)        # global summary (area-equal weighting)
        Linear(128 → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → 1)

    All spatial dimensions are preserved throughout (no strided convs or
    max-pooling) so the global average pool integrates the full resolution
    feature map.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv1 = ConvBlock(2,   32,  kernel=5, pad=2)
        self.conv2 = ConvBlock(32,  64,  kernel=5, pad=2)
        self.conv3 = ConvBlock(64,  128, kernel=5, pad=2)
        self.conv4 = ConvBlock(128, 128, kernel=3, pad=1)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.head  = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x).flatten(1)      # (batch, 128)
        return self.head(x).squeeze(-1)  # (batch,)
