"""Two-stream U-Net with feature-level fusion (variant B).

Ported from Nicoleta's ``model_b/models/{encoder,decoder,two_stream_fusion}.py`` (branch
``nicoleta``); the three files are merged into one module. The only change: all channel widths
are 1.5x hers ([32..512] -> [48..768]), so the model has about the same number of parameters
as the variant-A U-Net (~31M vs. 13.8M originally) and A vs. B compares fusion, not capacity.

* Two separate encoders: optical (3 channels, pre-event) and SAR (1 channel, post-event),
  each 5 levels wide [48, 96, 192, 384, 768] with 2x2 max-pool downsampling.
* Fusion: the two feature maps are concatenated at every resolution -> [96, ..., 1536].
* U-Net decoder with transpose-conv upsampling and skip connections, 1x1 classifier.

Input H and W must be divisible by 16 (512, 640 and 1024 all qualify).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = [48, 96, 192, 384, 768]
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        previous = in_channels
        for channel in channels:
            self.blocks.append(ConvBlock(previous, channel))
            previous = channel

    def forward(self, x):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)
            if i < len(self.blocks) - 1:
                x = self.pool(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:  # safety in case spatial dimensions differ
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FusionDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # each encoder: [48, 96, 192, 384, 768]; concatenated: [96, 192, 384, 768, 1536]
        self.decoder4 = DecoderBlock(1536, 768, 384)
        self.decoder3 = DecoderBlock(384, 384, 192)
        self.decoder2 = DecoderBlock(192, 192, 96)
        self.decoder1 = DecoderBlock(96, 96, 48)
        self.classifier = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, fused_features):
        x = fused_features[4]
        x = self.decoder4(x, fused_features[3])
        x = self.decoder3(x, fused_features[2])
        x = self.decoder2(x, fused_features[1])
        x = self.decoder1(x, fused_features[0])
        return self.classifier(x)


class TwoStreamFusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.optical_encoder = Encoder(in_channels=3)
        self.sar_encoder = Encoder(in_channels=1)
        self.decoder = FusionDecoder(num_classes=num_classes)

    def forward(self, optical, sar):
        optical_features = self.optical_encoder(optical)
        sar_features = self.sar_encoder(sar)
        fused_features = [torch.cat([o, s], dim=1) for o, s in zip(optical_features, sar_features)]
        return self.decoder(fused_features)
