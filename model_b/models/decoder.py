import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import ConvBlock

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Safety in case spatial dimensions differ
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FusionDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Each encoder has channels [32, 64, 128, 256, 512]
        # Concatenated features: [64, 128, 256, 512, 1024]

        self.decoder4 = DecoderBlock(1024, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 128, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)

        self.classifier = nn.Conv2d(
            32, num_classes, kernel_size=1
        )

    def forward(self, fused_features):

        x = fused_features[4]

        x = self.decoder4(x, fused_features[3])
        x = self.decoder3(x, fused_features[2])
        x = self.decoder2(x, fused_features[1])
        x = self.decoder1(x, fused_features[0])

        return self.classifier(x)

