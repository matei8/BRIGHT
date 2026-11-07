import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        channels = [32, 64, 128, 256, 512]

        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        previous = in_channels

        for channel in channels:
            self.blocks.append(
                ConvBlock(previous, channel)
            )
            previous = channel

    def forward(self, x):
        features = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            features.append(x)

            if i < len(self.blocks) - 1:
                x = self.pool(x)

        return features

