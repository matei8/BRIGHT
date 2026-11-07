import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import FusionDecoder

class TwoStreamFusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Separate encoders
        self.optical_encoder = Encoder(in_channels=3)
        self.sar_encoder = Encoder(in_channels=1)

        # Feature-level fusion
        self.decoder = FusionDecoder(num_classes=num_classes)

    def forward(self, optical, sar):

        # Extract features independently
        optical_features = self.optical_encoder(optical)
        sar_features = self.sar_encoder(sar)

        # Concatenate features at each resolution
        fused_features = []

        for optical_feat, sar_feat in zip(
            optical_features, sar_features
        ):
            fused = torch.cat(
                [optical_feat, sar_feat],
                dim=1
            )
            fused_features.append(fused)

        # Decode fused features
        output = self.decoder(fused_features)

        return output

