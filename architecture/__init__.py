"""Model registry: ``build_model(cfg)`` -> (model, forward_fn).

``forward_fn(model, optical, sar)`` hides how each architecture consumes the two
modalities, so the training loop, evaluation and metrics are identical for every
variant. Select with ``train.model`` in the config (``--set train.model=two_stream``).

    unet        variant A: one U-Net on the 4-channel stack (optical RGB + SAR), early fusion
    two_stream  variant B: separate optical / SAR encoders, features concatenated per level
"""

from src.engine import early_fusion_forward, two_stream_forward

from .two_stream import TwoStreamFusionNet
from .unet import UNet

MODELS = ("unet", "two_stream")


def build_model(cfg):
    if cfg.model == "unet":
        return UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes), early_fusion_forward
    if cfg.model == "two_stream":
        return TwoStreamFusionNet(num_classes=cfg.num_classes), two_stream_forward
    raise ValueError(f"train.model must be one of {MODELS}, got {cfg.model!r}")
