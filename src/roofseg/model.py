"""EfficientNet-B0 U-Net construction and initialization checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
TRUSTED_SHA256_PREFIX = "7f5810bc"


class DecoderBlock(nn.Module):
    """Upsample, concatenate an optional encoder skip, and refine twice."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        channels = input_channels + skip_channels
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="nearest")
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.convolutions(x)


class EfficientNetB0UNet(nn.Module):
    """U-Net decoder over all EfficientNet-B0 feature stages.

    The model maps float32 ``N×3×512×512`` normalized RGB images to float32
    ``N×1×512×512`` logits. Encoder skips are taken at 256, 128, 64 and 32
    pixels; the 1,280-channel encoder projection is the 16-pixel bottleneck.
    """

    def __init__(self):
        super().__init__()
        self.encoder = efficientnet_b0(weights=None).features
        self.up1 = DecoderBlock(1280, 112, 256)
        self.up2 = DecoderBlock(256, 40, 128)
        self.up3 = DecoderBlock(128, 24, 64)
        self.up4 = DecoderBlock(64, 16, 32)
        self.up5 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x0 = self.encoder[0](image)
        x1 = self.encoder[1](x0)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[3](x2)
        x4 = self.encoder[4](x3)
        x5 = self.encoder[5](x4)
        x6 = self.encoder[6](x5)
        x7 = self.encoder[7](x6)
        bottleneck = self.encoder[8](x7)
        decoded = self.up1(bottleneck, x5)
        decoded = self.up2(decoded, x3)
        decoded = self.up3(decoded, x2)
        decoded = self.up4(decoded, x1)
        decoded = self.up5(decoded)
        return self.head(decoded)


def state_digest(module: nn.Module) -> str:
    """Hash ordered parameter and buffer names and bytes on CPU."""
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pretrained_weight_record() -> dict:
    """Return the trusted weight identity and the locally verified full hash."""
    state = WEIGHTS.get_state_dict(progress=True, check_hash=True)
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not digest.startswith(TRUSTED_SHA256_PREFIX):
        raise ValueError("Cached EfficientNet-B0 weights fail the trusted torchvision hash prefix.")
    return {
        "enum": "torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1",
        "url": WEIGHTS.url,
        "trusted_sha256_prefix": TRUSTED_SHA256_PREFIX,
        "actual_sha256": digest,
        "cache_file": checkpoint.name,
        "categories": len(WEIGHTS.meta["categories"]),
        "state_tensors": len(state),
    }


def build_model(initialization: str, model_seed: int) -> tuple[EfficientNetB0UNet, dict]:
    """Build a paired random or ImageNet-encoder model from the same fresh seed."""
    if initialization not in {"random", "imagenet"}:
        raise ValueError("Initialization must be 'random' or 'imagenet'.")
    torch.manual_seed(model_seed)
    model = EfficientNetB0UNet()
    decoder_before = state_digest(model.up1) + state_digest(model.up2) + state_digest(model.up3) + state_digest(model.up4) + state_digest(model.up5) + state_digest(model.head)
    random_encoder_digest = state_digest(model.encoder)
    weight_record = None
    if initialization == "imagenet":
        weight_record = pretrained_weight_record()
        full_state = WEIGHTS.get_state_dict(progress=False, check_hash=True)
        encoder_state = {name.removeprefix("features."): value for name, value in full_state.items()
                         if name.startswith("features.")}
        model.encoder.load_state_dict(encoder_state, strict=True)
    decoder_after = state_digest(model.up1) + state_digest(model.up2) + state_digest(model.up3) + state_digest(model.up4) + state_digest(model.up5) + state_digest(model.head)
    if decoder_before != decoder_after or not all(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Pretraining changed the decoder or froze parameters.")
    return model, {
        "initialization": initialization,
        "model_seed": model_seed,
        "random_encoder_sha256": random_encoder_digest,
        "encoder_sha256": state_digest(model.encoder),
        "decoder_sha256": hashlib.sha256(decoder_after.encode()).hexdigest(),
        "pretrained_weights": weight_record,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
