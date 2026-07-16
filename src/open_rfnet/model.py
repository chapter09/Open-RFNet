"""Multi-domain feature extractor described in the Open-RFNet paper."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .constants import KNOWN_LABELS


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class SmallResNet18(nn.Module):
    """ResNet-18 with a narrow base width, matching the paper's ~0.7M baseline."""

    def __init__(self, base_width: int = 16):
        super().__init__()
        self.output_dim = base_width * 8
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.in_channels = base_width
        self.layer1 = self._layer(base_width, blocks=2, stride=1)
        self.layer2 = self._layer(base_width * 2, blocks=2, stride=2)
        self.layer3 = self._layer(base_width * 4, blocks=2, stride=2)
        self.layer4 = self._layer(base_width * 8, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride=stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


def sinusoidal_encoding(length: int, dimension: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class PositionEncoder(nn.Module):
    """MNL -> position encoding -> TransformerEncoder -> flattened MNL."""

    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        model_dim: int,
        heads: int,
        layers: int,
        feedforward_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.pre = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.post = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length * model_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )
        self.register_buffer("position", sinusoidal_encoding(sequence_length, model_dim), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, received {x.shape[1]}")
        x = self.pre(x)
        x = x + self.position.to(dtype=x.dtype, device=x.device).unsqueeze(0)
        return self.post(self.transformer(x))


class MultiDomainEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        size = int(config["input_size"])
        base_width = int(config["resnet_base_width"])
        model_dim = int(config["transformer_dim"])
        position_dim = int(config["position_dim"])
        dropout = float(config.get("dropout", 0.1))
        branch_args = dict(
            sequence_length=size,
            input_dim=size,
            model_dim=model_dim,
            heads=int(config["transformer_heads"]),
            layers=int(config["transformer_layers"]),
            feedforward_dim=int(config["transformer_ff_dim"]),
            output_dim=position_dim,
            dropout=dropout,
        )
        self.texture = SmallResNet18(base_width)
        self.time_position = PositionEncoder(**branch_args)
        self.frequency_position = PositionEncoder(**branch_args)
        fused_dim = int(config["fused_dim"])
        self.output_dim = fused_dim
        self.fusion = nn.Sequential(
            nn.Linear(self.texture.output_dim + 2 * position_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fused_dim),
        )

    def forward(self, image: Tensor) -> Tensor:
        matrix = image.squeeze(1)
        texture = self.texture(image)
        time_feature = self.time_position(matrix)
        frequency_feature = self.frequency_position(matrix.transpose(1, 2))
        return self.fusion(torch.cat((texture, time_feature, frequency_feature), dim=1))


class OpenRFNet(nn.Module):
    def __init__(self, config: dict[str, Any], num_classes: int = len(KNOWN_LABELS)):
        super().__init__()
        self.encoder = MultiDomainEncoder(config)
        fused_dim = self.encoder.output_dim
        self.projection = nn.Sequential(
            nn.Linear(fused_dim, int(config["projection_hidden_dim"])),
            nn.GELU(),
            nn.Linear(int(config["projection_hidden_dim"]), int(config["projection_dim"])),
        )
        self.classifier = nn.Linear(fused_dim, num_classes)

    def encode(self, image: Tensor) -> Tensor:
        return self.encoder(image)

    def project(self, feature: Tensor) -> Tensor:
        return nn.functional.normalize(self.projection(feature), dim=1)

    def forward(self, image: Tensor, return_projection: bool = False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        feature = self.encode(image)
        logits = self.classifier(feature)
        if return_projection:
            return logits, self.project(feature), feature
        return logits

    def expand_for_unknown(self) -> None:
        old = self.classifier
        new = nn.Linear(old.in_features, old.out_features + 1, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:-1].copy_(old.weight)
            new.bias[:-1].copy_(old.bias)
        self.classifier = new


def build_model(config: dict[str, Any], num_classes: int = len(KNOWN_LABELS)) -> OpenRFNet:
    return OpenRFNet(config, num_classes=num_classes)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

