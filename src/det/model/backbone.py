from typing import override
import torch
from torch import nn

from det.model.block import CompressorBlock, ConvBlock
from det.pipeline import Pipeline


class YOLOBackbone(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super(YOLOBackbone, self).__init__()

        self.layer1: nn.Module = nn.Sequential(
            ConvBlock(in_channels, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.layer2: nn.Module = nn.Sequential(
            ConvBlock(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.layer3: nn.Module = CompressorBlock(68, 128, 1)
        self.layer4: nn.Module = CompressorBlock(128, 256, 1)
        self.layer5: nn.Module = CompressorBlock(256, 512, 2)
        self.layer6: nn.Module = CompressorBlock(512, 1024, 2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Pipeline[torch.Tensor, torch.Tensor](
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
        )[x]
