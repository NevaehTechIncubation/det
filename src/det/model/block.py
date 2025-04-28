from typing import override
import torch
from torch import nn

from det.pipeline import Pipeline


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(ConvBlock, self).__init__()  # pyright: ignore[reportUnknownMemberType]
        self.conv: nn.Module = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn: nn.Module = nn.BatchNorm2d(out_channels)
        self.act: nn.Module = nn.LeakyReLU(0.1, inplace=True)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
        # return Pipeline[torch.Tensor, torch.Tensor](
        #     self.conv,
        #     self.bn,
        #     self.act,
        # )[x]


class CompressorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        pool: bool,
    ) -> None:
        assert depth > 0
        super(CompressorBlock, self).__init__()
        sublayers: list[nn.Module] = []
        for i in range(depth):
            sublayers.append(ConvBlock(in_channels, out_channels, 3, 1, 1))
            sublayers.append(ConvBlock(out_channels, in_channels, 1, 1, 0))
        sublayers.append(ConvBlock(in_channels, out_channels, 3, 1, 1))
        if pool:
            sublayers.append(nn.MaxPool2d(2, 2))
        self.layer: nn.Module = nn.Sequential(*sublayers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
