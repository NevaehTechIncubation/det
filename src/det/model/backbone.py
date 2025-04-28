from typing import override
import torch
from torch import nn

from det.model.block import CompressorBlock, ConvBlock
from det.pipeline import Pipeline


# class YOLOBackbone(nn.Module):
#     def __init__(self, in_channels: int = 3) -> None:
#         super(YOLOBackbone, self).__init__()
#
#         self.layer1: nn.Module = nn.Sequential(
#             ConvBlock(in_channels, 32, 3, stride=1, padding=1),
#             nn.MaxPool2d(2, 2),
#         )
#
#         self.layer2: nn.Module = nn.Sequential(
#             ConvBlock(32, 64, 3, stride=1, padding=1),
#             nn.MaxPool2d(2, 2),
#         )
#
#         self.layer3: nn.Module = CompressorBlock(64, 128, 1, pool=True)
#         self.layer4: nn.Module = CompressorBlock(128, 256, 1, pool=True)
#         self.layer5: nn.Module = CompressorBlock(256, 512, 2, pool=True)
#         self.layer6: nn.Module = CompressorBlock(512, 1024, 2, pool=False)
#
#     @override
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         return x
#         # return Pipeline[torch.Tensor, torch.Tensor](
#         #     self.layer1,
#         #     self.layer2,
#         #     self.layer3,
#         #     self.layer4,
#         #     self.layer5,
#         #     self.layer6,
#         # )[x]


class YOLOBackbone(nn.Module):
    def __init__(self, in_channels: int):
        super(YOLOBackbone, self).__init__()

        self.layer1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimensions by 2
        )

        self.layer2 = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer5 = nn.Sequential(
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer6 = nn.Sequential(
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        x = self.layer1(x)  # Output: (batch_size, 32, H/2, W/2)
        x = self.layer2(x)  # Output: (batch_size, 64, H/4, W/4)
        x = self.layer3(x)  # Output: (batch_size, 128, H/8, W/8)
        x = self.layer4(x)  # Output: (batch_size, 256, H/16, W/16)
        x = self.layer5(x)  # Output: (batch_size, 512, H/32, W/32)
        x = self.layer6(x)  # Output: (batch_size, 1024, H/32, W/32)
        return x
