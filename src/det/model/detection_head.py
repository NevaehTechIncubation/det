from typing import override
import torch
from torch import nn


class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, B: int) -> None:
        super(YOLODetectionHead, self).__init__()

        self.num_classes: int = num_classes
        self.B: int = B
        self.conv: nn.Module = nn.Conv2d(in_channels, B * (5 + num_classes), 1, 1, 0)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [BatchSize, InChannels, S, S] -> [BatchSize, S, S, B, (5 + num_classes)]
        """
        x = self.conv(x)
        batch_size, _, S_h, S_w = x.size()
        x = (
            x.view(batch_size, self.B, (5 + self.num_classes), S_h, S_w)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        x_xy = torch.sigmoid(x[..., :2])
        x_wh = torch.exp(x[..., 2:4])
        x_conf = torch.sigmoid(x[..., 4:5])
        x_class = x[..., 5:]

        x = torch.cat((x_xy, x_wh, x_conf, x_class), dim=-1)
        return x
