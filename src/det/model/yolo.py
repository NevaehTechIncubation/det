from typing import override
import torch
from torch import nn

from det.model.backbone import YOLOBackbone
from det.model.detection_head import YOLODetectionHead
from det.pipeline import Pipeline


class YOLODetector(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_boxes: int = 3) -> None:
        super(YOLODetector, self).__init__()

        self.backbone: nn.Module = YOLOBackbone(in_channels)
        self.detection_head: nn.Module = YOLODetectionHead(1024, num_classes, num_boxes)

    @property
    def device(self):
        return next(self.parameters()).device

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.detection_head(self.backbone(x))
        # return Pipeline[torch.Tensor, torch.Tensor](
        #     self.backbone,
        #     self.detection_head,
        # )[x]
