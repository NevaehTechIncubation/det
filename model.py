import torch
import torch.nn as nn

from backbone import YOLOBackbone
from detection_head import YOLODetectionHead


class YOLO(nn.Module):
    def __init__(self, num_classes=20, B=3):
        """
        num_classes: Total number of object classes (e.g., 20 for Pascal VOC).
        B: Number of bounding boxes predicted per grid cell.
        """
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.detection_head = YOLODetectionHead(
            in_channels=1024, num_classes=num_classes, B=B
        )

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, 3, H, W] (e.g., 416x416 images).
        Returns:
            Predictions of shape [batch_size, S, S, B, (5 + num_classes)]
            where S is the spatial dimension of the final grid (e.g., 13 for a 416x416 image).
        """
        features = self.backbone(x)
        predictions = self.detection_head(features)
        return predictions


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 640, 640)

    model = YOLO(num_classes=16, B=3)

    output = model(dummy_input)
    print("Integrated YOLO network output shape:", output.shape)
