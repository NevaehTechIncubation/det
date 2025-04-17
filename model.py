import torch
import torch.nn as nn

from backbone import YOLOBackbone
from detection_head import YOLODetectionHead

# Assuming YOLOBackbone and YOLODetectionHead have been defined as in previous steps.


class YOLO(nn.Module):
    def __init__(self, num_classes=20, B=3):
        """
        num_classes: Total number of object classes (e.g., 20 for Pascal VOC).
        B: Number of bounding boxes predicted per grid cell.
        """
        super(YOLO, self).__init__()
        # Backbone network that produces high-level features.
        self.backbone = YOLOBackbone()
        # Detection head that converts features to predictions.
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


# Testing the Integrated YOLO Network with Dummy Data
if __name__ == "__main__":
    # Create a dummy input tensor simulating a batch of one 416x416 RGB image.
    dummy_input = torch.randn(1, 3, 416, 416)

    # Instantiate the YOLO model.
    model = YOLO(num_classes=20, B=3)

    # Forward pass through the integrated network.
    output = model(dummy_input)
    print("Integrated YOLO network output shape:", output.shape)
    # Expected output shape: [1, S, S, 3, 25]
    # Typically, S=13 if the backbone downsamples 416x416 to 13x13.
    # 25 comes from 5 (bbox attributes) + 20 (class probabilities).
