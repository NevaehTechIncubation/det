import torch
from torch import nn


class YOLODetectionHead(nn.Module):
    def __init__(self, in_channels=1024, num_classes=20, B=3):
        """
        in_channels: Number of channels coming from the backbone.
        num_classes: Total number of object classes (e.g., 20 for Pascal VOC).
        B: Number of bounding boxes predicted per grid cell.
        """
        super(YOLODetectionHead, self).__init__()
        self.num_classes = num_classes
        self.B = B

        # The final conv layer outputs B*(5+num_classes) channels.
        # Here, kernel_size=1 keeps the spatial dimensions intact.
        self.predictions = nn.Conv2d(
            in_channels, B * (5 + num_classes), kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        """
        x: Input feature map of shape [batch_size, 1024, S, S].
        Returns:
            Predictions reshaped to [batch_size, S, S, B, (5+num_classes)].
        """
        # Apply the 1x1 convolution to predict bounding boxes, confidence, and class scores.
        x = self.predictions(x)
        # x now has the shape: (batch_size, B*(5+C), S, S)

        # Obtain spatial dimensions.
        batch_size, _, S_h, S_w = x.shape

        # Reshape x to have separate dimensions for bounding boxes and predictions.
        # First, view the tensor as (batch_size, B, (5+num_classes), S_h, S_w).
        x = x.view(batch_size, self.B, (5 + self.num_classes), S_h, S_w)

        # Permute to get the final output shape:
        # (batch_size, S_h, S_w, B, (5+num_classes))
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        return x


# Testing the detection head with dummy feature maps
if __name__ == "__main__":
    # Simulate a feature map from the backbone:
    # Let's assume a batch of 1, 1024 channels, spatial dims S x S (e.g., 20 x 20)
    dummy_features = torch.randn(1, 1024, 20, 20)

    # Instantiate the detection head. For example, for Pascal VOC: num_classes=20, B=3.
    detection_head = YOLODetectionHead(in_channels=1024, num_classes=20, B=3)

    # Forward pass through the detection head
    predictions = detection_head(dummy_features)

    print("Predictions shape:", predictions.shape)
    # Expected output shape: [1, 20, 20, 3, 25]
    # Where 25 = 5 (bbox coordinates + confidence) + 20 (class probabilities)
