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
        x = self.predictions(x)

        batch_size, _, S_h, S_w = x.shape

        x = x.view(batch_size, self.B, (5 + self.num_classes), S_h, S_w)

        # Permute to get the final output shape:
        # (batch_size, S_h, S_w, B, (5+num_classes))
        x = x.permute(0, 3, 4, 1, 2).contiguous()

        x_xy = torch.sigmoid(x[..., 0:2])  # Center x, y
        x_wh = torch.exp(x[..., 2:4])
        x_conf = torch.sigmoid(x[..., 4:5])
        x_class = x[..., 5:]

        x = torch.cat((x_xy, x_wh, x_conf, x_class), dim=-1)

        return x


if __name__ == "__main__":
    dummy_features = torch.randn(1, 1024, 20, 20)

    detection_head = YOLODetectionHead(in_channels=1024, num_classes=20, B=3)

    predictions = detection_head(dummy_features)

    print("Predictions shape:", predictions.shape)
