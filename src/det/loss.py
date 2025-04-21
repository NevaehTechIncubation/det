import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, grid_size, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the YOLO loss.
        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, grid_size, grid_size, num_classes + 5).
            targets (torch.Tensor): Ground truth targets of shape (batch_size, num_boxes, 5 + num_classes).
        Returns:
            torch.Tensor: Computed loss.
        """
        # Extract predictions
        pred_boxes = predictions[..., :4]
        pred_confidences = predictions[..., 4:5]
        pred_classes = predictions[..., 5:]
