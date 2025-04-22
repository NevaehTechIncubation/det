import torch
import torch.nn as nn


def compute_ious(pred_boxes, target_boxes):
    # convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
    pred_x_min = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_x_max = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y_min = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_y_max = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    target_x_min = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_x_max = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y_min = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_y_max = target_boxes[..., 1] + target_boxes[..., 3] / 2

    inter_x_min = torch.maximum(pred_x_min, target_x_min.unsqueeze(3))
    inter_x_max = torch.minimum(pred_x_max, target_x_max.unsqueeze(3))
    inter_y_min = torch.maximum(pred_y_min, target_y_min.unsqueeze(3))
    inter_y_max = torch.minimum(pred_y_max, target_y_max.unsqueeze(3))

    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_area = inter_width * inter_height

    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    target_area = (target_x_max - target_x_min) * (target_y_max - target_y_min)

    union_area = pred_area + target_area.unsqueeze(3) - inter_area

    ious = inter_area / (union_area + 1e-6)
    return iou


class YOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        grid_size,
        num_boxes,
        lambda_coord=5,
        lambda_noobj=0.5,
    ):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        """Weight for localization loss"""
        self.lambda_noobj = lambda_noobj
        """Weight for confidence loss for cells with no object"""
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the YOLO loss.
        Args:
            predictions (torch.Tensor): Model predictions of shape (batch_size, grid_size, grid_size, num_boxes, num_classes + 5).
            targets (torch.Tensor): Ground truth targets of shape (batch_size, grid_size, grid_size,  5 + num_classes).
        Returns:
            torch.Tensor: Computed loss.
        """
        batch_size = predictions.size(0)

        obj_mask = targets[..., 4:5]  # [batch, grid_size, grid_size, 1]

        pred_boxes = predictions[..., :4]
        pred_confidences = predictions[..., 4:5]
        pred_classes = predictions[..., 5:]

        target_boxes = targets[..., :4]
        target_confidences = targets[..., 4:5]
        target_classes = targets[..., 5:]

        ious = compute_ious(pred_boxes, target_boxes)
        iou, best_iou_idx = ious.max(-1)  # [batch_size, grid_size, grid_size]

        best_pred_box = pred_boxes.gather(
            dim=3,
            index=best_iou_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.grid_size, self.grid_size, 1, 4),
        ).squeeze(3)  # [batch_size, grid_size, grid_size, 4]

        best_pred_conf = pred_confidences.gather(
            dim=3, index=best_iou_idx.unsqueeze(-1)
            .unsqueeze(-1)
        ).squeeze(3)  # [batch_size, grid_size, grid_size, 1]  # fmt: skip

        best_pred_class = pred_confidences.gather(
            dim=3,
            index=best_iou_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.grid_size, self.grid_size, 1, self.num_classes),
        ).squeeze(3)  # [batch_size, grid_size, grid_size, 1]
