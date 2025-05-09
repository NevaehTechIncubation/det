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
    return ious


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

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
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

        best_pred_class = pred_classes.gather(
            dim=3,
            index=best_iou_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.grid_size, self.grid_size, 1, self.num_classes),
        ).squeeze(3)  # [batch_size, grid_size, grid_size, num_classes]

        # 1.a. Localization loss (x,y)
        loss_xy = self.mse(
            best_pred_box[..., :2] * obj_mask,
            target_boxes[..., :2] * obj_mask,
        )

        # 1.b. Localization loss (w,h)
        loss_wh = self.mse(
            torch.sqrt(best_pred_box[..., 2:4] * 1e-6) * obj_mask,
            torch.sqrt(target_boxes[..., 2:4] * 1e-6) * obj_mask,
        )

        # 2. Confidence loss
        # a. for cells with objects
        loss_conf = self.mse(best_pred_conf * obj_mask, target_confidences * obj_mask)
        # b. for cells with no objects
        noobj_mask = 1 - obj_mask
        loss_conf_noobj = self.mse(
            best_pred_conf * noobj_mask, torch.zeros_like(best_pred_conf) * noobj_mask
        )

        # 3. Classification loss
        loss_class = self.mse(best_pred_class * obj_mask, target_classes * obj_mask)

        return [
            (loss_xy + loss_wh) * self.lambda_coord / batch_size,
            loss_conf / batch_size,
            loss_conf_noobj * self.lambda_noobj / batch_size,
            loss_class / batch_size,
        ]

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

        best_pred_class = pred_classes.gather(
            dim=3,
            index=best_iou_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.grid_size, self.grid_size, 1, self.num_classes),
        ).squeeze(3)  # [batch_size, grid_size, grid_size, num_classes]

        # 1.a. Localization loss (x,y)
        loss_xy = self.mse(
            best_pred_box[..., :2] * obj_mask,
            target_boxes[..., :2] * obj_mask,
        )

        # 1.b. Localization loss (w,h)
        loss_wh = self.mse(
            torch.sqrt(best_pred_box[..., 2:4] * 1e-6) * obj_mask,
            torch.sqrt(target_boxes[..., 2:4] * 1e-6) * obj_mask,
        )

        # 2. Confidence loss
        # a. for cells with objects
        loss_conf = self.mse(best_pred_conf * obj_mask, target_confidences * obj_mask)
        # b. for cells with no objects
        noobj_mask = 1 - obj_mask
        loss_conf_noobj = self.mse(
            pred_confidences.squeeze() * noobj_mask,
            torch.zeros_like(pred_confidences.squeeze()) * noobj_mask,
        )

        # 3. Classification loss
        loss_class = self.mse(best_pred_class * obj_mask, target_classes * obj_mask)

        (
            localization_loss,
            confidence_loss,
            confidence_loss_noobj,
            classification_loss,
        ) = [
            (loss_xy + loss_wh) * self.lambda_coord,
            loss_conf,
            loss_conf_noobj * self.lambda_noobj,
            loss_class,
        ]

        loss_total = (
            localization_loss
            + confidence_loss
            + confidence_loss_noobj
            + classification_loss
        ) / batch_size
        return loss_total
