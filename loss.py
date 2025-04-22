import torch
import torch.nn as nn


def compute_iou(pred_boxes, gt_boxes):
    """
    Compute Intersection over Union (IoU) between predicted and ground-truth bounding boxes.

    Parameters:
        pred_boxes: Tensor of shape [batch, S, S, B, 4] in [x, y, w, h] format (all values normalized).
        gt_boxes: Tensor of shape [batch, S, S, 4] in [x, y, w, h] format, containing the ground truth
                  for each grid cell (if an object exists; otherwise these values are ignored).

    Returns:
        Tensor of shape [batch, S, S, B] with IoU values for each predicted box relative to the ground truth.
    """
    # Convert from [x, y, w, h] (center format) to [x_min, y_min, x_max, y_max]
    pred_x_min = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y_min = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x_max = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y_max = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    gt_x_min = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_y_min = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
    gt_x_max = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
    gt_y_max = gt_boxes[..., 1] + gt_boxes[..., 3] / 2

    # Calculate intersection coordinates
    inter_x_min = torch.maximum(pred_x_min, gt_x_min.unsqueeze(3))
    inter_y_min = torch.maximum(pred_y_min, gt_y_min.unsqueeze(3))
    inter_x_max = torch.minimum(pred_x_max, gt_x_max.unsqueeze(3))
    inter_y_max = torch.minimum(pred_y_max, gt_y_max.unsqueeze(3))

    # Compute intersection area
    inter_w = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_h = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_area = inter_w * inter_h

    # Compute areas of prediction and ground truth boxes
    pred_area = (pred_x_max - pred_x_min) * (pred_y_max - pred_y_min)
    gt_area = (gt_x_max - gt_x_min) * (gt_y_max - gt_y_min)

    # Since gt_area has shape [batch, S, S] we unsqueeze it to compare with B predictions
    gt_area = gt_area.unsqueeze(3)

    # Compute union area
    union_area = pred_area + gt_area - inter_area

    # Compute IoU, avoiding division by zero
    iou = inter_area / (union_area + 1e-6)
    return iou  # Shape: [batch, S, S, B]


class YOLOLoss(nn.Module):
    def __init__(self, S=20, B=2, num_classes=20, lambda_coord=5, lambda_noobj=0.5):
        """
        S: Number of grid cells along each dimension (e.g., 20 for a 20x20 grid).
        B: Number of bounding boxes predicted per grid cell.
        num_classes: Total number of object classes.
        lambda_coord: Weight for localization loss.
        lambda_noobj: Weight for confidence loss in cells with no objects.
        """
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, target):
        """
        predictions: Tensor of shape [batch, S, S, B, (5 + num_classes)]
                     Contains predictions for B bounding boxes in each grid cell.
        target: Tensor of shape [batch, S, S, (5 + num_classes)]
                For each cell, if an object exists then:
                - First 4 entries are [x, y, w, h] (normalized relative to the image)
                - Entry at index 4 is 1 (object confidence); 0 otherwise.
                - The remaining entries are one-hot encoded class labels.
        """
        batch_size = predictions.size(0)

        # Create an object mask: for each grid cell, 1 if an object is present, else 0.
        # Shape of obj_mask: [batch, S, S, 1]
        obj_mask = target[..., 4].unsqueeze(-1)

        # Extract ground truth bounding boxes and class info per grid cell
        gt_boxes = target[..., :4]  # [batch, S, S, 4]
        gt_conf = target[..., 4].unsqueeze(-1)  # [batch, S, S, 1]
        gt_class = target[..., 5:]  # [batch, S, S, num_classes]

        # Extract predicted bounding boxes, confidence and class predictions.
        # Predictions shape: [batch, S, S, B, (5+num_classes)]
        pred_boxes = predictions[..., :4]  # [batch, S, S, B, 4]
        pred_conf = predictions[..., 4]  # [batch, S, S, B]
        pred_class = predictions[..., 5:]  # [batch, S, S, B, num_classes]

        # Compute IoU between each predicted box and the ground-truth box (in each cell).
        # Note: For cells with no object, IoU values will be ignored.
        ious = compute_iou(pred_boxes, gt_boxes)  # Shape: [batch, S, S, B]

        # Select, for each grid cell, the predicted box with the highest IoU.
        best_ious, best_box_idx = torch.max(
            ious, dim=-1
        )  # Both have shape [batch, S, S]

        # Gather the best predicted bounding boxes for each grid cell.
        # We use gather along dimension 3 (the B dimension)
        best_box_idx_exp = (
            best_box_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, self.S, self.S, 1, 4)
        )
        pred_boxes_best = pred_boxes.gather(dim=3, index=best_box_idx_exp).squeeze(
            3
        )  # Shape: [batch, S, S, 4]

        # Similarly, gather best confidence scores.
        best_conf = (
            pred_conf.gather(dim=3, index=best_box_idx.unsqueeze(-1))
            .squeeze(3)
            .unsqueeze(-1)
        )  # [batch, S, S, 1]

        # And gather best class predictions.
        best_class = pred_class.gather(
            dim=3,
            index=best_box_idx.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, self.S, self.S, 1, self.num_classes),
        ).squeeze(3)  # [batch, S, S, num_classes]

        # -------------------------
        # 1. Localization Loss (for grid cells with objects)
        # -------------------------
        # Loss for x,y coordinates.
        loss_xy = self.mse(
            pred_boxes_best[..., :2] * obj_mask, gt_boxes[..., :2] * obj_mask
        )
        # Loss for width and height: apply square root (adding a small epsilon for stability)
        loss_wh = self.mse(
            torch.sqrt(pred_boxes_best[..., 2:4] + 1e-6) * obj_mask,
            torch.sqrt(gt_boxes[..., 2:4] + 1e-6) * obj_mask,
        )

        # -------------------------
        # 2. Confidence Loss
        # -------------------------
        # For cells with objects: use the confidence prediction of the best box.
        loss_conf_obj = self.mse(best_conf * obj_mask, gt_conf * obj_mask)

        # For cells without objects: we want *all* B predicted boxes to have low confidence.
        noobj_mask = 1 - obj_mask
        loss_conf_noobj = self.mse(
            pred_conf * noobj_mask, torch.zeros_like(pred_conf) * noobj_mask
        )

        # -------------------------
        # 3. Classification Loss (only for cells that contain an object)
        # -------------------------
        loss_class = self.mse(best_class * obj_mask, gt_class * obj_mask)

        # -------------------------
        # 4. Total Loss
        # -------------------------
        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_conf_obj
            + self.lambda_noobj * loss_conf_noobj
            + loss_class
        )

        total_loss = total_loss / batch_size
        return total_loss


# -------------------------
# Testing the YOLOLoss with Dummy Data
# -------------------------
if __name__ == "__main__":
    # Settings
    batch_size = 2
    S = 20
    B = 2
    num_classes = 16

    # Sample dummy predictions: [batch, S, S, B, (5 + num_classes)]
    dummy_predictions = torch.randn(batch_size, S, S, B, 5 + num_classes)

    # Sample dummy target: [batch, S, S, (5 + num_classes)]
    dummy_target = torch.zeros(batch_size, S, S, 5 + num_classes)
    # For the first sample, assume an object appears at cell (6,6)
    dummy_target[0, 6, 6, :4] = torch.tensor([0.5, 0.5, 1.0, 1.0])  # x, y, w, h
    dummy_target[0, 6, 6, 4] = 1.0  # Confidence = 1 (object present)
    dummy_target[0, 6, 6, 5] = 1.0  # One-hot: class 0

    # For the second sample, assume an object at cell (7,7)
    dummy_target[1, 7, 7, :4] = torch.tensor([0.3, 0.3, 0.8, 0.8])
    dummy_target[1, 7, 7, 4] = 1.0
    dummy_target[1, 7, 7, 10] = 1.0  # One-hot: e.g., class 5

    # Instantiate the loss and compute
    criterion = YOLOLoss(S=S, B=B, num_classes=num_classes)
    loss_value = criterion(dummy_predictions, dummy_target)
    print("Dummy YOLO Loss:", loss_value.item())
