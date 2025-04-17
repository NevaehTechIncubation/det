import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, S=20, B=1, num_classes=20, lambda_coord=5, lambda_noobj=0.5):
        """
        S: Number of grid cells along width/height (e.g., 20 for a 20x20 grid).
        B: Number of bounding boxes predicted per cell (here, simplified to 1).
        num_classes: Total number of object classes.
        lambda_coord: Weight for coordinate loss.
        lambda_noobj: Weight for the confidence loss on cells with no objects.
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
        predictions: Tensor of shape [batch, S, S, (5 + num_classes)]
                     Contains: [x, y, w, h, confidence, class1, ..., classN]
        target: Tensor of shape [batch, S, S, (5 + num_classes)]
                For each grid cell:
                  - The first 4 values are the ground truth bounding box (x, y, w, h),
                  - Index 4 is 1 if an object is present (0 otherwise),
                  - The rest is a one-hot encoding for the object's class.
        """
        batch_size = predictions.size(0)

        # Create an object mask: for cells with an object, mask==1; otherwise 0.
        # We assume the confidence target (at index 4) is 1 for object-present cells.
        obj_mask = target[..., 4].unsqueeze(-1)  # Shape: [batch, S, S, 1]

        # -------------------------
        # 1. Localization Loss
        # -------------------------
        # For x, y coordinates (predicted values are relative to the grid cell)
        loss_xy = self.mse(predictions[..., :2] * obj_mask, target[..., :2] * obj_mask)

        # For width and height, YOLO applies a square root to stabilize gradients.
        pred_wh = predictions[..., 2:4]
        target_wh = target[..., 2:4]
        loss_wh = self.mse(
            torch.sqrt(pred_wh + 1e-6) * obj_mask,
            torch.sqrt(target_wh + 1e-6) * obj_mask,
        )

        # -------------------------
        # 2. Confidence Loss
        # -------------------------
        # Confidence predictions are at index 4.
        pred_conf = predictions[..., 4:5]  # Shape: [batch, S, S, 1]
        target_conf = target[..., 4:5]

        # Loss for cells with objects:
        loss_conf_obj = self.mse(pred_conf * obj_mask, target_conf * obj_mask)

        # Loss for cells without objects:
        noobj_mask = 1 - obj_mask
        loss_conf_noobj = self.mse(pred_conf * noobj_mask, target_conf * noobj_mask)

        # -------------------------
        # 3. Classification Loss
        # -------------------------
        # The class predictions start from index 5 onward.
        pred_class = predictions[..., 5:]  # Shape: [batch, S, S, num_classes]
        target_class = target[..., 5:]
        loss_class = self.mse(pred_class * obj_mask, target_class * obj_mask)

        # -------------------------
        # Total Loss
        # -------------------------
        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_conf_obj
            + self.lambda_noobj * loss_conf_noobj
            + loss_class
        )

        # Average loss over batch
        total_loss = total_loss / batch_size
        return total_loss


# Test the YOLO loss with dummy data.
if __name__ == "__main__":
    # Suppose we have a 20x20 grid, 1 bounding box per cell, and 20 classes.
    batch_size = 2
    S = 20
    num_classes = 20

    # Create dummy predictions: random values for demonstration.
    dummy_predictions = torch.randn(batch_size, S, S, 5 + num_classes)

    # Create a dummy target tensor.
    # For simplicity, assume that only one cell per sample contains an object.
    dummy_target = torch.zeros(batch_size, S, S, 5 + num_classes)

    # For the first sample, assume an object is at cell (6,6):
    dummy_target[0, 6, 6, :4] = torch.tensor([0.5, 0.5, 1.0, 1.0])  # x, y, w, h
    dummy_target[0, 6, 6, 4] = 1.0  # Confidence = 1
    dummy_target[0, 6, 6, 5] = 1.0  # One-hot for class 0 (for example)

    # For the second sample, assume an object is at cell (7,7):
    dummy_target[1, 7, 7, :4] = torch.tensor([0.3, 0.3, 0.8, 0.8])
    dummy_target[1, 7, 7, 4] = 1.0
    dummy_target[1, 7, 7, 10] = 1.0  # One-hot for class 5 (for example)

    # Instantiate the loss.
    criterion = YOLOLoss(S=S, B=1, num_classes=num_classes)
    loss_value = criterion(dummy_predictions, dummy_target)
    print("Dummy YOLO loss:", loss_value.item())
