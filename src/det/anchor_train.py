import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.ops import nms  # Permissively licensed (BSD)

# --- Configuration ---
NUM_CLASSES = 20  # Example: PASCAL VOC
IMG_SIZE = 640

# Define anchor boxes (width, height) for each scale.
# These are relative to the feature map cell size (stride).
# In a real scenario, these are determined by k-means clustering on your dataset.
# Format: [(w1,h1), (w2,h2), (w3,h3)] for one scale
# Let's make them simple and identical for all scales for this example.
# These would typically be different for P3, P4, P5.
# Values here are illustrative, not optimized.
ANCHORS = {
    "stride_8": torch.tensor([[10, 13], [16, 30], [33, 23]], dtype=torch.float32),
    "stride_16": torch.tensor([[30, 61], [62, 45], [59, 119]], dtype=torch.float32),
    "stride_32": torch.tensor([[116, 90], [156, 198], [373, 326]], dtype=torch.float32),
}
NUM_ANCHORS_PER_SCALE = ANCHORS["stride_8"].shape[0]  # Should be consistent


# --- Helper Modules ---
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_bn=True,
        activation=nn.LeakyReLU(0.1),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(
            channels, channels // 2, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = ConvBlock(
            channels // 2, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


# --- Backbone ---
class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Initial convolution
        self.conv_initial = ConvBlock(
            in_channels, 32, kernel_size=3, stride=1, padding=1
        )  # 640x640x32

        # Downsampling path
        self.layer1 = nn.Sequential(  # Stride 2
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),  # 320x320x64
            ResidualBlock(64),
        )
        self.layer2 = nn.Sequential(  # Stride 4
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # 160x160x128
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.layer3 = nn.Sequential(  # Stride 8 -> P3 output
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # 80x80x256
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )
        self.layer4 = nn.Sequential(  # Stride 16 -> P4 output
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),  # 40x40x512
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        self.layer5 = nn.Sequential(  # Stride 32 -> P5 output
            ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1),  # 20x20x1024
            ResidualBlock(1024),
            ResidualBlock(1024),
        )

    def forward(self, x):
        c1 = self.conv_initial(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)  # Output for stride 8 features (from subsequent FPN path)
        c4 = self.layer3(c3)  # Output for stride 16 features
        c5 = self.layer4(c4)  # Output for stride 32 features
        c6 = self.layer5(c5)  # Deepest features
        return c4, c5, c6  # P3, P4, P5 from backbone (renamed for clarity)


# --- Neck (FPN-like) ---
class FPNNeck(nn.Module):
    def __init__(
        self, C3_in_channels, C4_in_channels, C5_in_channels, neck_channels=256
    ):
        super().__init__()
        # Lateral connections (1x1 conv to reduce channels)
        self.lat_c5 = ConvBlock(
            C5_in_channels, neck_channels, kernel_size=1, stride=1, padding=0
        )
        self.lat_c4 = ConvBlock(
            C4_in_channels, neck_channels, kernel_size=1, stride=1, padding=0
        )
        self.lat_c3 = ConvBlock(
            C3_in_channels, neck_channels, kernel_size=1, stride=1, padding=0
        )

        # Top-down pathway smoothing (3x3 conv)
        self.smooth_p4 = ConvBlock(
            neck_channels, neck_channels, kernel_size=3, stride=1, padding=1
        )
        self.smooth_p3 = ConvBlock(
            neck_channels, neck_channels, kernel_size=3, stride=1, padding=1
        )

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, C3, C4, C5):
        # Top-down pathway
        P5 = self.lat_c5(C5)

        P4_lat = self.lat_c4(C4)
        P5_upsampled = self.upsample(P5)
        P4 = self.smooth_p4(P4_lat + P5_upsampled)  # Element-wise sum

        P3_lat = self.lat_c3(C3)
        P4_upsampled = self.upsample(P4)
        P3 = self.smooth_p3(P3_lat + P4_upsampled)  # Element-wise sum

        return P3, P4, P5  # Features for detection heads (stride 8, 16, 32)


# --- Detection Head ---
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        self.num_outputs_per_anchor = 5 + num_classes  # x, y, w, h, conf, class_probs

        # A few conv layers before the final prediction layer
        self.conv_intermediate = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        )
        self.pred_conv = nn.Conv2d(
            in_channels,
            num_anchors * self.num_outputs_per_anchor,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.conv_intermediate(x)
        x = self.pred_conv(x)
        # x shape: (batch_size, num_anchors * (5 + num_classes), grid_h, grid_w)
        bs, _, grid_h, grid_w = x.shape
        # Reshape to: (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        x = x.view(
            bs, NUM_ANCHORS_PER_SCALE, self.num_outputs_per_anchor, grid_h, grid_w
        )
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x


# --- Full YOLO-like Model ---
class YOLOLikeModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_anchors_per_scale = NUM_ANCHORS_PER_SCALE

        # Backbone (example channel outputs)
        # C3: 256 channels (80x80)
        # C4: 512 channels (40x40)
        # C5: 1024 channels (20x20)
        self.backbone = SimpleBackbone(in_channels=3)

        # FPN Neck
        self.fpn_neck_channels = 256  # Common channel dimension for FPN outputs
        self.fpn = FPNNeck(
            C3_in_channels=256,  # From backbone.layer3
            C4_in_channels=512,  # From backbone.layer4
            C5_in_channels=1024,  # From backbone.layer5
            neck_channels=self.fpn_neck_channels,
        )

        # Detection Heads
        self.head_p3 = DetectionHead(
            self.fpn_neck_channels, num_classes, self.num_anchors_per_scale
        )  # Stride 8
        self.head_p4 = DetectionHead(
            self.fpn_neck_channels, num_classes, self.num_anchors_per_scale
        )  # Stride 16
        self.head_p5 = DetectionHead(
            self.fpn_neck_channels, num_classes, self.num_anchors_per_scale
        )  # Stride 32

        # Store anchors and strides for decoding
        self.anchors = ANCHORS
        self.strides = torch.tensor([8.0, 16.0, 32.0])

        self._initialize_weights()  # Optional: better weight initialization

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        c3_bb, c4_bb, c5_bb = self.backbone(x)
        p3_fpn, p4_fpn, p5_fpn = self.fpn(c3_bb, c4_bb, c5_bb)

        out_p3 = self.head_p3(p3_fpn)  # (bs, 80, 80, num_anchors, 5+C)
        out_p4 = self.head_p4(p4_fpn)  # (bs, 40, 40, num_anchors, 5+C)
        out_p5 = self.head_p5(p5_fpn)  # (bs, 20, 20, num_anchors, 5+C)

        # During training, these outputs are used by the loss function.
        # For inference, we often concatenate and then decode.
        # If training, return raw outputs for loss calculation
        if self.training:
            return [out_p3, out_p4, out_p5]
        else:
            # For inference, decode predictions
            # We need to process these outputs to get actual bounding boxes
            # This typically happens outside the model in a post-processing step
            # But we can do it here for convenience if desired, or in a separate function
            return self.decode_and_nms([out_p3, out_p4, out_p5])

    def decode_predictions(self, preds_per_scale, conf_thresh=0.25):
        """
        Decodes raw model outputs into bounding boxes.
        preds_per_scale: List of tensors, one for each scale.
                         Each tensor shape: (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        Returns:
            List of detections for each image in the batch.
            Each item in the list is a tensor of shape [num_detections, 6] (x1, y1, x2, y2, score, class_idx)
        """
        all_batch_detections = [
            [] for _ in range(preds_per_scale[0].shape[0])
        ]  # One list per image in batch

        for i, preds in enumerate(preds_per_scale):
            stride = self.strides[i].item()
            # Select anchors for the current scale/stride
            if stride == 8:
                current_anchors = self.anchors["stride_8"]
            elif stride == 16:
                current_anchors = self.anchors["stride_16"]
            else:
                current_anchors = self.anchors["stride_32"]
            current_anchors = current_anchors.to(preds.device)

            batch_size, grid_h, grid_w, num_anchors, _ = preds.shape

            # Create grid cells (cx, cy)
            gy, gx = torch.meshgrid(
                torch.arange(grid_h, device=preds.device),
                torch.arange(grid_w, device=preds.device),
                indexing="ij",
            )
            grid = torch.stack((gx, gy), dim=-1).float()  # Shape: (grid_h, grid_w, 2)
            grid = grid.view(1, grid_h, grid_w, 1, 2).repeat(
                batch_size, 1, 1, num_anchors, 1
            )

            # Predictions
            # Box coordinates (tx, ty, tw, th)
            box_xy = torch.sigmoid(preds[..., 0:2])  # sigmoid for x, y offsets
            box_wh = torch.exp(preds[..., 2:4]) * current_anchors.view(
                1, 1, 1, num_anchors, 2
            )  # exp for w, h scales

            # Calculate actual box coordinates (center_x, center_y, width, height) on the grid
            pred_boxes_center_xy = (box_xy + grid) * stride
            pred_boxes_wh = (
                box_wh * stride
            )  # In yolov3, anchors are relative to input image size. Here, relative to stride.
            # If anchors are absolute (like in ultralytics): pred_boxes_wh = box_wh (already scaled by anchors)

            # Objectness score and class probabilities
            conf = torch.sigmoid(preds[..., 4:5])
            class_probs = torch.sigmoid(
                preds[..., 5:]
            )  # Or softmax if your loss uses CrossEntropy

            # Filter by confidence threshold
            candidates = conf > conf_thresh

            for batch_idx in range(batch_size):
                batch_preds_boxes_center_xy = pred_boxes_center_xy[batch_idx][
                    candidates[batch_idx]
                ]
                batch_preds_boxes_wh = pred_boxes_wh[batch_idx][candidates[batch_idx]]
                batch_conf = conf[batch_idx][candidates[batch_idx]]
                batch_class_probs = class_probs[batch_idx][candidates[batch_idx]]

                if batch_preds_boxes_center_xy.numel() == 0:
                    continue

                # Convert to (x1, y1, x2, y2)
                x1 = (
                    batch_preds_boxes_center_xy[..., 0]
                    - batch_preds_boxes_wh[..., 0] / 2
                )
                y1 = (
                    batch_preds_boxes_center_xy[..., 1]
                    - batch_preds_boxes_wh[..., 1] / 2
                )
                x2 = (
                    batch_preds_boxes_center_xy[..., 0]
                    + batch_preds_boxes_wh[..., 0] / 2
                )
                y2 = (
                    batch_preds_boxes_center_xy[..., 1]
                    + batch_preds_boxes_wh[..., 1] / 2
                )

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

                # Get best class score and best class index
                best_class_scores, best_class_idx = torch.max(batch_class_probs, dim=-1)

                # Final scores: objectness_confidence * class_confidence
                final_scores = batch_conf.squeeze(-1) * best_class_scores

                # Append to this image's detections for this scale
                # Detections format: [x1, y1, x2, y2, score, class_idx]
                detections_this_scale = torch.cat(
                    [
                        boxes,
                        final_scores.unsqueeze(-1),
                        best_class_idx.float().unsqueeze(-1),
                    ],
                    dim=-1,
                )

                all_batch_detections[batch_idx].append(detections_this_scale)

        # Concatenate detections from all scales for each image and apply NMS
        output_detections = []
        for batch_idx in range(preds_per_scale[0].shape[0]):
            if not all_batch_detections[batch_idx]:
                output_detections.append(
                    torch.empty(0, 6, device=preds_per_scale[0].device)
                )
                continue

            img_detections = torch.cat(all_batch_detections[batch_idx], dim=0)

            # Clamp boxes to image dimensions
            img_detections[:, 0] = torch.clamp(
                img_detections[:, 0], min=0, max=self.img_size - 1
            )
            img_detections[:, 1] = torch.clamp(
                img_detections[:, 1], min=0, max=self.img_size - 1
            )
            img_detections[:, 2] = torch.clamp(
                img_detections[:, 2], min=0, max=self.img_size - 1
            )
            img_detections[:, 3] = torch.clamp(
                img_detections[:, 3], min=0, max=self.img_size - 1
            )

            output_detections.append(img_detections)

        return output_detections  # List of tensors [N_dets, 6] for each image in batch

    def decode_and_nms(self, preds_per_scale, conf_thresh=0.25, iou_thresh=0.45):
        decoded_outputs = self.decode_predictions(preds_per_scale, conf_thresh)

        final_batch_detections = []
        for img_detections in decoded_outputs:
            if img_detections.numel() == 0:
                final_batch_detections.append(
                    torch.empty(0, 6, device=img_detections.device)
                )
                continue

            # NMS per class (more robust, but slower. Common practice is NMS across all classes at once if classes are mutually exclusive)
            # For simplicity, let's do NMS across all classes after filtering by score.
            # If you want per-class NMS, you'd loop through unique class_idx.

            boxes_for_nms = img_detections[:, :4]  # x1,y1,x2,y2
            scores_for_nms = img_detections[:, 4]  # final_scores

            # torchvision.ops.nms expects boxes, scores
            keep_indices = nms(boxes_for_nms, scores_for_nms, iou_thresh)

            final_batch_detections.append(img_detections[keep_indices])

        return final_batch_detections


# --- Loss Function (Conceptual - NOT part of the model definition but needed for training) ---
# A proper YOLO loss is complex. It involves:
# 1. Matching predicted boxes to ground truth boxes (based on IoU and anchor assignment).
# 2. Calculating:
#    - Bounding box regression loss (e.g., GIoU, CIoU loss) for matched positive predictions.
#    - Objectness loss (BCEWithLogitsLoss) for all predictions (positive and negative).
#    - Classification loss (BCEWithLogitsLoss or CrossEntropyLoss) for matched positive predictions.
# 3. Balancing these losses.

# This is a placeholder to illustrate the components.
# You'd need to implement a `compute_loss(predictions, targets)` function.
# `predictions` would be the list [out_p3, out_p4, out_p5] from model.forward() in training mode.
# `targets` would be your ground truth labels formatted appropriately.


def conceptual_yolo_loss(predictions_per_scale, targets, model):
    # predictions_per_scale: list of tensors from model output, e.g. [P3_out, P4_out, P5_out]
    # targets: list of ground truth boxes and classes for each image in the batch
    # model: the YOLOLikeModel instance to access anchors, strides, etc.

    loss_box = 0.0
    loss_obj = 0.0
    loss_cls = 0.0

    # For each scale
    for i, preds_at_scale in enumerate(predictions_per_scale):
        bs, gh, gw, na, nc_plus_5 = preds_at_scale.shape
        stride = model.strides[i]
        # Anchors for this scale (e.g., model.anchors["stride_8"])
        # ...

        # 1. Prepare targets: Map ground truth boxes to the grid cells and anchors of this scale.
        #    This involves:
        #    - Calculating IoU between GT boxes and anchor templates.
        #    - Assigning GTs to responsible grid cells and anchors.
        #    - Creating masks for positive (object) and negative (no object) predictions.
        #    - Calculating target tx, ty, tw, th, target_obj, target_cls.
        # This is the most complex part.

        # Example (pseudo-code for one scale):
        # target_tx, target_ty, target_tw, target_th, obj_mask, noobj_mask, target_cls = build_targets(targets, preds_at_scale, anchors_this_scale, stride)

        # 2. Extract predictions
        pred_xy = torch.sigmoid(preds_at_scale[..., 0:2])
        pred_wh = preds_at_scale[
            ..., 2:4
        ]  # Raw tw, th (exp will be applied if needed or loss handles it)
        pred_obj = preds_at_scale[..., 4:5]
        pred_cls = preds_at_scale[..., 5:]

        # 3. Calculate losses (using BCEWithLogitsLoss for obj and cls, and some IoU loss for box)
        # loss_box += some_iou_loss(pred_xywh[obj_mask], target_xywh[obj_mask])
        # loss_obj += bce_obj_loss(pred_obj[obj_mask], target_obj[obj_mask]) + \
        #             bce_obj_loss(pred_obj[noobj_mask], target_obj[noobj_mask]) # Often weighted
        # loss_cls += bce_cls_loss(pred_cls[obj_mask], target_cls[obj_mask])
        pass  # This is very involved

    total_loss = loss_box + loss_obj + loss_cls
    return total_loss


# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model instance
    model = YOLOLikeModel(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    model.eval()  # Set to evaluation mode for inference

    # Create a dummy input image (batch size 1, 3 channels, 640x640)
    dummy_input = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)  # Batch size 2

    print(f"Input shape: {dummy_input.shape}")

    # --- Test Backbone ---
    print("\n--- Testing Backbone ---")
    backbone = SimpleBackbone().to(device)
    c3_bb, c4_bb, c5_bb = backbone(dummy_input)
    print(f"Backbone C3 out shape: {c3_bb.shape}")  # Expected: [B, 256, 80, 80]
    print(f"Backbone C4 out shape: {c4_bb.shape}")  # Expected: [B, 512, 40, 40]
    print(f"Backbone C5 out shape: {c5_bb.shape}")  # Expected: [B, 1024, 20, 20]

    # --- Test FPN ---
    print("\n--- Testing FPN ---")
    fpn_neck_channels = 256
    fpn = FPNNeck(
        C3_in_channels=c3_bb.shape[1],
        C4_in_channels=c4_bb.shape[1],
        C5_in_channels=c5_bb.shape[1],
        neck_channels=fpn_neck_channels,
    ).to(device)
    p3_fpn, p4_fpn, p5_fpn = fpn(c3_bb, c4_bb, c5_bb)
    print(
        f"FPN P3 out shape: {p3_fpn.shape}"
    )  # Expected: [B, fpn_neck_channels, 80, 80]
    print(
        f"FPN P4 out shape: {p4_fpn.shape}"
    )  # Expected: [B, fpn_neck_channels, 40, 40]
    print(
        f"FPN P5 out shape: {p5_fpn.shape}"
    )  # Expected: [B, fpn_neck_channels, 20, 20]

    # --- Test Detection Head ---
    print("\n--- Testing Detection Head ---")
    head = DetectionHead(fpn_neck_channels, NUM_CLASSES, NUM_ANCHORS_PER_SCALE).to(
        device
    )
    head_out_p3 = head(p3_fpn)
    print(
        f"Head P3 out shape: {head_out_p3.shape}"
    )  # Expected: [B, 80, 80, NUM_ANCHORS, 5+NUM_CLASSES]

    # --- Test Full Model Inference ---
    print("\n--- Testing Full Model (Inference) ---")
    with torch.no_grad():  # Disable gradient calculations for inference
        # When model.training is False, forward() calls decode_and_nms()
        detections = model(dummy_input)

    print(f"Number of images in batch for detections: {len(detections)}")
    for i, img_dets in enumerate(detections):
        print(
            f"Image {i+1} detections shape: {img_dets.shape}"
        )  # [num_detected_boxes, 6]
        if img_dets.numel() > 0:
            print(
                f"Example detection for image {i+1} (x1,y1,x2,y2,score,class): \n{img_dets[0]}"
            )
        else:
            print(
                f"No detections for image {i+1} (with default thresholds and random weights)"
            )

    # --- Test Full Model Training Output ---
    print("\n--- Testing Full Model (Training Output) ---")
    model.train()  # Set to training mode
    train_outputs = model(dummy_input)  # Returns list of raw outputs per scale
    print(f"Number of output scales for training: {len(train_outputs)}")
    for i, scale_out in enumerate(train_outputs):
        print(f"Scale {i} output shape (for loss): {scale_out.shape}")
        # Expected: (batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        # e.g., [B, 80, 80, 3, 25], [B, 40, 40, 3, 25], [B, 20, 20, 3, 25]

    # Summary of the model
    # from torchinfo import summary # if you have torchinfo installed: pip install torchinfo
    # summary(model, input_size=(2, 3, IMG_SIZE, IMG_SIZE))
