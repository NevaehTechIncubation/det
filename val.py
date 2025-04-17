import torch
from torchvision.ops import nms  # Torch's built-in Non-Maximum Suppression


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Parameters:
        box1: Tensor of shape [4], format [x_min, y_min, x_max, y_max]
        box2: Tensor of shape [4], format [x_min, y_min, x_max, y_max]

    Returns:
        IoU value (float).
    """
    # Calculate intersection
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def non_maximum_suppression(predictions, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.

    Parameters:
        predictions: Tensor of shape [num_boxes, 6], where each box is [x_min, y_min, x_max, y_max, confidence, class].
        iou_threshold: IoU threshold to filter overlapping boxes.

    Returns:
        Filtered predictions.
    """
    boxes = predictions[:, :4]  # Bounding box coordinates
    scores = predictions[:, 4]  # Confidence scores
    classes = predictions[:, 5]  # Class labels

    # Perform NMS
    keep_indices = nms(boxes, scores, iou_threshold)
    return predictions[keep_indices]


def validate_model(model, dataloader, iou_threshold=0.5, conf_threshold=0.5):
    """
    Perform validation on a dataset.

    Parameters:
        model: Trained YOLO model.
        dataloader: Validation DataLoader.
        iou_threshold: IoU threshold for Non-Maximum Suppression.
        conf_threshold: Confidence threshold to filter low-confidence predictions.

    Returns:
        Dictionary with validation metrics (e.g., mAP, precision, recall).
    """
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():  # Disable gradient computation
        for images, targets in dataloader:
            # Move data to the same device as the model
            device = next(model.parameters()).device
            images = images.to(device)
            targets = targets.to(device)  # Ground truth in grid cell format

            # Forward pass
            raw_predictions = model(images)

            # Post-process predictions for each image
            for i, image_preds in enumerate(raw_predictions):
                # Convert raw grid cell predictions to bounding boxes (scale to image size)
                processed_preds = convert_predictions_to_boxes(
                    image_preds, conf_threshold
                )

                # Apply Non-Maximum Suppression
                filtered_preds = non_maximum_suppression(processed_preds, iou_threshold)

                # Store predictions and corresponding ground truths
                all_predictions.append(filtered_preds)
                all_ground_truths.append(convert_targets_to_boxes(targets[i]))

        # Compute evaluation metrics
        metrics = compute_validation_metrics(all_predictions, all_ground_truths)
        return metrics


def convert_predictions_to_boxes(predictions, conf_threshold):
    """
    Convert raw predictions (grid format) to bounding boxes.

    Parameters:
        predictions: Tensor of shape [S, S, B, 5 + num_classes].
        conf_threshold: Threshold to filter low-confidence predictions.

    Returns:
        Tensor of bounding boxes in the format [x_min, y_min, x_max, y_max, confidence, class].
    """
    boxes = []

    S = predictions.shape[0]  # Grid size
    B = predictions.shape[2]  # Number of boxes per grid cell
    num_classes = predictions.shape[-1] - 5  # Classes

    for i in range(S):
        for j in range(S):
            for b in range(B):
                box = predictions[i, j, b]
                confidence = box[4]

                # Filter boxes with confidence below threshold
                if confidence < conf_threshold:
                    continue

                # Compute bounding box coordinates relative to the image size
                x_center, y_center, w, h = box[:4]
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                # Find class with highest probability
                class_probs = box[5:]
                class_id = torch.argmax(class_probs)

                # Append box with format [x_min, y_min, x_max, y_max, confidence, class]
                boxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

    return torch.tensor(boxes)


def convert_predictions_to_boxes_vectorized(
    predictions, conf_threshold=0.5, S=20, image_size=640
):
    """
    Vectorized conversion of raw YOLO predictions (grid format) to bounding boxes.

    Parameters:
        predictions: Tensor of shape [batch_size, S, S, B, 5 + num_classes].
                     Contains grid cell predictions for bounding boxes.
        conf_threshold: Confidence threshold to filter predictions.
        S: Grid size (e.g., 20 for a 20x20 grid).
        image_size: Image size (e.g., 640 for a 640x640 image).

    Returns:
        List of tensors, each containing bounding boxes in the format
        [x_min, y_min, x_max, y_max, confidence, class].
    """
    batch_size, S, S, B, num_features = predictions.shape
    num_classes = num_features - 5

    # 1. Extract bounding box attributes
    box_coords = predictions[..., :4]  # [x_offset, y_offset, w, h]
    confidences = predictions[..., 4]  # Confidence scores
    class_probs = predictions[..., 5:]  # Class probabilities

    # 2. Calculate class with the highest probability for each prediction
    class_scores, class_ids = torch.max(class_probs, dim=-1)

    # 3. Filter predictions by confidence threshold
    mask = confidences > conf_threshold  # Boolean mask for valid predictions

    # 4. Compute absolute bounding box coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S), torch.arange(S), indexing="ij"
    )  # Grid cell indices
    grid_x = (
        grid_x.to(predictions.device).unsqueeze(-1).unsqueeze(0)
    )  # Shape: [1, S, S, 1]
    grid_y = (
        grid_y.to(predictions.device).unsqueeze(-1).unsqueeze(0)
    )  # Shape: [1, S, S, 1]

    x_center = (box_coords[..., 0] + grid_x) / S * image_size
    y_center = (box_coords[..., 1] + grid_y) / S * image_size
    w = box_coords[..., 2] * image_size
    h = box_coords[..., 3] * image_size

    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    # 5. Combine results
    all_boxes = torch.cat(
        [
            x_min.unsqueeze(-1),
            y_min.unsqueeze(-1),
            x_max.unsqueeze(-1),
            y_max.unsqueeze(-1),
            confidences.unsqueeze(-1),
            class_ids.unsqueeze(-1),
        ],
        dim=-1,
    )  # [batch_size, S, S, B, 6]

    # 6. Apply mask to filter low-confidence predictions
    all_boxes = all_boxes[mask]  # Shape: [filtered_boxes, 6]

    return all_boxes


def convert_targets_to_boxes(target, image_size=640):
    """
    Convert grid cell ground-truth targets to bounding box format.

    Parameters:
        target: Tensor of shape [S, S, 5 + num_classes].
        image_size: Size of the input image (e.g., 640 for a 640x640 image).

    Returns:
        List of bounding boxes in the format [x_min, y_min, x_max, y_max, class_id].
    """
    boxes = []
    S = target.shape[0]  # Grid size
    for i in range(S):
        for j in range(S):
            if target[i, j, 4] > 0:  # Check object confidence
                # Get bounding box attributes
                x_offset, y_offset, w, h = target[i, j, :4]
                class_id = torch.argmax(target[i, j, 5:])  # Class label

                # Convert offsets to actual image coordinates
                x_center = (j + x_offset) / S * image_size
                y_center = (i + y_offset) / S * image_size
                w = w * image_size
                h = h * image_size

                # Convert to [x_min, y_min, x_max, y_max] format
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                boxes.append([x_min, y_min, x_max, y_max, class_id])
    return torch.tensor(boxes)


def compute_validation_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute validation metrics (IoU, precision, recall, mAP).

    Parameters:
        predictions: List of predicted bounding boxes (from NMS).
                     Each element is a tensor of shape [num_boxes, 6], format [x_min, y_min, x_max, y_max, confidence, class].
        ground_truths: List of ground-truth bounding boxes.
                      Each element is a tensor of shape [num_boxes, 5], format [x_min, y_min, x_max, y_max, class].
        iou_threshold: IoU threshold for true positive classification.

    Returns:
        Dictionary with validation metrics.
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matched_gt = torch.zeros(
            gt_boxes.shape[0]
        )  # Keep track of matched ground truths

        for pred_box in pred_boxes:
            ious = torch.tensor(
                [compute_iou(pred_box[:4], gt_box[:4]) for gt_box in gt_boxes]
            )
            max_iou, gt_idx = torch.max(ious, dim=0)
            if max_iou >= iou_threshold and matched_gt[gt_idx] == 0:  # True positive
                tp += 1
                matched_gt[gt_idx] = 1  # Mark ground truth as matched
            else:  # False positive
                fp += 1

        # Count false negatives (ground truths not matched)
        fn += torch.sum(matched_gt == 0).item()

    # Compute metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1}
