import torch
from torchvision.ops import nms


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two sets of boxes.

    Parameters:
        box1: Tensor of shape [4], format [x_min, y_min, x_max, y_max]
        box2: Tensor of shape [4], format [x_min, y_min, x_max, y_max]

    Returns:
        IoU value (float).
    """

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

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
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5]

    # breakpoint()
    keep_indices = nms(boxes, scores, iou_threshold)
    return predictions[keep_indices]


def validate_model(model, dataloader, iou_threshold=0.5, conf_threshold=0.0):
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
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in dataloader:
            device = next(model.parameters()).device
            images = images.to(device)
            targets = targets.to(device)

            raw_predictions = model(images)

            for i, image_preds in enumerate(raw_predictions):
                processed_preds = convert_predictions_to_boxes_vectorized(
                    image_preds.unsqueeze(0), conf_threshold
                )

                # if processed_preds.size(-1) == 0:
                #     continue
                filtered_preds = non_maximum_suppression(processed_preds, iou_threshold)
                # print(filtered_preds.size())

                all_predictions.append(filtered_preds)
                all_ground_truths.append(convert_targets_to_boxes(targets[i]))

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

    S = predictions.shape[0]
    B = predictions.shape[2]
    num_classes = predictions.shape[-1] - 5

    for i in range(S):
        for j in range(S):
            for b in range(B):
                box = predictions[i, j, b]
                confidence = box[4]

                if confidence < conf_threshold:
                    continue

                x_center, y_center, w, h = box[:4]
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                class_probs = box[5:]
                class_id = torch.argmax(class_probs)

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

    box_coords = predictions[..., :4]
    confidences = predictions[..., 4]
    class_probs = predictions[..., 5:]

    class_scores, class_ids = torch.max(class_probs, dim=-1)

    mask = confidences > conf_threshold

    grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    grid_x = grid_x.to(predictions.device).unsqueeze(-1).unsqueeze(0)
    grid_y = grid_y.to(predictions.device).unsqueeze(-1).unsqueeze(0)

    x_center = (box_coords[..., 0] + grid_x) / S * image_size
    y_center = (box_coords[..., 1] + grid_y) / S * image_size
    w = box_coords[..., 2] * image_size
    h = box_coords[..., 3] * image_size

    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

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
    )

    all_boxes = all_boxes[mask]

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
    S = target.shape[0]
    for i in range(S):
        for j in range(S):
            if target[i, j, 4] > 0:
                x_offset, y_offset, w, h = target[i, j, :4]
                class_id = torch.argmax(target[i, j, 5:])

                x_center = (j + x_offset) / S * image_size
                y_center = (i + y_offset) / S * image_size
                w = w * image_size
                h = h * image_size

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
    tp = 0
    fp = 0
    fn = 0

    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        matched_gt = torch.zeros(gt_boxes.shape[0])

        for pred_box in pred_boxes:
            ious = torch.tensor(
                [compute_iou(pred_box[:4], gt_box[:4]) for gt_box in gt_boxes]
            )
            max_iou, gt_idx = torch.max(ious, dim=0)
            if max_iou >= iou_threshold and matched_gt[gt_idx] == 0:
                tp += 1
                matched_gt[gt_idx] = 1
            else:
                fp += 1

        fn += torch.sum(matched_gt == 0).item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1}
