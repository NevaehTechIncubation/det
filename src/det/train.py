from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import nms
import yaml

from det.data import create_dataloader
from det.loss import YOLOLoss


def get_num_classes(dataset_dir: Path) -> int:
    data = yaml.safe_load(dataset_dir.joinpath("data.yaml").read_text())
    return len(data["names"])


def train(
    model: nn.Module,
    dataset_dir: Path,
    num_epochs: int,
    batch_size: int,
    image_size: int = 640,
    num_workers: int = 0,
    device: str = "cpu",
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),  # ImageNet stats
            # transforms.Normalize(
            #     [0.94827438, 0.94827438, 0.94827438],
            #     [0.21538803, 0.21538803, 0.21538803],
            # ), # CHC stats
        ]
    )
    num_classes = get_num_classes(dataset_dir)
    train_dataloader, val_dataloader = create_dataloader(
        dataset_dir,
        num_classes,
        image_size,
        batch_size,
        20,
        transform,
        num_workers,
    )

    criterion = YOLOLoss(grid_size=20, num_boxes=1, num_classes=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # print(f"{running_loss / len(val_dataloader)}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        metrics = validate_model(
            model,
            val_dataloader,
            num_classes,
            grid_size=20,
            num_boxes=1,
            device=device,
        )


def validate_model(
    model,
    dataloader,
    num_classes,
    grid_size,
    num_boxes,
    device="cpu",
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    criterion=None,
):
    criterion = (
        YOLOLoss(grid_size=grid_size, num_boxes=num_boxes, num_classes=num_classes)
        if criterion is None
        else criterion
    )
    with torch.no_grad():
        model.eval()
        running_val_loss = 0.0
        running_val_loss_detailed = torch.tensor([0.0, 0.0, 0.0, 0.0])
        all_predictions = []
        all_ground_truths = []
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            raw_predictions = model(images)
            detailed_loss = criterion.compute_loss(raw_predictions, images)
            running_val_loss_detailed += detailed_loss
            running_val_loss += detailed_loss.sum().item()

            for i, predictions in enumerate(raw_predictions):
                processed_preds = convert_predictions_to_boxes(
                    predictions, conf_threshold=conf_threshold
                )
                filtered_preds = non_maximum_suppression(processed_preds, iou_threshold)

                all_predictions.append(filtered_preds)
                all_ground_truths.append(convert_targets_to_boxes(labels[i]))

        metrics = compute_validation_metrics(all_predictions, all_ground_truths)
        val_loss = running_val_loss / len(dataloader)


def convert_predictions_to_boxes(
    predictions, conf_threshold=0.5, grid_size=20, image_size=640
):
    """
    Vectorized conversion of raw YOLO predictions (grid format) to bounding boxes.

    Parameters:
        predictions: Tensor of shape [S, S, B, 5 + num_classes].
                     Contains grid cell predictions for bounding boxes.
        conf_threshold: Confidence threshold to filter predictions.
        S: Grid size (e.g., 20 for a 20x20 grid).
        image_size: Image size (e.g., 640 for a 640x640 image).

    Returns:
        List of tensors, each containing bounding boxes in the format
        [x_min, y_min, x_max, y_max, confidence, class].
    """
    grid_size, grid_size, B, num_features = predictions.shape
    num_classes = num_features - 5

    box_coords = predictions[..., :4]
    confidences = predictions[..., 4]
    class_probs = predictions[..., 5:]

    class_scores, class_ids = torch.max(class_probs, dim=-1)

    mask = confidences > conf_threshold

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
    )
    grid_x = grid_x.to(predictions.device).unsqueeze(-1).unsqueeze(0)
    grid_y = grid_y.to(predictions.device).unsqueeze(-1).unsqueeze(0)

    x_center = (box_coords[..., 0] + grid_x) / grid_size * image_size
    y_center = (box_coords[..., 1] + grid_y) / grid_size * image_size
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
    Convert grid cell ground-truth targets to bounding box format in a vectorized manner.

    Parameters:
        target: Tensor of shape [S, S, 5 + num_classes].
                For each grid cell the first 4 values are [x_offset, y_offset, w, h],
                the 5th value is the object confidence (1 for an object, 0 for no object),
                and the remaining values are a one-hot encoding of the class.
        image_size: The size (in pixels) of the input image (e.g., 640 for a 640x640 image).

    Returns:
        Tensor of shape [N, 5], where N is the number of grid cells with an object.
        Each row is in the format [x_min, y_min, x_max, y_max, class_id].
    """
    S = target.shape[0]
    num_classes = target.shape[-1] - 5

    # Create a grid for the S x S cells.
    # We use torch.meshgrid; note that the first output corresponds to rows (y coordinate)
    # and the second to columns (x coordinate). (i, j) here means: i = row index, j = column index.
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=target.device),
        torch.arange(S, device=target.device),
        indexing="ij",
    )
    # Flatten these grids so that we have one value per cell.
    grid_x = grid_x.reshape(-1).float()  # Shape: [S*S]
    grid_y = grid_y.reshape(-1).float()  # Shape: [S*S]

    # Flatten the target tensor: each grid cell becomes a row.
    target_flat = target.reshape(-1, 5 + num_classes)  # Shape: [S*S, 5 + num_classes]

    # Create a mask to select cells where an object is present.
    obj_mask = target_flat[:, 4] > 0
    valid_targets = target_flat[obj_mask]
    valid_grid_x = grid_x[obj_mask]
    valid_grid_y = grid_y[obj_mask]

    # Extract the offset predictions, width and height from valid cells.
    x_offset = valid_targets[:, 0]
    y_offset = valid_targets[:, 1]
    w = valid_targets[:, 2]
    h = valid_targets[:, 3]

    # Calculate absolute center coordinates:
    # Add the cell coordinate to the relative offset and scale by image size.
    x_center = (valid_grid_x + x_offset) / S * image_size
    y_center = (valid_grid_y + y_offset) / S * image_size

    # Calculate absolute width and height.
    w_abs = w * image_size
    h_abs = h * image_size

    # Derive the top-left and bottom-right corners.
    x_min = x_center - w_abs / 2
    y_min = y_center - h_abs / 2
    x_max = x_center + w_abs / 2
    y_max = y_center + h_abs / 2

    # For the class, we assume the target provides a one-hot vector from index 5 onward.
    # Compute the index of the maximal value to get the class ID.
    class_id = torch.argmax(valid_targets[:, 5:], dim=1).float()

    # Stack the results into a tensor of shape [N, 5].
    boxes = torch.stack([x_min, y_min, x_max, y_max, class_id], dim=1)
    return boxes


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
