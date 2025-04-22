import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_boxes(image, ground_truths=None, predictions=None, image_size=640):
    """
    Visualize ground truth and predicted bounding boxes on an image.

    Parameters:
        image: The input image as a tensor (shape [3, H, W]) or NumPy array (shape [H, W, 3]).
        ground_truths: Ground truth bounding boxes, tensor of shape [num_boxes, 5].
                       Format: [x_min, y_min, x_max, y_max, class_id].
        predictions: Predicted bounding boxes, tensor of shape [num_boxes, 6].
                     Format: [x_min, y_min, x_max, y_max, confidence, class_id].
        image_size: Image size (e.g., 640 for a 640x640 image).
    """
    # Convert tensor to NumPy array if needed
    # breakpoint()
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert [3, H, W] to [H, W, 3]

    # Rescale image to [0, 1] if needed
    if image.max() > 1:
        image = image / 255.0

    # Create the plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Plot ground truth boxes
    if ground_truths is not None:
        for box in ground_truths:
            x_min, y_min, x_max, y_max, class_id = box.tolist()
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="green",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 10,
                f"GT Class {int(class_id)}",
                color="green",
                fontsize=12,
            )
    # Plot predicted boxes
    if predictions is not None:
        for box in predictions:
            x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x_min,
                y_min - 10,
                f"Pred Class {int(class_id)} ({confidence:.2f})",
                color="red",
                fontsize=12,
            )

    plt.axis("off")
    plt.show()
