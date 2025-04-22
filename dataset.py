from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def normalize_bounding_boxes(bboxes, image_width, image_height):
    """
    Normalizes bounding box coordinates relative to the image dimensions.

    Parameters:
        bboxes: Tensor of shape [num_boxes, 5], where each row is [x, y, w, h, class_id].
        image_width: Width of the input image.
        image_height: Height of the input image.

    Returns:
        Normalized bounding box tensor of shape [num_boxes, 5].
    """
    # normalized_bboxes = bboxes.clone()
    # normalized_bboxes[:, 0] = bboxes[:, 0] / image_width  # x_center
    # normalized_bboxes[:, 1] = bboxes[:, 1] / image_height  # y_center
    # normalized_bboxes[:, 2] = bboxes[:, 2] / image_width  # width
    # normalized_bboxes[:, 3] = bboxes[:, 3] / image_height  # height
    # return normalized_bboxes
    return bboxes


def assign_to_grid(bboxes, S, num_classes):
    """
    Assign bounding boxes to grid cells.

    Parameters:
        bboxes: Tensor of shape [num_boxes, 5], where each row is [x, y, w, h, class_id].
                Coordinates are normalized (between 0 and 1).
        S: Grid size (e.g., 20 for a 20x20 grid).
        num_classes: Number of object classes.

    Returns:
        Ground-truth tensor of shape [S, S, 5 + num_classes].
    """
    grid = torch.zeros((S, S, 5 + num_classes))  # Initialize grid
    for bbox in bboxes:
        x, y, w, h, class_id = bbox.tolist()

        # Calculate grid cell indices
        grid_x = int(x * S)
        grid_y = int(y * S)

        # Assign bounding box attributes to the grid cell
        grid[grid_y, grid_x, :4] = torch.tensor(
            [x * S - grid_x, y * S - grid_y, w, h]
        )  # [x_offset, y_offset, w, h]
        grid[grid_y, grid_x, 4] = 1.0  # Object confidence
        grid[grid_y, grid_x, 5 + int(class_id)] = 1.0  # One-hot class label

    return grid


class YOLODataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_dir,
        image_size,
        S,
        num_classes,
        transform=None,
        subset: int | None = None,
    ):
        """
        YOLO Dataset for handling images with optional annotations.

        Parameters:
            image_dir: Directory containing image files.
            annotation_dir: Directory containing annotation text files.
            image_size: Tuple (width, height) for resizing images (e.g., (640, 640)).
            S: Grid size (e.g., 20 for a 20x20 grid).
            num_classes: Number of object classes.
            transform: Optional torchvision transforms for data augmentation and normalization.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.S = S
        self.num_classes = num_classes
        self.transform = transform
        paths = (
            Path(image_dir).iterdir()
            if subset is None
            else islice(Path(image_dir).iterdir(), subset)
        )
        self.image_files = [
            f.name
            for f in paths
            if f.suffix.lower() in (".tiff", ".tif", ".jpg", ".png")
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = Path(self.image_dir) / image_file

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = image.resize(self.image_size)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        annotation_file = Path(image_file).stem + ".txt"
        annotation_path = (
            Path(self.annotation_dir) / Path(image_file).with_suffix(".txt").name
        )

        if annotation_path.exists():
            bboxes = self._load_annotations(annotation_path)
            bboxes = normalize_bounding_boxes(bboxes, *self.image_size)
            target = assign_to_grid(bboxes, self.S, self.num_classes)
        else:
            target = torch.zeros((self.S, self.S, 5 + self.num_classes))

        return image, target

    def _load_annotations(self, annotation_path):
        """
        Load bounding box annotations from a text file.

        Parameters:
            annotation_path: Path to the annotation file.

        Returns:
            Tensor of shape [num_boxes, 5] -> [x_center, y_center, width, height, class_id].
        """
        boxes = []
        with annotation_path.open("r") as f:
            for line in f:
                cls, *box = list(map(float, line.strip().split()))
                boxes.append([*box, cls])
        return torch.tensor(boxes)
