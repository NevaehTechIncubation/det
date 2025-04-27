from collections.abc import Callable
from typing import override
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def normalize_annotations(
    annotations: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    return annotations / torch.tensor([width, height, width, height, 1.0])


def denormalize_annotations(
    normalized_annotations: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    return normalized_annotations * torch.tensor([width, height, width, height, 1.0])


def split_into_grids(
    annotation: torch.Tensor,
    grid_size: int,
    num_classes: int,
) -> torch.Tensor:
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
    grids = torch.zeros((grid_size, grid_size, 5 + num_classes))
    grids_xy = (annotation[:, :2] * grid_size).int()
    col, row = grids_xy[:, 0], grids_xy[:, 1]
    grids[row, col, :2] = (annotation[:, :2] * grid_size - grids_xy).float()
    grids[row, col, 2:4] = annotation[:, 2:4].float()
    grids[row, col, 4] = 1.0
    grids[row, col, annotation[:, 4].int() + 5] = 1.0
    return grids


class YOLODataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int,
        grid_size: int,
        num_classes: int,
        transform: Callable[..., torch.Tensor] | None = None,
        _subset: slice | None = None,
    ) -> None:
        assert images_dir.is_dir() and labels_dir.is_dir()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.grid_size = grid_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = (
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
            if transform is None
            else transform
        )
        image_files = [
            f
            for f in self.images_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
        self.image_files = image_files if _subset is None else image_files[_subset]

    def __len__(self) -> int:
        return len(self.image_files)

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, annotation = self.getitem_raw(index)

        image = self.transform(img)
        target = (
            split_into_grids(annotation, self.grid_size, self.num_classes)
            if annotation is not None
            else torch.zeros((self.grid_size, self.grid_size, (5 + self.num_classes)))
        )
        return image, target

    def getitem_raw(
        self,
        index: int,
        resize: int | None = None,
    ) -> tuple[Image.Image, torch.Tensor | None]:
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")

        annotation_file = self.labels_dir / image_file.with_suffix(".txt").name
        annotation = (
            self._load_annotation(annotation_file) if annotation_file.exists() else None
        )
        return image, annotation

    def _load_annotation(self, annotation_file: Path):
        return (
            torch
            .from_numpy(np.loadtxt(annotation_file).reshape(-1, 5))
            .roll(-1, dims=1)  # move class label from first col to last
        )  # fmt: skip


def create_dataloader(
    dataset_dir: Path,
    num_classes: int,
    image_size: int = 640,
    batch_size: int = 16,
    grid_size: int = 20,
    transform: Callable[..., torch.Tensor] | None = None,
    num_workers: int = 0,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not (train_dir := images_dir / "train").is_dir():
        raise NotADirectoryError(str(train_dir))

    if (images_dir / "val").is_dir():
        train_subdir, val_subdir = "train", "val"
        train_subset, val_subset = None, None
    else:
        train_subdir, val_subdir = "train", "train"
        train_subset, val_subset = slice(0, 160), slice(-32, None)

    train_dataloader = DataLoader(
        YOLODataset(
            images_dir / train_subdir,
            labels_dir / train_subdir,
            image_size,
            grid_size,
            num_classes,
            transform,
            _subset=train_subset,
        ),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        YOLODataset(
            images_dir / val_subdir,
            labels_dir / val_subdir,
            image_size,
            grid_size,
            num_classes,
            transform,
            _subset=val_subset,
        ),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader
