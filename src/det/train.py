from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
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

    print(f"{running_loss / len(val_dataloader)}")

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

        metrics = validate_model(model, val_dataloader, device)


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

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            detailed_loss = loss.compute_loss(predictions, images)
            running_val_loss_detailed += detailed_loss
            running_val_loss += detailed_loss.sum().item()

        val_loss = running_val_loss / len(dataloader)
