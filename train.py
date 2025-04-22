from pathlib import Path
import torch

from torch.utils.data import DataLoader
from dataset import YOLODataset
from torchvision import transforms, datasets

from loss import YOLOLoss
from model import YOLO
from val import (
    convert_predictions_to_boxes,
    convert_predictions_to_boxes_vectorized,
    convert_targets_to_boxes,
    validate_model,
)
from viz import visualize_boxes

image_transforms = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # ImageNet stats
    ]
)

plot_transforms = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
)

model = YOLO(num_classes=16)  # Change num_classes as needed
criterion = YOLOLoss(S=20, B=1, num_classes=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if torch.cuda.is_available():
    model = model.to("cuda")
    criterion = criterion.to("cuda")
else:
    print("CUDA is not available. Training on CPU.")

num_epochs = 50

dataset = YOLODataset(
    image_dir=Path("/home/sayandip/projects/pylang/ai/det/master_data/images/train"),
    annotation_dir=Path(
        "/home/sayandip/projects/pylang/ai/det/master_data/labels/train/"
    ),
    image_size=(640, 640),
    S=20,
    num_classes=16,
    transform=image_transforms,  # Define transforms like resizing and normalization
    subset=160,
)

val_dataset = YOLODataset(
    image_dir=Path("/home/sayandip/projects/pylang/ai/det/master_data/images/train"),
    annotation_dir=Path(
        "/home/sayandip/projects/pylang/ai/det/master_data/labels/train/"
    ),
    image_size=(640, 640),
    S=20,
    num_classes=16,
    transform=image_transforms,  # Define transforms like resizing and normalization
    # transform=image_transforms,  # Define transforms like resizing and normalization
    subset=32,
)
plot_dataset = YOLODataset(
    image_dir=Path("/home/sayandip/projects/pylang/ai/det/master_data/images/train"),
    annotation_dir=Path(
        "/home/sayandip/projects/pylang/ai/det/master_data/labels/train/"
    ),
    image_size=(640, 640),
    S=20,
    num_classes=16,
    transform=plot_transforms,  # Define transforms like resizing and normalization
    # transform=image_transforms,  # Define transforms like resizing and normalization
    subset=32,
)


dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
plot_dataloader = DataLoader(plot_dataset, batch_size=4, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

        predictions = model(images)

        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}"
    )
    # if epoch == 4:
    #     breakpoint()

    model.eval()
    metrics = validate_model(
        model, val_dataloader, conf_threshold=0.5, iou_threshold=0.5
    )
    print(metrics)

model.eval()
with torch.no_grad():
    for (plot_images, _), (images, targets) in zip(plot_dataloader, val_dataloader):
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")
        predictions = model(images)
        for image, target, pred in zip(plot_images, targets, predictions):
            pred_boxes = convert_predictions_to_boxes_vectorized(
                pred.unsqueeze(0), conf_threshold=0.5
            )
            target_boxes = convert_targets_to_boxes(target)
            visualize_boxes(image, target_boxes, pred_boxes)
