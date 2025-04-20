from pathlib import Path
import torch

from torch.utils.data import DataLoader
from dataset import YOLODataset
from torchvision import transforms, datasets

from loss import YOLOLoss
from model import YOLO

image_transforms = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # ImageNet stats
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

num_epochs = 20

dataset = YOLODataset(
    image_dir=Path("/home/sayandip/projects/pylang/ai/det/master_data/images/train"),
    annotation_dir=Path(
        "/home/sayandip/projects/pylang/ai/det/master_data/labels/train/"
    ),
    image_size=(640, 640),
    S=20,
    num_classes=16,
    transform=image_transforms,  # Define transforms like resizing and normalization
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

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
