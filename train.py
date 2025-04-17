import torch

from torch.utils.data import DataLoader
from dataset import YOLODataset
from torchvision import transforms, datasets

from loss import YOLOLoss
from model import YOLO

# Define transformations (resize and normalize images)
image_transforms = transforms.Compose(
    [
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),  # ImageNet stats
    ]
)

model = YOLO(num_classes=20)  # Change num_classes as needed
criterion = YOLOLoss(S=13, B=1, num_classes=20)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

# Example usage
dataset = YOLODataset(
    images=["path/to/image1.jpg", "path/to/image2.jpg"],
    annotations=[
        torch.tensor([[0.5, 0.5, 0.2, 0.3, 0], [0.2, 0.8, 0.1, 0.1, 1]]),
        torch.tensor([[0.3, 0.4, 0.1, 0.2, 2]]),
    ],
    image_size=(640, 640),
    S=20,
    num_classes=3,
    transform=image_transforms,  # Define transforms like resizing and normalization
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Preprocessed targets directly from the DataLoader
    for images, targets in dataloader:
        images = images.to("cuda" if torch.cuda.is_available() else "cpu")
        targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

        # Forward pass
        predictions = model(images)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}"
    )
