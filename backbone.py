import torch
import torch.nn as nn


# Define a convolutional block that includes Convolution -> BatchNorm -> LeakyReLU activation.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leaky(out)
        return out


# Define the YOLO Backbone network
class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()

        # Layer 1: Initial convolution and downsampling (using MaxPool for spatial reduction)
        self.layer1 = nn.Sequential(
            ConvBlock(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial dimensions by 2
        )

        # Layer 2: Increase channels and downsample further
        self.layer2 = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 3: Introduce some block repetition to learn richer features
        self.layer3 = nn.Sequential(
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 4: Further increase depth
        self.layer4 = nn.Sequential(
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 5: Heavier feature extraction stage with deeper layers
        self.layer5 = nn.Sequential(
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Layer 6: Final stage to produce high-level feature maps
        # The spatial dimensions remain the same as the previous layer,
        # but the depth (number of channels) increases.
        self.layer6 = nn.Sequential(
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
            ConvBlock(
                in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0
            ),
            ConvBlock(
                in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, x):
        x = self.layer1(x)  # Output: (batch_size, 32, H/2, W/2)
        x = self.layer2(x)  # Output: (batch_size, 64, H/4, W/4)
        x = self.layer3(x)  # Output: (batch_size, 128, H/8, W/8)
        x = self.layer4(x)  # Output: (batch_size, 256, H/16, W/16)
        x = self.layer5(x)  # Output: (batch_size, 512, H/32, W/32)
        x = self.layer6(x)  # Output: (batch_size, 1024, H/32, W/32)
        return x


# Testing the backbone with dummy data
if __name__ == "__main__":
    # Create a dummy input tensor simulating a batch of one 416x416 RGB image.
    dummy_input = torch.randn(1, 3, 416, 416)

    # Instantiate the backbone model
    model = YOLOBackbone()

    # Forward pass through the backbone network
    features = model(dummy_input)
    print("Output feature map shape:", features.shape)
