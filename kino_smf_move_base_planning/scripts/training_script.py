import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths,
        robot_size,
        start_orientation,
        goal_orientation,
        transform=None,
    ):
        self.image_paths = image_paths
        self.robot_size = robot_size
        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return image, robot_size, start_orientation, goal_orientation


# Define ASPP module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3_dilated = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=2, dilation=2
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv3_dilated(x)
        x4 = self.global_avg_pool(x)
        x4 = self.conv1x1(x4)
        x4 = F.interpolate(x4, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.final_conv(x)
        return x


class AtrousConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(AtrousConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,
        )

    def forward(self, x):
        return self.conv(x)


# Define the main model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutions for image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers for double values
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)

        # Further convolutional layers
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.aspp = ASPP(128, 128)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, image, robot_size, start_orientation, goal_orientation):
        # Image convolutions
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))

        # Double values fully connected layers
        values = torch.cat((robot_size, start_orientation, goal_orientation), dim=1)
        values = F.relu(self.fc1(values))
        values = F.relu(self.fc2(values))
        values = values.view(values.size(0), 128, 1, 1)
        values = values.expand(-1, -1, x.size(2), x.size(3))

        # Concatenate and process further
        x = torch.cat((x, values), dim=1)
        x1 = F.relu(self.conv3(x))
        x2 = self.aspp(x1)
        x2 = self.upsample(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.upsample(x)
        x = self.final_conv(x)

        return torch.sigmoid(x)


# Hyperparameters and dataset setup
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]  # Replace with actual paths
robot_sizes = [0.5, 0.7]
start_orientations = [0.0, 0.5]
goal_orientations = [1.0, 0.3]

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset = CustomDataset(
    image_paths, robot_sizes, start_orientations, goal_orientations, transform
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, robot_sizes, start_orientations, goal_orientations in dataloader:
        images = images.to(device)
        robot_sizes = robot_sizes.to(device)
        start_orientations = start_orientations.to(device)
        goal_orientations = goal_orientations.to(device)

        outputs = model(images, robot_sizes, start_orientations, goal_orientations)
        loss = criterion(
            outputs, images
        )  # Assuming the target is the same as input for simplicity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
