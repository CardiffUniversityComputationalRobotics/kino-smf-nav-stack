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
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.atrous_blocks = nn.ModuleList()

        # 1x1 convolution
        self.atrous_blocks.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        # Atrous convolutions with different rates
        for rate in atrous_rates:
            self.atrous_blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=rate,
                    dilation=rate,
                )
            )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_global = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        # Final convolution to aggregate all the feature maps
        self.conv1x1_output = nn.Conv2d(
            out_channels * (len(atrous_rates) + 2),
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        size = x.shape[2:]  # Height and Width of the input feature map
        outputs = []

        # Apply all the convolutions in the atrous blocks
        for block in self.atrous_blocks:
            outputs.append(block(x))

        # Apply global average pooling
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = self.conv1x1_global(global_avg_pool)
        global_avg_pool = F.interpolate(
            global_avg_pool, size=size, mode="bilinear", align_corners=False
        )
        outputs.append(global_avg_pool)

        # Concatenate all the outputs
        x = torch.cat(outputs, dim=1)

        # Final 1x1 convolution to aggregate all the feature maps
        x = self.conv1x1_output(x)

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


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Assuming input attributes are of size 3
        self.fc2 = nn.Linear(3, 64)

    def forward(self, attributes):
        fla = self.fc1(attributes)
        fla = torch.relu(fla)
        fla = fla.view(-1, 1, 1, 32)  # Reshape to (batch_size, 1, 1, 32)

        fha = self.fc2(attributes)
        fha = torch.relu(fha)
        fha = fha.view(-1, 1, 1, 64)  # Reshape to (batch_size, 1, 1, 64)

        return fla, fha


# Define the main model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutions for image
        self.conv1 = AtrousConvLayer(3, 64, kernel_size=3, padding=1, dilation=5)
        self.conv2 = AtrousConvLayer(64, 128, kernel_size=3, padding=1, dilation=5)

        # Fully connected layers for double values
        self.fcn = FullyConnectedNN()

        # Further convolutional layers
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.aspp = ASPP(128, 128)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, image, robot_size, start_orientation, goal_orientation):

        # ! FIRST PROCESS

        # Individual attributes
        path_attributes = torch.tensor(
            [[robot_size, start_orientation, goal_orientation]], dtype=torch.float32
        )

        fla, fha = self.fcn(path_attributes)
        fla = F.relu(fla)
        fha = F.relu(fha)

        # input image convolutions

        l_conv = F.relu(self.conv1(image))
        h_conv = F.relu(self.conv2(l_conv))

        # ! CONCATENATIONS

        low_concat = torch.cat((l_conv, fla), dim=1)
        high_concat = torch.cat((h_conv, fha), dim=1)

        # ! high res section second convolution

        h_conv = F.relu(self.conv2(high_concat))

        # ! ASPP AND UP SAMPLING OF HIGH RES

        h_aspp = self.aspp(h_conv)
        h_upsample = self.upsample(h_aspp)

        # ! FINAL CONCATENATION
        final_concat = torch.cat(low_concat, h_upsample)

        x = self.upsample(final_concat)
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
