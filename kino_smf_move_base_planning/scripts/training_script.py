import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.models as models
import os
import cv2


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths,
        pdf_maps,
        robot_size,
        start_orientation,
        goal_orientation,
        transform=None,
    ):
        self.image_paths = image_paths
        self.pdf_maps = pdf_maps
        self.robot_size = robot_size
        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        pdf_image = Image.fromarray(self.pdf_maps[idx])
        if self.transform:
            image = self.transform(image)
            pdf_image = self.transform(pdf_image)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return image, pdf_image, robot_size, start_orientation, goal_orientation


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


class AtrousResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(AtrousResNet50, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # Modify layer3 to use atrous convolutions with dilation rate 2
        self.layer3 = self._make_atrous_layer(resnet.layer3, dilation=2)

        # Modify layer4 to use atrous convolutions with dilation rate 4
        self.layer4 = self._make_atrous_layer(resnet.layer4, dilation=4)

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def _make_atrous_layer(self, layer, dilation):
        for n, m in layer.named_modules():
            if "conv2" in n:
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)
        return layer

    def forward(self, x):
        x = self.layer0(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c4


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Assuming input attributes are of size 3
        self.fc2 = nn.Linear(3, 64)

    def forward(self, attributes):

        attributes = attributes.view(attributes.size(0), -1)
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
        self.init_conv = AtrousResNet50(pretrained=True)

        # Fully connected layers for double values
        self.fcn = FullyConnectedNN()

        # Further convolutional layers
        self.conv = nn.Conv2d(
            288, 128, kernel_size=3, padding=1
        )  # Adjusted in_channels
        self.aspp = ASPP(2112, 128, [3, 6, 12])  # Adjusted in_channels
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # Ensure the final output has 1 channel
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, image, robot_size, start_orientation, goal_orientation):
        # Combine individual attributes into a single tensor
        path_attributes = torch.stack(
            [robot_size, start_orientation, goal_orientation], dim=1
        )

        # Pass attributes through the fully connected network
        fla, fha = self.fcn(path_attributes)

        # input image convolutions
        c1, c4 = self.init_conv(image)

        # Concatenations
        fla = fla.expand(
            -1, c1.size(2), c1.size(3), -1
        )  # Adjust fla to match spatial dimensions
        low_concat = torch.cat(
            (c1, fla.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        fha = fha.expand(
            -1, c4.size(2), c4.size(3), -1
        )  # Adjust fha to match spatial dimensions
        high_concat = torch.cat(
            (c4, fha.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        # Low res section second convolution
        fcl = F.relu(self.conv(low_concat))

        # ASPP and upsampling of high res
        fh = self.aspp(high_concat)
        fh = self.upsample(fh)

        # Ensure dimensions match for concatenation
        fcl_resized = F.interpolate(
            fcl, size=(fh.size(2), fh.size(3)), mode="bilinear", align_corners=False
        )

        # Final concatenation
        final_concat = torch.cat((fcl_resized, fh), dim=1)

        # Apply final convolution to get the output with the desired number of channels
        x = self.final_conv(final_concat)

        # Ensure output size is [batch_size, 1, 128, 128]
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)

        return torch.sigmoid(x)


def list_files_in_directory(directory):

    filename_list = []

    try:
        # List all files and directories in the specified directory
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    filename_list.append(entry.name)
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")

    return filename_list


def get_images_and_params(directory):

    filenames = list_files_in_directory(directory)

    complete_file_paths = []
    start_orientation = []
    goal_orientation = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)
        split_name = i.split("_")
        start_orientation.append(float(round(float(split_name[2]), 3)))
        goal_orientation.append(float(round(float(split_name[-1][:-5]), 3)))

    return complete_file_paths, start_orientation, goal_orientation


def create_probability_map(image_size, path_points, sigma=1.0):
    """
    Creates a probability map given a list of path points.

    :param image_size: Tuple (H, W) indicating the size of the map.
    :param path_points: List of (x, y) tuples indicating the path.
    :param sigma: Standard deviation for the Gaussian blur.
    :return: Probability map as a numpy array of shape (H, W).
    """
    probability_map = np.zeros(image_size, dtype=np.float32)

    # Draw the path on the map
    for point in path_points:
        cv2.circle(probability_map, point, radius=1, color=1.0, thickness=-1)

    # Apply Gaussian blur to create a smooth probability distribution
    probability_map = cv2.GaussianBlur(probability_map, (0, 0), sigma)
    probability_map = np.clip(probability_map, 0, 1)

    return probability_map


def get_pdf_images(directory):

    filenames = list_files_in_directory(directory)

    complete_file_paths = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)

    pdf_maps = []

    for i in complete_file_paths:
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 5.25 > image[i][j] > 4.0:
                    image[i, j] = 100.0
                else:
                    image[i, j] = 0.0

        probability_map = cv2.GaussianBlur(image, (0, 0), 1.0)
        probability_map = np.clip(probability_map, 0, 1)

        pdf_maps.append(probability_map)

    return pdf_maps


def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for (
            inputs,
            labels,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in train_loader:
            inputs, labels, robot_size, start_orientations, goal_orientations = (
                inputs.to(device),
                labels.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            optimizer.zero_grad()
            outputs = model(inputs, robot_size, start_orientations, goal_orientations)
            print(outputs.size())
            print(labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

    print("Training complete")
    return model


# Hyperparameters and dataset setup

raw_image_path = "/home/sasm/Downloads/dataset/test"
path_image_path = "/home/sasm/Downloads/dataset/train"

image_paths, start_orientations, goal_orientations = get_images_and_params(
    raw_image_path
)

start_orientations = np.array(start_orientations).astype(np.float32)
goal_orientations = np.array(goal_orientations).astype(np.float32)
robot_sizes = np.array([0.35] * len(image_paths)).astype(np.float32)

pdf_maps = get_pdf_images(path_image_path)


transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset = CustomDataset(
    image_paths, pdf_maps, robot_sizes, start_orientations, goal_orientations, transform
)


train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)
