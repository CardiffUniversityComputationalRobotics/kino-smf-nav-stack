import torch
import torch.nn as nn
import torch.nn.functional as F

from model_primitives import *


# Define the main model
class XYCNN(nn.Module):
    def __init__(self):
        super(XYCNN, self).__init__()

        # Convolutions for image
        # self.init_conv = ResNet50Encoder()
        self.init_conv = AtrousResNet50()

        # Fully connected layers for double values
        self.fcn = FullyConnectedNN()

        # Further convolutional layers
        self.conv = nn.Conv2d(
            288, 256 + 32, kernel_size=3, padding=1
        )  # Adjusted in_channels
        self.aspp = ASPP(2112, 256 + 64, [3, 6, 12])  # Adjusted in_channels
        self.upsample = nn.Upsample(size=(25, 25), mode="bilinear", align_corners=False)

        # Ensure the final output has 1 channel
        self.final_conv = nn.ConvTranspose2d(
            in_channels=608, out_channels=1, kernel_size=6, stride=4, padding=1
        )

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

        # Final concatenation
        final_concat = torch.cat((fcl, fh), dim=1)

        x = self.final_conv(final_concat)

        xy_inference = torch.sigmoid(x)

        return xy_inference


# Define the main model
class YawCNN(nn.Module):
    def __init__(self):
        super(YawCNN, self).__init__()

        # Convolutions for image
        # self.init_conv = ResNet50Encoder()
        self.init_conv = AtrousResNet50()

        # Fully connected layers for double values
        self.fcn = FullyConnectedNN()

        # Further convolutional layers
        self.conv = nn.Conv2d(
            288, 256 + 32, kernel_size=3, padding=1
        )  # Adjusted in_channels
        self.aspp = ASPP(2112, 256 + 64, [3, 6, 12])  # Adjusted in_channels
        self.upsample = nn.Upsample(size=(25, 25), mode="bilinear", align_corners=False)

        # Ensure the final output has 1 channel
        self.final_conv = nn.ConvTranspose2d(
            in_channels=608, out_channels=1, kernel_size=6, stride=4, padding=1
        )

    def forward(self, image, robot_size, start_orientation, goal_orientation):
        # Combine individual attributes into a single tensor
        path_attributes = torch.stack(
            [robot_size, start_orientation, goal_orientation], dim=1
        )

        # Pass attributes through the fully connected network
        fla, fha = self.fcn(path_attributes)

        # print(image.size())
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

        # Final concatenation
        final_concat = torch.cat((fcl, fh), dim=1)

        # Apply final convolution to get the output with the desired number of channels
        x = self.final_conv(final_concat)

        yaw_inference = F.relu(x)

        return yaw_inference


# Define the main model
class SingleYawCNN(nn.Module):
    def __init__(self):
        super(SingleYawCNN, self).__init__()

        # Convolutions for image
        # self.init_conv = ResNet50Encoder()
        self.init_conv = AtrousResNet50()

        # Fully connected layers for double values
        self.fcn = SingleFullyConnectedNN()

        # Further convolutional layers
        self.conv = nn.Conv2d(
            288, 256 + 32, kernel_size=3, padding=1
        )  # Adjusted in_channels
        self.aspp = ASPP(2112, 256 + 64, [3, 6, 12])  # Adjusted in_channels

        self.max_pool_2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.final_layer = nn.Linear(608, 1)

    def forward(
        self, image, robot_size, start_orientation, goal_orientation, x_pose, y_pose
    ):
        # Combine individual attributes into a single tensor
        path_attributes = torch.stack(
            [robot_size, start_orientation, goal_orientation, x_pose, y_pose], dim=1
        )

        # Pass attributes through the fully connected network
        fla, fha = self.fcn(path_attributes)

        # print(image.size())
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

        fcl = self.max_pool_2d(fcl)

        final_concat = torch.cat((fh, fcl), dim=1)

        flat_vector = self.flatten(final_concat)

        yaw_inference = self.final_layer(flat_vector)

        return yaw_inference
