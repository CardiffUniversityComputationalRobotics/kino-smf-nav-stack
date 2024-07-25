import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision


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


class ResNet50Encoder(nn.Module):
    def __init__(
        self,
        freeze_original_weights=True,
    ):
        super(ResNet50Encoder, self).__init__()
        resnet = torchvision.models.resnet.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )
        if freeze_original_weights:
            for child in resnet.children():
                for param in child.parameters():
                    param.requires_grad = False
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        """
        inputs:
            - x: tensor. (batch_size, 3, H, W)
        outputs:
            - features: list of 6 tensors.
                x: tensor. (batch_size, 3, H, W)
                torch.Size([batch_size, 64, H/2, W/2])
                torch.Size([batch_size, 256, H/4, W/4])
                torch.Size([batch_size, 512, H/8, W/8])
                torch.Size([batch_size, 1024, H/16, W/16])
                torch.Size([batch_size, 2048, H/32, W/32])
        """
        features = [x]
        x = self.input_block(x)
        features.append(x)
        x = self.input_pool(x)
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        return features[2], features[-1]


class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Assuming input attributes are of size 3
        self.fc2 = nn.Linear(3, 64)

    def forward(self, attributes):
        # print("=============")
        # print(attributes.size())
        # print(attributes)
        attributes = attributes.view(attributes.size(0), -1)
        # print(attributes)
        # print(attributes.size())
        # print("++++++++++==")

        fla = self.fc1(attributes)
        fla = torch.relu(fla)
        fla = fla.view(-1, 1, 1, 32)  # Reshape to (batch_size, 1, 1, 32)

        fha = self.fc2(attributes)
        fha = torch.relu(fha)
        fha = fha.view(-1, 1, 1, 64)  # Reshape to (batch_size, 1, 1, 64)

        return fla, fha


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

        # print(image.size())
        # input image convolutions
        c1, c4 = self.init_conv(image)

        # print("==========")
        # print(c1.size())
        # print(c4.size())

        # Concatenations
        fla = fla.expand(
            -1, c1.size(2), c1.size(3), -1
        )  # Adjust fla to match spatial dimensions

        # print(fla.size())
        # print(fla)

        low_concat = torch.cat(
            (c1, fla.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        fha = fha.expand(
            -1, c4.size(2), c4.size(3), -1
        )  # Adjust fha to match spatial dimensions
        high_concat = torch.cat(
            (c4, fha.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        # print(fla.permute(0, 3, 1, 2).size())
        # print(fha.permute(0, 3, 1, 2).size())

        # print(low_concat.size())
        # print(high_concat.size())

        # Low res section second convolution
        fcl = F.relu(self.conv(low_concat))

        # ASPP and upsampling of high res
        fh = self.aspp(high_concat)

        fh = self.upsample(fh)

        # print(fcl.size())
        # print(fh.size())

        # Ensure dimensions match for concatenation
        # fcl_resized = F.interpolate(
        #     fcl, size=(fh.size(2), fh.size(3)), mode="bilinear", align_corners=False
        # )

        # Final concatenation
        final_concat = torch.cat((fcl, fh), dim=1)
        # print(final_concat.size())
        # Apply final convolution to get the output with the desired number of channels
        x = self.final_conv(final_concat)
        # print(x.size())
        # Ensure output size is [batch_size, 1, 128, 128]
        # x = F.interpolate(x, size=(100, 100), mode="bilinear", align_corners=False)

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

        # print("==========")
        # print(c1.size())
        # print(c4.size())

        # Concatenations
        fla = fla.expand(
            -1, c1.size(2), c1.size(3), -1
        )  # Adjust fla to match spatial dimensions

        # print(fla.size())
        # print(fla)

        low_concat = torch.cat(
            (c1, fla.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        fha = fha.expand(
            -1, c4.size(2), c4.size(3), -1
        )  # Adjust fha to match spatial dimensions
        high_concat = torch.cat(
            (c4, fha.permute(0, 3, 1, 2)), dim=1
        )  # Concatenate along channel dimension

        # print(fla.permute(0, 3, 1, 2).size())
        # print(fha.permute(0, 3, 1, 2).size())

        # print(low_concat.size())
        # print(high_concat.size())

        # Low res section second convolution
        fcl = F.relu(self.conv(low_concat))

        # ASPP and upsampling of high res
        fh = self.aspp(high_concat)

        fh = self.upsample(fh)

        # print(fcl.size())
        # print(fh.size())

        # Ensure dimensions match for concatenation
        # fcl_resized = F.interpolate(
        #     fcl, size=(fh.size(2), fh.size(3)), mode="bilinear", align_corners=False
        # )

        # Final concatenation
        final_concat = torch.cat((fcl, fh), dim=1)
        # print(final_concat.size())
        # Apply final convolution to get the output with the desired number of channels
        x = self.final_conv(final_concat)
        # print(x.size())
        # Ensure output size is [batch_size, 1, 128, 128]
        # x = F.interpolate(x, size=(100, 100), mode="bilinear", align_corners=False)

        yaw_inference = F.relu(x)

        return yaw_inference
