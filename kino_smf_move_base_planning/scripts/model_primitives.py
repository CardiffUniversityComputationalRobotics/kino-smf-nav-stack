import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision
import math

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

        attributes = attributes.view(attributes.size(0), -1)

        fla = self.fc1(attributes)
        fla = torch.relu(fla)
        fla = fla.view(-1, 1, 1, 32)  # Reshape to (batch_size, 1, 1, 32)

        fha = self.fc2(attributes)
        fha = torch.relu(fha)
        fha = fha.view(-1, 1, 1, 64)  # Reshape to (batch_size, 1, 1, 64)

        return fla, fha


class SingleFullyConnectedNN(nn.Module):
    def __init__(self):
        super(SingleFullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)  # Assuming input attributes are of size 3
        self.fc2 = nn.Linear(5, 64)

    def forward(self, attributes):

        attributes = attributes.view(attributes.size(0), -1)

        fla = self.fc1(attributes)
        fla = torch.relu(fla)
        fla = fla.view(-1, 1, 1, 32)  # Reshape to (batch_size, 1, 1, 32)

        fha = self.fc2(attributes)
        fha = torch.relu(fha)
        fha = fha.view(-1, 1, 1, 64)  # Reshape to (batch_size, 1, 1, 64)

        return fla, fha

class MultiLayerFCN(nn.Module):
    def __init__(self):
        super(MultiLayerFCN, self).__init__()
        self.fc1 = nn.Linear(9728, 512)  # First layer
        self.fc2 = nn.Linear(512, 128)   # Second layer
        self.fc3 = nn.Linear(128, 1)     # Final layer producing a single output

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation after first layer
        x = torch.relu(self.fc2(x))  # ReLU activation after second layer
        x = self.fc3(x)              # No activation for the final layer
        return x
    
class MaskedCircularLoss(nn.Module):
    def __init__(self):
        super(MaskedCircularLoss, self).__init__()

    def forward(self, predictions, targets, mask):
        diff = (predictions - targets + math.pi) % (2 * math.pi) - math.pi
        masked_diff = diff * mask
        return torch.mean(masked_diff ** 2)