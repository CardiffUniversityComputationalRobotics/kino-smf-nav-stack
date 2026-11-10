import math
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
class XYCNNArtifacts(nn.Module):
    def __init__(self):
        super(XYCNNArtifacts, self).__init__()

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
        self.final_conv = nn.Sequential(
            # First ConvTranspose2d layer
            nn.ConvTranspose2d(
                in_channels=608, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),  # ReLU activation after the first deconvolution
            # Second ConvTranspose2d layer
            nn.ConvTranspose2d(
                in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
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


class XYYawDiscreetCNN(nn.Module):
    def __init__(self):
        super(XYYawDiscreetCNN, self).__init__()

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
            in_channels=608, out_channels=100, kernel_size=6, stride=4, padding=1
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
        self.final_conv = nn.Sequential(
            # First ConvTranspose2d layer
            nn.ConvTranspose2d(
                in_channels=608, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),  # ReLU activation after the first deconvolution
            # Second ConvTranspose2d layer
            nn.ConvTranspose2d(
                in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
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

        self.max_pool_2d = nn.MaxPool2d(kernel_size=6, stride=6)

        self.flatten = nn.Flatten()

        self.final_layer = MultiLayerFCN()

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


class XYYawCNN(nn.Module):
    """Combined position + orientation model: one shared trunk, two heads.

    XYCNNArtifacts and YawCNN had byte-identical trunks (AtrousResNet50 +
    FullyConnectedNN + conv + ASPP + upsample) and differed only in the final head, so
    running both duplicated ~49M parameters of encoder work per planning cycle. This
    merges them, matching Fig. 3b of the paper ("similar architectures but differ in the
    final head block, which provides classification and regression inferences").

    Input is the same tensor gridMapToTensor() already builds: [B,3,H,W] whose channel 0
    holds the cell classes (0 free, 1 obstacle, 2 start, 3 goal, 5 social agent) and whose
    other two channels are zero. Those classes are categorical, so feeding them as a
    single intensity implies a false ordering (obstacle < start < goal < agent); the model
    one-hots them internally instead. That is a change of representation only -- the C++
    side keeps passing exactly what it passes today.

    Deployment contract (PathLengthDirectInfSamplerMod.cpp):
      xy  in [0,1]           -- checkXYInference() keeps cells > 0.2
      yaw in [1, 1+2*pi]     -- getSampleYaw() treats < 1 as "no data"
    """

    # Final so torch.jit.script inlines them as constants (the old scripts saved with
    # torch.jit.script, and the C++ side loads a scripted module).
    N_CLASSES: torch.jit.Final[int]
    YAW_OFFSET: torch.jit.Final[float]
    XY_THRESHOLD: torch.jit.Final[float]

    def __init__(self, pretrained=True, head="simple"):
        super(XYYawCNN, self).__init__()

        self.N_CLASSES = 6
        self.YAW_OFFSET = 1.0      # getSampleYaw(): angle < 1 means no data
        self.XY_THRESHOLD = 0.2    # checkXYInference()

        self.init_conv = AtrousResNet50(pretrained=pretrained)

        # Re-stem for one-hot input. Seed plane k with the pretrained filter for input
        # channel 0 SCALED BY k -- channel 0 is the one the scalar path actually multiplies,
        # since gridMapToTensor() leaves channels 1-2 zero. Because exactly one plane is hot per pixel, the
        # stem then computes mean_filter * class_value at init -- i.e. bit-for-bit the
        # response the scalar-input stem gives -- so training starts from the known-good
        # operating point and only gains the freedom to learn per-class filters from
        # there. Copying the same filter into every plane instead (the obvious thing)
        # makes the response identical for every class, destroying the class signal at
        # init and forcing the net to break that symmetry from scratch.
        old = self.init_conv.layer0[0]
        new = nn.Conv2d(self.N_CLASSES, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            base = old.weight[:, 0]                           # [out, kh, kw]
            for k in range(self.N_CLASSES):
                new.weight[:, k] = base * float(k)
        self.init_conv.layer0[0] = new

        # 3 scalars: robot size, start yaw, goal yaw. The default input_dim is 10, which
        # is what the expansion models use -- passing 3 here is what the old scripts
        # silently relied on before that default changed.
        self.fcn = FullyConnectedNN(input_dim=3)

        self.conv = nn.Conv2d(288, 256 + 32, kernel_size=3, padding=1)
        self.aspp = ASPP(2112, 256 + 64, [3, 6, 12])
        self.upsample = nn.Upsample(size=(25, 25), mode="bilinear", align_corners=False)

        # Position head -> 1 logit per cell (classification).
        # "simple" is the original XYCNN decoder: ONE transposed conv, 21.9K parameters.
        # "deep" is the two-stage 256-channel decoder (~2.5M params, 115x more). The extra
        # capacity measurably overfits this target -- the label is one random SST draw, so
        # a decoder able to reproduce a specific 1px curve memorises it, while the tiny one
        # can only emit a smooth corridor, which is what actually generalises.
        def make_head(out_ch):
            if head == "simple":
                return nn.ConvTranspose2d(608, out_ch, kernel_size=6, stride=4, padding=1)
            return nn.Sequential(
                nn.ConvTranspose2d(608, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, out_ch, kernel_size=4, stride=2, padding=1),
            )
        self.pos_head = make_head(1)
        # Orientation head -> (cos, sin) per cell (regression). Predicting the angle
        # directly puts a discontinuity at the 0/2pi wrap in the middle of the target
        # range; cos/sin removes it and atan2 recovers the angle.
        self.yaw_head = make_head(2)

    # ------------------------------------------------------------------
    def _trunk(self, image, robot_size, start_orientation, goal_orientation):
        cls = image[:, 0:1].round().clamp(0, self.N_CLASSES - 1)
        planes = []
        for k in range(self.N_CLASSES):
            planes.append((cls - float(k)).abs().lt(0.5).float())
        x = torch.cat(planes, dim=1)

        path_attributes = torch.stack(
            [robot_size, start_orientation, goal_orientation], dim=1
        )
        fla, fha = self.fcn(path_attributes)

        c1, c4 = self.init_conv(x)

        fla = fla.expand(-1, c1.size(2), c1.size(3), -1)
        low_concat = torch.cat((c1, fla.permute(0, 3, 1, 2)), dim=1)

        fha = fha.expand(-1, c4.size(2), c4.size(3), -1)
        high_concat = torch.cat((c4, fha.permute(0, 3, 1, 2)), dim=1)

        fcl = F.relu(self.conv(low_concat))
        fh = self.upsample(self.aspp(high_concat))
        return torch.cat((fcl, fh), dim=1)

    @torch.jit.export
    def raw(self, image, robot_size, start_orientation, goal_orientation):
        """Training view: position logits and unnormalised (cos, sin)."""
        f = self._trunk(image, robot_size, start_orientation, goal_orientation)
        yaw = self.yaw_head(f)
        return self.pos_head(f), yaw[:, 0:1], yaw[:, 1:2]

    def forward(self, image, robot_size, start_orientation, goal_orientation):
        """Deployment view: the two grids the sampler consumes."""
        pos_logits, cos_t, sin_t = self.raw(
            image, robot_size, start_orientation, goal_orientation
        )
        xy = torch.sigmoid(pos_logits)

        angle = torch.atan2(sin_t, cos_t)                       # [-pi, pi]
        angle = torch.remainder(angle, 2.0 * math.pi)           # [0, 2pi)
        yaw = angle + self.YAW_OFFSET                           # [1, 1+2pi)
        # orientation is only defined where a near-optimal path exists; elsewhere emit
        # the "no data" value the sampler tests for
        yaw = torch.where(xy > self.XY_THRESHOLD, yaw, torch.zeros_like(yaw))
        return xy, yaw
