import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import sys

# Define a custom dataset
class CustomDatasetXY(Dataset):
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
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        image = np.rint(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)
        # cv2.imwrite("input_image.png", image * 20)

        image = Image.fromarray(image)

        # print("#####")
        # print(self.pdf_maps[idx].shape)
        pdf_image = torch.from_numpy(self.pdf_maps[idx]).unsqueeze(0)
        # pdf_image = Image.fromarray(self.pdf_maps[idx])
        # print(pdf_image.shape)

        if self.transform:
            image = self.transform(image)
            zero_channels = torch.zeros(2, image.size(1), image.size(2))
            image = torch.cat((image, zero_channels), dim=0)
            # pdf_image = self.transform(pdf_image)
            # pdf_image = torch.from_numpy(pdf_image)
            # print(pdf_image.shape)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return (
            image,
            pdf_image,
            robot_size,
            start_orientation,
            goal_orientation,
        )


# Define a custom dataset
class CustomDatasetXYYawDiscreet(Dataset):
    def __init__(
        self,
        image_paths,
        xy_yaw_discreet_maps,
        robot_size,
        start_orientation,
        goal_orientation,
        transform=None,
    ):
        self.image_paths = image_paths
        self.xy_yaw_discreet_maps = xy_yaw_discreet_maps
        self.robot_size = robot_size
        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.transform = transform
        # self.transform_disc = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        image = np.rint(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)
        # cv2.imwrite("input_image.png", image * 20)

        image = Image.fromarray(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(type(self.xy_yaw_discreet_maps[idx]))
        # print(self.xy_yaw_discreet_maps[idx].shape)
        xy_yaw_disc_image = torch.from_numpy(self.xy_yaw_discreet_maps[idx])
        

        if self.transform:
            image = self.transform(image)
            zero_channels = torch.zeros(2, image.size(1), image.size(2))
            image = torch.cat((image, zero_channels), dim=0)
            # xy_yaw_disc_image = self.transform_disc(xy_yaw_disc_image)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return (
            image,
            xy_yaw_disc_image,
            robot_size,
            start_orientation,
            goal_orientation,
        )

class CustomDatasetYaw(Dataset):
    def __init__(
        self,
        image_paths,
        yaw_maps,
        robot_size,
        start_orientation,
        goal_orientation,
        transform=None,
    ):
        self.image_paths = image_paths
        self.yaw_maps = yaw_maps
        self.robot_size = robot_size
        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        image = np.rint(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)
        # cv2.imwrite("input_image.png", image * 20)

        image = Image.fromarray(image)

        yaw_image = torch.from_numpy(self.yaw_maps[idx]).unsqueeze(0)
        # yaw_image = self.yaw_maps[idx]

        # print("+++++++++++++++++++++++++++++++++++++++")
        # print("YAW MAPS WITHOUT TRANSFORM")
        # print("+++++++++++++++++++++++++++++++++++++++")
        # np.set_printoptions(threshold=sys.maxsize)
        # print(np.array(yaw_image))

        if self.transform:
            image = self.transform(image)
            zero_channels = torch.zeros(2, image.size(1), image.size(2))
            image = torch.cat((image, zero_channels), dim=0)

            # yaw_image = self.transform(yaw_image)

        # print("+++++++++++++++++++++++++++++++++++++++")
        # print("YAW MAPS WITH TRANSFORM")
        # print("+++++++++++++++++++++++++++++++++++++++")
        # np.set_printoptions(threshold=sys.maxsize)
        # print(yaw_image.detach().cpu().numpy())

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return (
            image,
            yaw_image,
            robot_size,
            start_orientation,
            goal_orientation,
        )


class CustomDatasetSingleYaw(Dataset):
    def __init__(
        self,
        image_paths,
        yaw_groundtruths,
        robot_sizes,
        start_orientations,
        goal_orientations,
        x_poses,
        y_poses,
        transform=None,
    ):
        self.image_paths = image_paths
        self.yaw_groundtruths = yaw_groundtruths
        self.robot_sizes = robot_sizes
        self.start_orientations = start_orientations
        self.goal_orientations = goal_orientations
        self.x_poses = x_poses
        self.y_poses = y_poses
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        image = np.rint(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)
        # cv2.imwrite("input_image.png", image * 20)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
            zero_channels = torch.zeros(2, image.size(1), image.size(2))
            image = torch.cat((image, zero_channels), dim=0)

        yaw_groundtruth = torch.tensor(
            [self.yaw_groundtruths[idx]], dtype=torch.float32
        )

        robot_size = torch.tensor([self.robot_sizes[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientations[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientations[idx]], dtype=torch.float32
        )

        x_pose = torch.tensor([self.x_poses[idx]], dtype=torch.float32)

        y_pose = torch.tensor([self.y_poses[idx]], dtype=torch.float32)

        return (
            image,
            yaw_groundtruth,
            robot_size,
            start_orientation,
            goal_orientation,
            x_pose,
            y_pose,
        )


class CustomDatasetXYYaw(Dataset):
    def __init__(
        self,
        image_paths,
        pdf_maps,
        yaw_maps,
        robot_size,
        start_orientation,
        goal_orientation,
        transform=None,
    ):
        self.image_paths = image_paths
        self.pdf_maps = pdf_maps
        self.yaw_maps = yaw_maps
        self.robot_size = robot_size
        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        image = np.rint(image)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)
        # cv2.imwrite("input_image.png", image * 20)

        image = Image.fromarray(image)

        pdf_image = Image.fromarray(self.pdf_maps[idx])

        yaw_image = Image.fromarray(self.yaw_maps[idx])

        if self.transform:
            image = self.transform(image)
            zero_channels = torch.zeros(2, image.size(1), image.size(2))
            image = torch.cat((image, zero_channels), dim=0)

            pdf_image = self.transform(pdf_image)
            yaw_image = self.transform(yaw_image)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return (
            image,
            pdf_image,
            yaw_image,
            robot_size,
            start_orientation,
            goal_orientation,
        )
