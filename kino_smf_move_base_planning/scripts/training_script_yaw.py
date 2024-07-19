import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from dataset_utils import CustomDatasetXYYaw
from cnn_model import CustomCNN

from utils import get_images_and_params, get_pdf_images, get_yaw_images

torch.cuda.empty_cache()


def train_model(
    model, train_loader, xy_criterion, yaw_criterion, optimizer, num_epochs=25
):
    model.train()
    for epoch in range(num_epochs):
        epoch_xy_loss = 0.0
        epoch_yaw_loss = 0.0

        for (
            inputs,
            pdf_groundtruth,
            yaw_groundtruth,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in train_loader:
            (
                inputs,
                pdf_groundtruth,
                yaw_groundtruth,
                robot_size,
                start_orientations,
                goal_orientations,
            ) = (
                inputs.to(device),
                pdf_groundtruth.to(device),
                yaw_groundtruth.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            optimizer.zero_grad()
            xy_pdf, yaw_inference = model(
                inputs, robot_size, start_orientations, goal_orientations
            )
            xy_loss = xy_criterion(xy_pdf, pdf_groundtruth)
            yaw_loss = yaw_criterion(yaw_inference, yaw_groundtruth)
            total_loss = xy_loss + yaw_loss
            total_loss.backward()
            optimizer.step()

            epoch_xy_loss += xy_loss.item()
            epoch_yaw_loss += yaw_loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss1: {epoch_xy_loss/len(train_loader)}, Loss2: {epoch_yaw_loss/len(train_loader)}"
    )

    print("Training complete")
    return model


# Hyperparameters and dataset setup

raw_image_path = "/home/sasm/Downloads/dataset_18_07_2024/dataset/test"
xy_image_path = "/home/sasm/Downloads/dataset_18_07_2024/dataset/train/path"
yaw_image_path = "/home/sasm/Downloads/dataset_18_07_2024/dataset/train/orientation"

image_paths, start_orientations, goal_orientations = get_images_and_params(
    raw_image_path
)

start_orientations = np.array(start_orientations).astype(np.float32)
goal_orientations = np.array(goal_orientations).astype(np.float32)
robot_sizes = np.array([0.35] * len(image_paths)).astype(np.float32)

pdf_maps = get_pdf_images(xy_image_path)
yaw_maps = get_yaw_images(yaw_image_path)

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

dataset = CustomDatasetXYYaw(
    image_paths,
    pdf_maps,
    yaw_maps,
    robot_sizes,
    start_orientations,
    goal_orientations,
    transform,
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)

xy_criterion = nn.BCELoss()
yaw_criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trained_model = train_model(
    model, train_loader, xy_criterion, yaw_criterion, optimizer, num_epochs=50
)

model_path = (
    "/home/sasm/ros/humble/system/src/kino_smf_move_base_planning/scripts/model.pth"
)
torch.save(model.state_dict(), model_path)
