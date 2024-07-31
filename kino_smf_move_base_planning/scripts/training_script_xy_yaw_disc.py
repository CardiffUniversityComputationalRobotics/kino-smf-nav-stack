import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from dataset_utils import CustomDatasetXYYawDiscreet
from cnn_model import XYYawDiscreetCNN

from utils import get_images_and_params, get_xy_yaw_discreet_images
import sys

torch.cuda.empty_cache()


def train_model(model, train_loader, xy_criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        epoch_xy_loss = 0.0

        for (
            inputs,
            xy_yaw_groundtruth,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in train_loader:
            (
                inputs,
                xy_yaw_groundtruth,
                robot_size,
                start_orientations,
                goal_orientations,
            ) = (
                inputs.to(device),
                xy_yaw_groundtruth.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            optimizer.zero_grad()
            xy_yaw_inference = model(
                inputs, robot_size, start_orientations, goal_orientations
            )
            xy_yaw_loss = xy_criterion(xy_yaw_inference, xy_yaw_groundtruth)

            xy_yaw_loss.backward()
            optimizer.step()

            epoch_xy_loss += xy_yaw_loss.item()
            # np.set_printoptions(threshold=sys.maxsize)
            # print(xy_groundtruth.detach().cpu().numpy())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss1: {epoch_xy_loss/len(train_loader)}")

        # if epoch_xy_loss / len(train_loader) < 0.024:
        #     return model

    print("Training complete")
    return model


def evaluate_model(model, test_loader, xy_criterion):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for (
            inputs,
            xy_yaw_groundtruth,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in test_loader:
            (
                inputs,
                xy_yaw_groundtruth,
                robot_size,
                start_orientations,
                goal_orientations,
            ) = (
                inputs.to(device),
                xy_yaw_groundtruth.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            xy_yaw_inference = model(
                inputs, robot_size, start_orientations, goal_orientations
            )
            xy_yaw_loss = xy_criterion(xy_yaw_inference, xy_yaw_groundtruth)

            test_loss += xy_yaw_loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")


# Hyperparameters and dataset setup

raw_image_path = "/home/sasm/ros/humble/system/src/social_nav_benchmark/results/kino_smf_planner/dataset/test"
xy_image_path = "/home/sasm/ros/humble/system/src/social_nav_benchmark/results/kino_smf_planner/dataset/train/path"
yaw_image_path = "/home/sasm/ros/humble/system/src/social_nav_benchmark/results/kino_smf_planner/dataset/train/orientation"

image_paths, start_orientations, goal_orientations = get_images_and_params(
    raw_image_path
)

start_orientations = np.array(start_orientations).astype(np.float32)
goal_orientations = np.array(goal_orientations).astype(np.float32)
robot_sizes = np.array([0.35] * len(image_paths)).astype(np.float32)

xy_yaw_maps = get_xy_yaw_discreet_images(yaw_image_path)

transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

dataset = CustomDatasetXYYawDiscreet(
    image_paths,
    xy_yaw_maps,
    robot_sizes,
    start_orientations,
    goal_orientations,
    transform,
)

# Split the dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XYYawDiscreetCNN().to(device)

xy_criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trained_model = train_model(
    model, train_loader, xy_criterion, optimizer, num_epochs=2000
)

model_path = (
    "/home/sasm/ros/humble/system/src/kino_smf_move_base_planning/scripts/model_xy_yaw_disc.pth"
)
torch.save(model.state_dict(), model_path)

evaluate_model(trained_model, test_loader, xy_criterion)
