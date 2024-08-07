import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from dataset_utils import CustomDatasetXY
from cnn_model import XYCNN

from utils import get_images_and_params, get_pdf_images
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import time

torch.cuda.empty_cache()


def train_model(model, train_loader, xy_criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        epoch_xy_loss = 0.0

        for (
            inputs,
            xy_groundtruth,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in train_loader:
            (
                inputs,
                xy_groundtruth,
                robot_size,
                start_orientations,
                goal_orientations,
            ) = (
                inputs.to(device),
                xy_groundtruth.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            # image_input = inputs[3][0].squeeze().cpu().numpy()
            # cv2.imwrite("input_image.png", image_input / 7 * 255)
            # print("printing")

            # image_groundtruth = xy_groundtruth[3][0].squeeze().cpu().numpy()
            # cv2.imwrite("groundtruth_image.png", image_groundtruth/ 1 * 255)

            # time.sleep(2)

            optimizer.zero_grad()
            xy_inference = model(
                inputs, robot_size, start_orientations, goal_orientations
            )
            xy_loss = xy_criterion(xy_inference, xy_groundtruth)

            xy_loss.backward()
            optimizer.step()

            epoch_xy_loss += xy_loss.item()
            # np.set_printoptions(threshold=sys.maxsize)
            # print(xy_groundtruth.detach().cpu().numpy())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss1: {epoch_xy_loss/len(train_loader)}")

        # if epoch_xy_loss / len(train_loader) < 0.0149:
        #     return model

    print("Training complete")
    return model


def evaluate_model(model, test_loader, xy_criterion):
    model.eval()
    test_loss = 0.0

    groundtruths = []
    inferences = []

    with torch.no_grad():
        for (
            inputs,
            xy_groundtruth,
            robot_size,
            start_orientations,
            goal_orientations,
        ) in test_loader:
            (
                inputs,
                xy_groundtruth,
                robot_size,
                start_orientations,
                goal_orientations,
            ) = (
                inputs.to(device),
                xy_groundtruth.to(device),
                robot_size.to(device),
                start_orientations.to(device),
                goal_orientations.to(device),
            )

            xy_inference = model(
                inputs, robot_size, start_orientations, goal_orientations
            )
            groundtruths.extend(xy_groundtruth.flatten().cpu().numpy())
            inferences.extend(xy_inference.flatten().cpu().numpy())
            # print(xy_inference.cpu().numpy().shape)
            xy_loss = xy_criterion(xy_inference, xy_groundtruth)

            test_loss += xy_loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")
    return groundtruths, inferences


# Hyperparameters and dataset setup

raw_image_path = "/home/steven/ros/humble/system/src/social_nav_benchmark/results/kino_smf_planner/dataset/test"
xy_image_path = "/home/steven/ros/humble/system/src/social_nav_benchmark/results/kino_smf_planner/dataset/train/path"

image_paths, start_orientations, goal_orientations = get_images_and_params(
    raw_image_path
)

start_orientations = np.array(start_orientations).astype(np.float32)
goal_orientations = np.array(goal_orientations).astype(np.float32)
robot_sizes = np.array([0.35] * len(image_paths)).astype(np.float32)

pdf_maps = get_pdf_images(xy_image_path)

transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

dataset = CustomDatasetXY(
    image_paths,
    pdf_maps,
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
model = XYCNN().to(device)

xy_criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


trained_model = train_model(
    model, train_loader, xy_criterion, optimizer, num_epochs=2000
)

model_path = (
    "/home/steven/ros/humble/system/src/kino_smf_move_base_planning/scripts/model_xy_0.0001_final.pth"
)

model_scripted = torch.jit.script(trained_model) 

model_scripted.save(model_path)

# torch.save(trained_model.state_dict(), model_path)

groundtruths, inferences = evaluate_model(trained_model, test_loader, xy_criterion)

# Convert lists to numpy arrays
groundtruths = np.array(groundtruths, np.float32)
inferences = np.array(inferences, np.float32)

# Apply a threshold to binarize the inferences
threshold = 0.5
groundtruths = (groundtruths > threshold).astype(np.float32)
inferences = (inferences > threshold).astype(np.float32)


# Calculate and print metrics
accuracy = accuracy_score(groundtruths, inferences)
precision = precision_score(groundtruths, inferences, average="binary")
recall = recall_score(groundtruths, inferences, average="binary")
f1 = f1_score(groundtruths, inferences, average="binary")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
