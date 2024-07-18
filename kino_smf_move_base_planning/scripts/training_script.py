import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from dataset_utils import CustomDataset
from cnn_model import CustomCNN

from utils import get_images_and_params, get_pdf_images

torch.cuda.empty_cache()


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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}")

    print("Training complete")
    return model


# Hyperparameters and dataset setup

raw_image_path = "/home/sasm/Downloads/dataset_18_07_2024/dataset/test"
path_image_path = "/home/sasm/Downloads/dataset_18_07_2024/dataset/train/path"

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

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

model_path = (
    "/home/sasm/ros/humble/system/src/kino_smf_move_base_planning/scripts/model.pth"
)
torch.save(model.state_dict(), model_path)
