import torch
from PIL import Image
import torchvision.transforms as transforms

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2
from cnn_model import XYYawDiscreetCNN
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Initialize the model
loaded_model = XYYawDiscreetCNN()
model_path = (
    "/home/steven/ros/humble/system/src/kino_smf_move_base_planning/scripts/model_xy_yaw_disc.pth"
)

# Load the state dictionary into the model
loaded_model.load_state_dict(torch.load(model_path))

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()


# Function to preprocess a single image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = (image / 255.0) * 6.0
    image = np.rint(image)
    # np.set_printoptions(threshold=sys.maxsize)
    # print(image)
    # cv2.imwrite("input_image.png", image * 20)

    image = Image.fromarray(image)

    transform = transforms.Compose(
        [transforms.Resize((100, 100)), transforms.ToTensor()]
    )

    image = transform(image)

    zero_channels = torch.zeros(2, image.size(1), image.size(2))
    image = torch.cat((image, zero_channels), dim=0)

    return image.unsqueeze(0)  # Add batch dimension


# Function to preprocess other attributes
def preprocess_attributes(robot_size, start_orientation, goal_orientation):
    robot_size_tensor = torch.tensor([robot_size], dtype=torch.float32).unsqueeze(
        0
    )  # Add batch dimension
    start_orientation_tensor = torch.tensor(
        [start_orientation], dtype=torch.float32
    ).unsqueeze(0)
    goal_orientation_tensor = torch.tensor(
        [goal_orientation], dtype=torch.float32
    ).unsqueeze(0)
    return robot_size_tensor, start_orientation_tensor, goal_orientation_tensor


# Example file path and attributes
# image_path = "/home/sasm/test_dataset/test/2024-07-19_18-33-45_-2.191234_-2.425865.png"
image_path = "/home/steven/Downloads/26_07_dataset/dataset/test/2024-07-26_08-32-18_4.060375_1.058260.png"
robot_size = 0.35
start_orientation = 4.06
goal_orientation = 1.058

# Preprocess inputs
image = preprocess_image(image_path).to(device)
robot_size, start_orientation, goal_orientation = preprocess_attributes(
    robot_size, start_orientation, goal_orientation
)
robot_size = robot_size.to(device)
start_orientation = start_orientation.to(device)
goal_orientation = goal_orientation.to(device)

# Make predictions
with torch.no_grad():
    xy_pdf = loaded_model(image, robot_size, start_orientation, goal_orientation)

# Output shape should be [1, 1, 128, 128] for a single input
print("Output shape:", xy_pdf.shape)

xy_pdf_image = xy_pdf.squeeze().cpu().numpy()

# for i in range(0, len(xy_pdf_image)):
#     for j in range(0, len(xy_pdf_image)):
#         if xy_pdf_image[i, j] < 0.4:
#             xy_pdf_image[i, j] = 0.0

# np.set_printoptions(threshold=sys.maxsize)
# print(xy_pdf_image)

# If the output tensor values are in the range [0, 1], scale to [0, 255]
# xy_pdf_image = (xy_pdf_image * 255).astype(np.uint8)

# # Convert the numpy array to a PIL image
# xy_pil_image = Image.fromarray(xy_pdf_image)

# # Save the PIL image
# xy_pdf_image_path = "xy_pdf.png"
# xy_pil_image.save(xy_pdf_image_path)

# print(f"Output image saved to {xy_pdf_image_path}")

np_tensor = xy_pdf.cpu().numpy()[0]


# Mask to filter out values lower than 0.5 and zero values
mask = np_tensor > 0.5

# Get the coordinates and values of the elements that are non-zero and greater than 0.5
x, y, z = np.nonzero(mask)
values = np_tensor[mask]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the scatter plot
scatter = ax.scatter(x, y, z, c=values, cmap='viridis')

# Add a color bar
fig.colorbar(scatter, ax=ax, label='Values')

# Set axis limits to 0-100
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()