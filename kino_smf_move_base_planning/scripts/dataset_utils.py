import torch
from torch.utils.data import Dataset
from PIL import Image


# Define a custom dataset
class CustomDataset(Dataset):
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
        image = Image.open(self.image_paths[idx])
        pdf_image = Image.fromarray(self.pdf_maps[idx])
        if self.transform:
            image = self.transform(image)
            pdf_image = self.transform(pdf_image)

        robot_size = torch.tensor([self.robot_size[idx]], dtype=torch.float32)
        start_orientation = torch.tensor(
            [self.start_orientation[idx]], dtype=torch.float32
        )
        goal_orientation = torch.tensor(
            [self.goal_orientation[idx]], dtype=torch.float32
        )

        return image, pdf_image, robot_size, start_orientation, goal_orientation
