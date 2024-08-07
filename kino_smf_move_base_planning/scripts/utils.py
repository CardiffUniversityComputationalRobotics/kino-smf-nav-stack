import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_primitives import MaskedCircularLoss
import torch

def list_files_in_directory(directory):

    filename_list = []

    try:
        # List all files and directories in the specified directory
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    filename_list.append(entry.name)
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")

    return filename_list


def get_images_and_params(directory):

    filenames = list_files_in_directory(directory)

    # print(filenames[0])

    complete_file_paths = []
    start_orientation = []
    goal_orientation = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)
        split_name = i.split("_")
        start_orientation.append(float(round(float(split_name[2]), 3)))
        goal_orientation.append(float(round(float(split_name[-1][:-5]), 3)))

    return complete_file_paths, start_orientation, goal_orientation

def get_images_and_params_yaw(directory):

    filenames = list_files_in_directory(directory)


    complete_file_paths = []
    yaw_images = []
    start_orientation = []
    goal_orientation = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)
        split_name = i.split("_")
        start_orientation.append(float(round(float(split_name[2]), 3)))
        goal_orientation.append(float(round(float(split_name[-1][:-5]), 3)))
        yaw_images.append(get_yaw_image(directory + "/../train/orientation/" + i[:-4] + "yaw.png"))

    return complete_file_paths, yaw_images, start_orientation, goal_orientation

def get_images_and_params_yaw_mask(directory):

    filenames = list_files_in_directory(directory)


    complete_file_paths = []
    yaw_images = []
    start_orientation = []
    goal_orientation = []
    yaw_masks = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)
        split_name = i.split("_")
        start_orientation.append(float(round(float(split_name[2]), 3)))
        goal_orientation.append(float(round(float(split_name[-1][:-5]), 3)))

        yaw_image, mask_image = get_yaw_with_mask_image(directory + "/../train/orientation/" + i[:-4] + "yaw.png")

        yaw_images.append(yaw_image)
        yaw_masks.append(mask_image)

    return complete_file_paths, yaw_images, yaw_masks, start_orientation, goal_orientation

def get_images_yaw_and_params(map_directory, orientation_directory):

    filenames = list_files_in_directory(map_directory)

    complete_file_paths = []
    yaw_groundtruths = []
    start_orientations = []
    goal_orientations = []
    x_poses = []
    y_poses = []

    for i in filenames:

        data = []

        with open(orientation_directory + "/" + i[:-4] + ".txt", "r") as file:
            for line in file:
                # Strip any leading/trailing whitespace and split by comma
                values = line.strip().split(",")
                # Convert values to float if needed, otherwise keep as strings
                values = [
                    float(value) if value.replace(".", "", 1).isdigit() else value
                    for value in values
                ]
                data.append(values)

        for path_data in data:
            complete_file_paths.append(map_directory + "/" + i)
            split_name = i.split("_")
            start_orientations.append(float(round(float(split_name[2]), 3)))
            goal_orientations.append(float(round(float(split_name[-1][:-5]), 3)))
            x_poses.append(path_data[0])
            y_poses.append(path_data[1])
            yaw_groundtruths.append(path_data[2])

    return (
        complete_file_paths,
        yaw_groundtruths,
        start_orientations,
        goal_orientations,
        x_poses,
        y_poses,
    )


def create_probability_map(image_size, path_points, sigma=1.0):
    """
    Creates a probability map given a list of path points.

    :param image_size: Tuple (H, W) indicating the size of the map.
    :param path_points: List of (x, y) tuples indicating the path.
    :param sigma: Standard deviation for the Gaussian blur.
    :return: Probability map as a numpy array of shape (H, W).
    """
    probability_map = np.zeros(image_size, dtype=np.float32)

    # Draw the path on the map
    for point in path_points:
        cv2.circle(probability_map, point, radius=1, color=1.0, thickness=-1)

    # Apply Gaussian blur to create a smooth probability distribution
    probability_map = cv2.GaussianBlur(probability_map, (0, 0), sigma)
    probability_map = np.clip(probability_map, 0, 1)

    return probability_map


def get_pdf_images(directory):

    filenames = list_files_in_directory(directory)

    complete_file_paths = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)

    pdf_maps = []

    for i in complete_file_paths:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 6.0
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)

        pdf_image = np.copy(image)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 4.2 > image[i][j] > 3.8:
                    pdf_image[i][j] = 1.0
                else:
                    pdf_image[i][j] = 0.0

        # cv2.imwrite("pdf_image.png", image)

        probability_map = cv2.GaussianBlur(pdf_image, (0, 0), 1.5)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 4.2 > image[i][j] > 3.8:
                    probability_map[i][j] = 1.0

        probability_map = probability_map * 5

        probability_map = np.clip(probability_map, 0, 1)

        probability_map = np.around(probability_map, decimals=4)

        pdf_maps.append(probability_map)

    return pdf_maps


def blur_around_cell(array, i, j, blur_radius=3, sigma=1):
    # Define the size of the region to blur around the cell
    kernel_size = blur_radius * 2 + 1

    # Define the region of interest (ROI)
    i_start = max(i - blur_radius, 0)
    j_start = max(j - blur_radius, 0)
    i_end = min(i + blur_radius + 1, array.shape[0])
    j_end = min(j + blur_radius + 1, array.shape[1])

    # Extract the region of interest (ROI)
    roi = array[i_start:i_end, j_start:j_end]

    # Create the Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T  # Create a 2D Gaussian kernel

    # Apply the Gaussian blur to the ROI
    blurred_roi = cv2.filter2D(roi, -1, gaussian_kernel)

    # Combine the original array with the blurred ROI
    blurred_array = array.copy()
    blurred_array[i_start:i_end, j_start:j_end] = np.clip(blurred_roi, 0, 6.28)

    return blurred_array


def propagate_value_with_radius(array, index, radius):
    """
    Propagate the value from a specific index to its neighbors within a given radius in a 2D array.

    Parameters:
    array (np.ndarray): The input 2D NumPy array.
    index (tuple): The (row, column) index to propagate from.
    radius (int): The radius within which to propagate the value.

    Returns:
    np.ndarray: The modified 2D NumPy array.
    """
    rows, cols = array.shape
    row, col = index
    value = array[row, col]

    # Propagate the value to the neighbors within the given radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                array[r, c] = value

    return array


def propagate_value_with_radius_3d(array, index, radius):
    """
    Propagate the value from a specific index to its neighbors within a given radius
    in the first two dimensions of a 3D array.

    Parameters:
    array (np.ndarray): The input 3D NumPy array.
    index (tuple): The (depth, row, column) index to propagate from.
    radius (int): The radius within which to propagate the value.

    Returns:
    np.ndarray: The modified 3D NumPy array.
    """
    depth, rows, cols = array.shape
    d, row, col = index
    value = array[d, row, col]

    # Propagate the value to the neighbors within the given radius
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                array[d, r, c] = value

    return array


def get_yaw_image(directory):

    # print(directory)
    image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = (image / 255.0) * 8.0

    yaw_image = np.copy(image)

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if 6.8 < image[i][j] < 7.2:
                yaw_image[i][j] = 0.0

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if image[i][j] >= 0.0 and image[i][j] < 6.8:
                yaw_image[i][j] = image[i][j] + 1
                yaw_image = propagate_value_with_radius(yaw_image, (i, j), 2)

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if image[i][j] >= 0.0 and image[i][j] < 6.8:
                yaw_image[i][j] = image[i][j] + 1

    yaw_image = np.around(yaw_image, decimals=4)
            # np.set_printoptions(threshold=sys.maxsize)
            # print(probability_map)
            # cv2.imwrite("pdf_image.png", probability_map * 255)
            # probability_map = np.clip(probability_map, 0, 1)


    return yaw_image


def get_yaw_images(directory):

    filenames = list_files_in_directory(directory)

    complete_file_paths = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)

    yaw_maps = []

    for i in complete_file_paths:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 8.0

        yaw_image = np.copy(image)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 6.8 < image[i][j] < 7.2:
                    yaw_image[i][j] = 0.0

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if image[i][j] >= 0.0 and image[i][j] < 6.8:
                    yaw_image[i][j] = image[i][j] + 1
                    yaw_image = propagate_value_with_radius(yaw_image, (i, j), 2)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if image[i][j] >= 0.0 and image[i][j] < 6.5:
                    yaw_image[i][j] = image[i][j] + 1

        yaw_image = np.around(yaw_image, decimals=4)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(probability_map)
        # cv2.imwrite("pdf_image.png", probability_map * 255)
        # probability_map = np.clip(probability_map, 0, 1)

        yaw_maps.append(yaw_image)

    return yaw_maps

def get_yaw_with_mask_image(directory):

    # loss = MaskedCircularLoss()

    masked_map = np.zeros((100, 100), np.float32)

    image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = (image / 255.0) * 8.0

    yaw_image = np.copy(image)

    # test_yaw_image = np.full((100, 100), 0)

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if 6.8 < image[i][j] < 7.2:
                yaw_image[i][j] = 0.0

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if image[i][j] >= 0.0 and image[i][j] < 6.8:
                yaw_image[i][j] = image[i][j]
                # test_yaw_image[i][j] = 10
                masked_map[i][j] = 1
                yaw_image = propagate_value_with_radius(yaw_image, (i, j), 2)
                masked_map = propagate_value_with_radius(masked_map, (i, j), 2)
                # test_yaw_image = propagate_value_with_radius(test_yaw_image, (i, j), 2)

    for i in range(0, len(image)):
        for j in range(0, len(image)):
            if image[i][j] >= 0.0 and image[i][j] < 6.5:
                yaw_image[i][j] = image[i][j]
                # test_yaw_image[i][j] = 10

    yaw_image = np.around(yaw_image, decimals=4)

    # test_yaw_image = np.around(test_yaw_image, decimals=4)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(yaw_image)

    # print(loss(torch.tensor(test_yaw_image), torch.tensor(yaw_image), torch.tensor(masked_map)).item())

    return yaw_image, masked_map

def get_yaw_with_mask_images(directory):

    

    filenames = list_files_in_directory(directory)

    complete_file_paths = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)

    yaw_maps = []
    masked_maps = []

    for i in complete_file_paths:
        masked_map = np.zeros((100, 100), np.float32)

        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 8.0

        yaw_image = np.copy(image)

        

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 6.8 < image[i][j] < 7.2:
                    yaw_image[i][j] = 0.0

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if image[i][j] >= 0.0 and image[i][j] < 6.8:
                    yaw_image[i][j] = image[i][j]
                    
                    masked_map[i][j] = 1
                    yaw_image = propagate_value_with_radius(yaw_image, (i, j), 2)
                    
                    masked_map = propagate_value_with_radius(masked_map, (i, j), 2)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if image[i][j] >= 0.0 and image[i][j] < 6.5:
                    yaw_image[i][j] = image[i][j]
                    
        yaw_image = np.around(yaw_image, decimals=4)

        yaw_maps.append(yaw_image)
        masked_maps.append(masked_map)

    return yaw_maps, masked_maps


def get_xy_yaw_discreet_images(directory):

    filenames = list_files_in_directory(directory)

    complete_file_paths = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)

    yaw_maps = []
    xy_yaw_disc_maps = []

    for i in complete_file_paths:
        # print(i)
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32)
        image = (image / 255.0) * 8.0

        yaw_image = np.copy(image)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 6.8 < image[i][j] < 7.2:
                    yaw_image[i][j] = 20.0

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if image[i][j] >= 0.0 and image[i][j] < 6.5:
                    yaw_image[i][j] = image[i][j]

        xy_yaw_disc_map = np.zeros((100, 100, 100), np.float32)
        for i in range(0, len(yaw_image)):
            for j in range(0, len(yaw_image)):
                if yaw_image[i][j] >= 0.0 and yaw_image[i][j] < 6.5:
                    depth = yaw_image[i][j] / 0.0628
                    depth = round(depth)
                    if depth >= 100:
                        depth = 0
                    xy_yaw_disc_map[depth][i][j] = 1.0
                    xy_yaw_disc_map = propagate_value_with_radius_3d(
                        xy_yaw_disc_map, (depth, i, j), 2
                    )

        # np_tensor = xy_yaw_disc_map

        # # Mask to filter out zero values
        # mask = np_tensor != 0

        # # Get the coordinates and values of the elements that are non-zero
        # x, y, z = np.nonzero(mask)
        # values = np_tensor[mask]

        # # Create a 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Create the scatter plot
        # scatter = ax.scatter(x, y, z, c=values, cmap='viridis')

        # # Add a color bar
        # fig.colorbar(scatter, ax=ax, label='Values')

        # # Set axis limits to 0-100
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 100)
        # ax.set_zlim(0, 100)

        # # Set labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # # Show the plot
        # plt.show()

        xy_yaw_disc_maps.append(xy_yaw_disc_map)

    return xy_yaw_disc_maps
