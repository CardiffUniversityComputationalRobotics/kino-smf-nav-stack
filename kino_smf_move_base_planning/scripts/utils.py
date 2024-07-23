import os
import cv2
import numpy as np
import sys


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

    complete_file_paths = []
    start_orientation = []
    goal_orientation = []

    for i in filenames:
        complete_file_paths.append(directory + "/" + i)
        split_name = i.split("_")
        start_orientation.append(float(round(float(split_name[2]), 3)))
        goal_orientation.append(float(round(float(split_name[-1][:-5]), 3)))

    return complete_file_paths, start_orientation, goal_orientation


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
                    pdf_image[i, j] = 1.0
                else:
                    pdf_image[i, j] = 0.0

        # cv2.imwrite("pdf_image.png", image)

        probability_map = cv2.GaussianBlur(pdf_image, (0, 0), 1.5)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 4.2 > image[i][j] > 3.8:
                    probability_map[i, j] = 1.0

        probability_map = np.around(probability_map, decimals=4)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(probability_map)
        # cv2.imwrite("pdf_image.png", probability_map * 255)
        # probability_map = np.clip(probability_map, 0, 1)

        pdf_maps.append(probability_map)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(pdf_maps[-1])
    # cv2.imwrite("pdf_image.png", pdf_maps[-1] * 255)
    # print(complete_file_paths[-1])

    return pdf_maps


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
        # np.set_printoptions(threshold=sys.maxsize)
        # print(image)

        yaw_image = np.copy(image)

        for i in range(0, len(image)):
            for j in range(0, len(image)):
                if 6.8 < image[i][j] < 7.2:
                    yaw_image[i, j] = -1.0

        np.set_printoptions(threshold=sys.maxsize)
        # print(yaw_image)

        # cv2.imwrite("yaw_image.png", yaw_image)

        yaw_image = np.around(yaw_image, decimals=4)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(probability_map)
        # cv2.imwrite("pdf_image.png", probability_map * 255)
        # probability_map = np.clip(probability_map, 0, 1)

        yaw_maps.append(yaw_image)

    # np.set_printoptions(threshold=sys.maxsize)
    # print(yaw_maps[-6])
    # cv2.imwrite("yaw_image.png", yaw_maps[-6] * 255)
    # print(complete_file_paths[-6])

    return yaw_maps
