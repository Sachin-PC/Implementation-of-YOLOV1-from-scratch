# Sachin Palahalli Chandrakumar
# Code to implement dataset operations

import pandas as pd
import torch
from PIL import Image


class PascalVOCData(torch.utils.data.Dataset):
    """
        Class to manage and retrieve the data being considered
    """

    def __init__(self, csv_file, grid_cells=7, bounding_box=2, object_classes=20,
                 transform=None):
        self.samples = pd.read_csv(csv_file)
        self.grid_cells = grid_cells
        self.bounding_box = bounding_box
        self.object_classes = object_classes
        self.transform = transform
        self.image_directory = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                               "Semester/PRCV/Projects/6/YOLO_V1/data/archive/images"
        self.labels_directory = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                                "Semester/PRCV/Projects/6/YOLO_V1/data/archive/labels"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label_path = self.labels_directory + "/" + self.samples.iloc[index, 1]
        image_path = self.image_directory + "/" + self.samples.iloc[index, 0]
        bounding_boxes = []
        with open(label_path) as labels:
            for label in labels.readlines():
                # print("label = ", label)
                labels_values_as_string = label.replace("\n", "").split()
                labels_values = [float(value) if '.' in value else int(value) for value in labels_values_as_string]
                bounding_boxes.append(labels_values)

        image = Image.open(image_path)
        bounding_boxes = torch.tensor(bounding_boxes)
        # print("bounding_boxes = \n", bounding_boxes)
        if self.transform:
            image, bounding_boxes = self.transform(image, bounding_boxes)

        labels_data = torch.zeros((self.grid_cells, self.grid_cells, self.bounding_box*5 + self.object_classes))

        # print("labels_data shaep = ",labels_data.shape)
        for bounding_b in bounding_boxes:
            class_label, normalized_x, normalized_y, normalized_width, normalized_height = bounding_b.tolist()
            class_label = int(class_label)
            i = int(self.grid_cells*normalized_y)
            j = int(self.grid_cells*normalized_x)

            cell_x_coordinate = (self.grid_cells*normalized_x) - int(self.grid_cells*normalized_x)
            cell_y_coordinate = (self.grid_cells*normalized_y) - int(self.grid_cells*normalized_y)
            cell_width_value = normalized_width*self.grid_cells
            cell_height_value = normalized_height*self.grid_cells

            # print("class label = ",class_label)
            #
            # print("cell_x_coordinate = ",cell_x_coordinate)
            # print("cell_y_coordinate = ",cell_y_coordinate)
            # print("cell_width_value = ",cell_width_value)
            # print("cell_height_value = ",cell_height_value)

            if labels_data[i,j,20] == 0:
                labels_data[i,j,20] = 1
                labels_data[i,j,21:25] = torch.tensor([cell_x_coordinate,cell_y_coordinate,cell_width_value,cell_height_value])
                labels_data[i,j,class_label] = 1

        return image, labels_data



