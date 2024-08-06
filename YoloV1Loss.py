# Sachin Palahalli Chandrakumar
# Code to implement yolo loss value

import torch.nn as nn
import torch
from utils import *


class YoloV1Loss(nn.Module):
    """
    Yolo loss calculation implementation. It uses multiple loss values to be combined to obtain Yolo loss
    parameters
    grid_cells: grid cells for each image
    bounding_box: bounding box for each grid cell
    object_classes: total  umber of classes
    """
    def __init__(self, grid_cells=7, bounding_box=2, object_classes=20):
        super(YoloV1Loss, self).__init__()

        self.grid_cells = grid_cells
        self.bounding_box = bounding_box
        self.object_classes = object_classes
        self.mse_loss = nn.MSELoss(reduction="sum")

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # -1 because, it is used to store batch size
        # target is the actual label data
        # predictions is the prediction from the model
        predictions = predictions.reshape(-1, self.grid_cells, self.grid_cells,
                                          self.bounding_box * 5 + self.object_classes)

        intersection_over_union_bounding_box_1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        intersection_over_union_bounding_box_2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        intersection_over_unions = torch.cat([intersection_over_union_bounding_box_1.unsqueeze(0),
                                              intersection_over_union_bounding_box_2.unsqueeze(0)], dim=0)
        intersection_over_union_max, box_considered = torch.max(intersection_over_unions, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        predicted_box_values = exists_box * ((
                box_considered * predictions[..., 26:30] + (1 - box_considered) * predictions[..., 21:25]
        ))

        predicted_box_values[..., 2:4] = torch.sign(predicted_box_values[..., 2:4]) * torch.sqrt(
            torch.abs(predicted_box_values[..., 2:4] + 1e-6))

        target_box_values = exists_box * target[..., 21:25]

        target_box_values[..., 2:4] = torch.sqrt(target_box_values[..., 2:4])

        mse_box_loss = self.mse_loss(torch.flatten(predicted_box_values, end_dim=-2),
                                     torch.flatten(target_box_values, end_dim=-2))

        noobject_loss = self.mse_loss(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                                    torch.flatten((1 - exists_box)* target[..., 20: 21], start_dim=1)
                                    )

        noobject_loss += self.mse_loss(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                                    torch.flatten((1 - exists_box) * target[..., 20: 21], start_dim=1)
                                    )

        class_loss = self.mse_loss(
            torch.flatten(exists_box*predictions[..., :20], end_dim = -2,),
            torch.flatten(exists_box*target[..., :20], end_dim = -2,),
        )

        total_loss = (self.lambda_coord*mse_box_loss + noobject_loss + self.lambda_noobj*noobject_loss + class_loss)

        return total_loss