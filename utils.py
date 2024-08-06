# Sachin Palahalli Chandrakumar
# Code to implement util functions

import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def intersection_over_union(predictions, target, box_type="midpoint"):
    """
        Method to find intersection over union of predicted and target
        Parameters:
        predictions : prediction values
        target : target values
        box_type: type of box considered
    """
    if box_type == "midpoint":
        b1_x1 = predictions[..., 0:1] - predictions[..., 2:3] / 2
        b1_x2 = predictions[..., 0:1] - predictions[..., 2:3] / 2
        b1_y1 = predictions[..., 1:2] - predictions[..., 3:4] / 2
        b1_y2 = predictions[..., 1:2] - predictions[..., 3:4] / 2
        b2_x1 = target[..., 0:1] - target[..., 2:3] / 2
        b2_x2 = target[..., 0:1] - target[..., 2:3] / 2
        b2_y1 = target[..., 1:2] - target[..., 3:4] / 2
        b2_y2 = target[..., 1:2] - target[..., 3:4] / 2

    if box_type == "corners":
        b1_x1 = predictions[..., 0:1]
        b1_y1 = predictions[..., 1:2]
        b1_x2 = predictions[..., 2:3]
        b1_y2 = predictions[..., 3:4]
        b2_x1 = target[..., 0:1]
        b2_y1 = target[..., 1:2]
        b2_x2 = target[..., 2:3]
        b2_y2 = target[..., 3:4]

    x1 = torch.max(b1_x1, b2_x1)
    x2 = torch.max(b1_x2, b2_x2)
    y1 = torch.max(b1_y1, b2_y1)
    y2 = torch.max(b1_y2, b2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    b1_area = abs((b1_x1 - b1_x2) * (b1_y1 - b1_y2))
    b2_area = abs((b2_x1 - b2_x2) * (b2_y1 - b2_y2))

    iou = intersection / (b1_area + b2_area - intersection + 12 - 6)

    return iou


def non_max_suppression(bounding_boxes, intersection_over_union_threshold, threshold, box_type="corners"):
    """
        finds the non maxima suppression of the input bounding boxes
        Parameters:
        bounding_boxes : bounding box values
        intersection_over_union_threshold : intersection_over_union_threshold value
        threshold: threshold value
        box_type: type of box considered
    """
    bounding_boxes_considered = []
    for bounding_box in bounding_boxes:
        if bounding_box[1] > threshold:
            bounding_boxes_considered.append(bounding_box)

    bounding_boxes_considered = sorted(bounding_boxes_considered, key=lambda bb: bb[1], reverse=True)
    nms_bounding_boxes = []

    while bounding_boxes_considered:
        considered_bounding_box = bounding_boxes_considered.pop(0)
        bounding_boxes_considered = [bounding_box for bounding_box in bounding_boxes_considered if
                                     bounding_box[0] != considered_bounding_box[0]
                                     or intersection_over_union(torch.tensor(considered_bounding_box[2:]),
                                                                torch.tensor(bounding_box[2:]),
                                                                box_type=box_type, ) < intersection_over_union_threshold]

        nms_bounding_boxes.append(considered_bounding_box)

    return nms_bounding_boxes


def mean_average_precision(predicted_boxes, true_boxes, intersection_over_union_threshold=0.5, box_type="midpoint",
                           num_classes=20):
    """
        finds the mean_average_precision of the predicted_boxes and true boxes
        Parameters:
        predicted_boxes : bounding box values
        true_boxes: true box values
        intersection_over_union_threshold : intersection_over_union_threshold value
        box_type: type of box considered
        num_classes = total class
    """
    class_average_precisions = []
    for object_class in range(num_classes):
        object_detections = []
        ground_truths = []

        for object_detection in predicted_boxes:
            if object_detection[1] == object_class:
                object_detections.append(object_detection)

        for true_box in true_boxes:
            if true_box[1] == object_class:
                ground_truths.append(true_box)

        bounding_boxes_amount_map = Counter([ground_truth[0] for ground_truth in ground_truths])

        for key, value in bounding_boxes_amount_map.items():
            bounding_boxes_amount_map[key] = torch.zeros(value)

        object_detections.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(object_detections)))
        false_positive = torch.zeros((len(object_detections)))
        total_true_bounding_boxes = len(ground_truths)

        if total_true_bounding_boxes == 0:
            continue

        for object_detections_index, object_detection in enumerate(object_detections):

            ground_truth_image = []
            for bounding_box in ground_truths:
                if bounding_box[0] == object_detection[0]:
                    ground_truth_image.append(bounding_box)

            number_ground_truths = len(ground_truth_image)

            best_intersection_of_union = 0

            for index, ground_truth in enumerate(ground_truth_image):
                iou = intersection_over_union(torch.tensor(object_detection[3:]), torch.tensor(ground_truth[3:]),
                                              box_type=box_type)

                if iou > best_intersection_of_union:
                    best_intersection_of_union = iou
                    best_ground_truth_index = index

            if best_intersection_of_union > intersection_over_union_threshold:
                if bounding_boxes_amount_map[object_detection[0]][best_ground_truth_index] == 0:
                    true_positive[object_detections_index] = 1
                    bounding_boxes_amount_map[object_detection[0]][best_ground_truth_index] = 1
                else:
                    false_positive[object_detections_index] = 1
            else:
                false_positive[object_detections_index] = 1

        true_positive_cumsum = torch.cumsum(true_positive, dim=0)
        false_positive_cumsum = torch.cumsum(false_positive, dim=0)
        recalls = true_positive_cumsum / (total_true_bounding_boxes + 1e-6)
        precisions = torch.divide(true_positive_cumsum, (true_positive_cumsum + false_positive_cumsum + 1e-6))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        class_average_precisions.append(torch.trapz(precisions, recalls))

        return sum(class_average_precisions) / len(class_average_precisions)


def convert_cell_boxes(predictions, grid_cells=7):
    """
    converts bounding box values relative to entire image than to cell ratios
    Parameters:
    predictions : prediction values
    grid_cells : number of grid cells in the image
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bounding_boxes_1 = predictions[..., 21:25]
    bounding_boxes_2 = predictions[..., 26:30]

    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)

    bounding_box_considered = scores.argmax(0).unsqueeze(-1)

    best_boxes = bounding_boxes_1 * (1 - bounding_box_considered) + bounding_box_considered * bounding_boxes_2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x_coordinate = 1 / grid_cells * (best_boxes[..., : 1] + cell_indices)
    y_coordinate = 1 / grid_cells * (best_boxes[..., : 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / grid_cells * best_boxes[..., 2:4]
    converted_bounding_boxes = torch.cat((x_coordinate, y_coordinate, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_predictions = torch.cat((predicted_class, best_confidence, converted_bounding_boxes), dim=-1)

    return converted_predictions


def cell_boxes_to_boxes(out, grid_cells=7):
    """
        Method to get all bounding box values
        Parameters:
        out : labels
        grid_cells : grid cells of the image
    """
    converted_predictions = convert_cell_boxes(out).reshape(out.shape[0], grid_cells * grid_cells, -1)
    converted_predictions[..., 0] = converted_predictions[..., 0].long()
    all_bounding_boxes = []

    for ex_index in range(out.shape[0]):
        bounding_boxes = []

        for bounding_box_index in range(grid_cells * grid_cells):
            bounding_boxes.append([x.item() for x in converted_predictions[ex_index, bounding_box_index, :]])
        all_bounding_boxes.append(bounding_boxes)

    return all_bounding_boxes


def get_bounding_boxes(loader, model, intersection_over_union_threshold, threshold,
                       box_type="midpoint", device="cpu"):
    """
        Method to get bounding box values
        Parameters:
        loader : data loader
        model : yolo model
        intersection_over_union_threshold : intersection_over_union_threshold value
        threshold: threshold value
        box_type: type of box considered
        device = cpu/gpu
    """
    all_prediction_boxes = []
    all_true_boxes = []

    model.eval()
    train_index = 0

    for batch_index, batch in enumerate(loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            predictions = model(images)

        batch_size = images.shape[0]
        true_bounding_boxes = cell_boxes_to_boxes(labels)
        bounding_boxes = cell_boxes_to_boxes(predictions)

        for index in range(batch_size):
            non_max_suppression_boxes = non_max_suppression(bounding_boxes[index],
                                                            intersection_over_union_threshold=intersection_over_union_threshold,
                                                            threshold=threshold,
                                                            box_type=box_type
                                                            )

            for non_max_suppression_box in non_max_suppression_boxes:
                all_prediction_boxes.append([train_index] + non_max_suppression_box)

            for bounding_box in true_bounding_boxes[index]:
                if bounding_box[1] > threshold:
                    all_true_boxes.append([train_index] + bounding_box)

            train_index += 1

    model.train()

    return all_prediction_boxes, all_true_boxes


def train_batch(model, epoch, optimizer, loss_function, train_data_loader, device):
    """
        Method to train the model on one epoch
        Parameters:
        model : yolo model
        epoch : epoch value
        optimizer : optimizer used for training the model
        loss_function : loss function used for calculating the loss
        train_data_loader: data loader
        device = cpu/gpu
    """
    average_loss = []
    model.train()
    enumerator = tqdm(train_data_loader, leave=True)
    for index, batch in enumerate(enumerator):
        no_of_batches = len(train_data_loader)
        images, targets = batch
        images, targets = images.to(device), targets.to(device)
        predictions = model(images)

        prediction_loss = loss_function(predictions, targets)
        optimizer.zero_grad()
        prediction_loss.backward()
        optimizer.step()
        average_loss.append(prediction_loss.item())
        enumerator.set_postfix(loss=prediction_loss.item)

    average_loss = sum(average_loss) / len(average_loss)
    print("Epoch ", epoch, " average loss = ", average_loss)


def plot_image(image, bounding_boxes):
    """
        Method to plot predicted images
        Parameters:
        image : input image
        bounding_boxes : bounding_boxes values
    """
    img = np.array(image)
    height, width, _ = img.shape

    figure, axes = plt.subplots(1)
    axes.imshow(img)

    for bounding_box in bounding_boxes:
        bounding_box = bounding_box[2:]
        assert len(bounding_box) == 4
        upper_left_x = bounding_box[0] - bounding_box[2] / 2
        upper_left_y = bounding_box[1] - bounding_box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            bounding_box[2] * width,
            bounding_box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        axes.add_patch(rect)

    plt.show()
