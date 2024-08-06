# Sachin Palahalli Chandrakumar
# main method used to create the model and train the model

from PascalVOCData import PascalVOCData
from Transformation import Transformation
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from YoloV1Loss import YoloV1Loss
from utils import *
from model import Yolov1

import os

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    IMG_DIR = '/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth ' \
              'Semester/PRCV/Projects/6/YOLO_V1/data/archive/images'
    LABEL_DIR = '/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth ' \
                'Semester/PRCV/Projects/6/YOLO_V1/data/archive/labels'

    transform = Transformation([transforms.Resize((448, 448)), transforms.ToTensor(), ])

    train_dataset = PascalVOCData(
        "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth "
        "Semester/PRCV/Projects/6/YOLO_V1/data/archive/train.csv",
        transform=transform,
    )

    test_dataset = PascalVOCData(
        "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth "
        "Semester/PRCV/Projects/6/YOLO_V1/data/archive/test.csv",
        transform=transform,
    )

    # train_dataset.getitem(0)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=2, shuffle=True, drop_last=True)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=16, num_workers=2, shuffle=True, drop_last=True)

    model = Yolov1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)
    loss_function = YoloV1Loss()

    n_epochs = 100
    intersection_over_union_threshold = 0.5
    threshold = 0.4
    box_type = "midpoint"
    for epoch in range(n_epochs):
        print("epoch = ",epoch)
        predicted_boxes, target_boxes = get_bounding_boxes(
            train_data_loader,model,
            intersection_over_union_threshold=intersection_over_union_threshold, threshold=threshold)
        mean_average_precision_value = mean_average_precision(
            predicted_boxes, target_boxes,
            intersection_over_union_threshold=intersection_over_union_threshold, box_type=box_type)

        print("mean_average_precision_value = ",mean_average_precision_value)

        train_batch(model=model.float(), epoch=epoch,optimizer=optimizer,loss_function=loss_function,
                    train_data_loader=train_data_loader, device=device)

        model_save_path = os.path.join("/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth "
                                       "Semester/PRCV/Projects/6/YOLO_V1/saved_model",f'epoch_{epoch}_model.pth')
        torch.save(model.state_dict(),model_save_path)

