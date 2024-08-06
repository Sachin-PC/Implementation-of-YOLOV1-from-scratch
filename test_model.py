# Sachin Palahalli Chandrakumar
# Code to itest the YOLO model trained

from PascalVOCData import PascalVOCData
from Transformation import Transformation
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
from model import Yolov1

# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":

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

    saved_model_path = "/Users/sachinpc/Documents/Northeastern University/Semesters/Fourth " \
                       "Semester/PRCV/Projects/6/YOLO_V1/saved_model/epoch_99_model.pth"
    model = Yolov1().to(device)
    model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0)

    intersection_over_union_threshold = 0.5
    threshold = 0.4
    box_type = "midpoint"

    for images, labels in train_data_loader:
        images = images.to(device)
        for index in range(10):
            bounding_boxes = cell_boxes_to_boxes(model(images))
            bounding_boxes = non_max_suppression(bounding_boxes[index],
                                                 intersection_over_union_threshold=intersection_over_union_threshold,
                                                 threshold=threshold, box_type=box_type)
            plot_image(images[index].permute(1, 2, 0), bounding_boxes)
