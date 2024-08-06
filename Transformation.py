# Sachin Palahalli Chandrakumar
# Code to implement custom transformations

class Transformation(object):
    """
    transformation class used for custom transformation implementation
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bounding_boxes):
        for transform in self.transforms:
            image, bounding_boxes = transform(image), bounding_boxes

        return image, bounding_boxes
