import numpy as np

import torch
from torchvision import transforms
from torchvision.transforms.functional import rotate


class RGBDataTransform:
    def __call__(self, data):
        return transforms.functional.to_tensor(data).float()
    
class CustomRotationTransform:
    """ Custom rotation transform to rotate image by one of the given angles. """
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = float(np.random.choice(self.angles))
        return rotate(x, angle)