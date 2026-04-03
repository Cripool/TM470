import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms


# Transforming the data before it comes into the model

transform = transforms.Compose([
    transforms.ToTensor()
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
])

train_data torchvision. 