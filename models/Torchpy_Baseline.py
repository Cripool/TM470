import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


import os

print(os.listdir("data/processed/test"))

# ---Data Transform --- 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("data/processed/train", transform=transform)
test_data = datasets.ImageFolder("data/processed/test", transform=transform)
val_data = datasets.ImageFolder("data/processed/val", transform=transform)