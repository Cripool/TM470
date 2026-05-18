import torch
from time import perf_counter
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import ImageFile
import os
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score
)
import seaborn as sns

# ------------------------
# SETTINGS
# ------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True # Temporary safety net

train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"
run_id = time.strftime("%d%m%Y_%H%M")

base_results_dir = os.path.join("results", "baseline")
results_dir = os.path.join(base_results_dir, f"run_{run_id}")

os.makedirs(results_dir, exist_ok=True)

print(f"Saving results to: {results_dir}")

Batch_size = 32
epochs = 10
Learning_Rate = 0.001
Image_size = (224, 224)

# ------------------------
# DEVICE
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Transforms
# ------------------------

transform = transforms.Compose([
    transforms.Resize((Image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])
])


model = models.resnet18(pretrained=True)

#Freeze all layers except the final classification layer

for name, param in model.named_parameters():
    if "fc" in name: # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate, momentum=0.9) # using the same parameters as baseline


# Move the modfel to the GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


