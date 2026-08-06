import torch
import os
import time
from time import perf_counter
from PIL import ImageFile
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score
)
import seaborn as sns
import random


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
# ------------------------
# SETTINGS
# ------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True # Temporary safety net

train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"
run_id = time.strftime("%d%m%Y_%H%M%S")

ResNet18_results_dir = os.path.join("results", "resnet18_corrected")
results_dir = os.path.join(ResNet18_results_dir, f"run_{run_id}")

os.makedirs(results_dir, exist_ok=False)

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

# ------------------------
# DATASETS
# ------------------------

train_data = datasets.ImageFolder("data/processed/train", transform=transform)
test_data = datasets.ImageFolder("data/processed/test", transform=transform)
val_data = datasets.ImageFolder("data/processed/val", transform=transform)

assert train_data.class_to_idx == val_data.class_to_idx, (
    "Training an validation class mappings do not match."
)

assert train_data.class_to_idx == test_data.class_to_idx, (
    "Training and test class mappings do not match."
)

trainloader = DataLoader(train_data, Batch_size, shuffle=True)
valloader = DataLoader(val_data, Batch_size, shuffle=False)
testloader = DataLoader(test_data, Batch_size, shuffle=False)

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Train samples:", len(train_data))
print("Val samples:", len(val_data))
print("Test samples:", len(test_data))


# ------------------------
# MODEL
# ------------------------

model = models.resnet18(weights=None)
    
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print ("Trainable parameters:", trainable_parameters)
print ("Total parameters:", total_parameters)

if trainable_parameters != total_parameters:
    raise ValueError(
        "Not all ResNet18 Parameters are trainable."
    )
# ------------------------
# EVALUATE FUNCTION
# ------------------------

def evaluate_model(model, loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    correct = 0.0
    total = 0
    
    all_labels = []
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim = 1)
            confidences, preds = torch.max(probs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_confidence = float(np.mean(all_confidences))
    
    
    return{
        
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "confidence": avg_confidence,
        "labels": all_labels,
        "preds": all_preds
    }

# ------------------------
# TRAINING LOOP
# ------------------------ 
history = {
    
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": [],
    "epoch_time_sec": []
}

for epoch in range (epochs):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.perf_counter()
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        
    train_loss = running_loss / len(trainloader)
    train_accuracy = correct / total
    
    
    val_metrics = evaluate_model(model, valloader, criterion, device)
    epoch_time = time.perf_counter() - start_time
    
    
    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_metrics["loss"])
    history["val_accuracy"].append(val_metrics["accuracy"])
    history["val_f1"].append(val_metrics["f1"])
    history["val_precision"].append(val_metrics["precision"])
    history["val_recall"].append(val_metrics["recall"])
    history["epoch_time_sec"].append(epoch_time)
        
        
    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

print("Finished Training")

model_path = os.path.join(results_dir, "resnet18_model.pth")
torch.save(model.state_dict(), model_path)

print(f"Saved model to: {model_path}")

# -----------------------------
# SAVE EPOCH HISTORY
# -----------------------------
history_df = pd.DataFrame(history)
excel_path = os.path.join(results_dir, "resnet18_results.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    history_df.to_excel(writer, sheet_name="Epoch Metrics", index=False)

# -----------------------------
# SAVE TRAINING CURVES
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "loss_curve.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
plt.close()

# -----------------------------
# TEST EVALUATION
# -----------------------------
test_metrics = evaluate_model(model, testloader, criterion, device)

print("\nTest Results")
print(f"Test Loss: {test_metrics['loss']:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test F1: {test_metrics['f1']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall: {test_metrics['recall']:.4f}")
print(f"Average Confidence: {test_metrics['confidence']:.4f}")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
report_dict = classification_report(
    test_metrics["labels"],
    test_metrics["preds"],
    target_names=class_names,
    output_dict=True,
    zero_division=0
)
report_df = pd.DataFrame(report_dict).transpose()

summary_df = pd.DataFrame([{
    "model_name": "ResNet18",
    "seed": seed,
    "epochs": epochs,
    "batch_size": Batch_size,
    "learning_rate": Learning_Rate,
    "Image_size": Image_size,
    "test_loss": test_metrics["loss"],
    "test_accuracy": test_metrics["accuracy"],
    "test_f1_macro": test_metrics["f1"],
    "test_precision_macro": test_metrics["precision"],
    "test_recall_macro": test_metrics["recall"],
    "avg_confidence": test_metrics["confidence"],
    "avg_epoch_time_sec": history_df["epoch_time_sec"].mean()
}])

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
    history_df.to_excel(writer,sheet_name="Epoch Metrics", index=False)
    summary_df.to_excel(writer, sheet_name="Test Summary", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report")
    cm_df.to_excel(writer, sheet_name="Confusion Matrix")
   
# -----------------------------
# PER-IMAGE PREDICTION LOG
# -----------------------------
def prediction_log(model, dataset, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for idx, (path, true_label) in enumerate(dataset.samples):
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)
        
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            start = time.perf_counter()
            outputs = model(image)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
                    
            inference_time = time.perf_counter() - start

            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)


            rows.append({
                    "image_path": path,
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred.item()],
                    "correct": class_names[true_label] == class_names[pred.item()],
                    "confidence": confidence.item(),
                    "inference_time_sec": inference_time,
                    "model_name": "ResNet18",
                    "split": "test"
                })

    return pd.DataFrame(rows)

prediction_df = prediction_log(model, test_data, device)

with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    prediction_df.to_excel(writer, sheet_name="Per Image Predictions", index=False)

csv_path = os.path.join(results_dir, "resnet18_per_image_predictions.csv")
prediction_df.to_csv(csv_path, index=False)

print(prediction_df.head())
print(f"Saved per-image CSV to: {csv_path}")
print(f"Per-imge rows: {len(prediction_df)}")
print(f"Expected test images: {len(test_data)}")

print(f"\nSaved results to: {excel_path}")
print(f"Saved graphs to: {results_dir}")