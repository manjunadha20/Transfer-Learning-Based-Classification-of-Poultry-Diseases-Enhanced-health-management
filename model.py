
# poultry_disease_train.py
import os
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torchvision.models import resnet18, ResNet18_Weights

# Data Preparation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

num_workers = 2
pin_memory = True if torch.cuda.is_available() else False
batch_size = 32

train_data_path = 'data/train'
val_data_path = 'data/val'
test_data_path = 'data/test'

train_loader = DataLoader(datasets.ImageFolder(train_data_path, transform=data_transforms),
                          batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(datasets.ImageFolder(val_data_path, transform=data_transforms),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(datasets.ImageFolder(test_data_path, transform=data_transforms),
                         batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

class_labels = datasets.ImageFolder(train_data_path).classes

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "Training on CPU")

# Model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 4)
model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Training
os.makedirs("resnet18", exist_ok=True)
csv_file = "resnet18/training_log.csv"
num_epochs = 40
best_val_loss = float('inf')

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "Time (s)"])

for epoch in range(num_epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    start = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_acc = 100 * correct_train / total_train

    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_acc = 100 * correct_val / total_val
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "resnet18/best.pt")

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss/len(train_loader), val_loss/len(val_loader), train_acc, val_acc, time.time()-start])

    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Time: {time.time()-start:.2f}s")

# Testing
model.load_state_dict(torch.load("resnet18/best.pt"))
model.eval()
correct_test, total_test, test_loss = 0, 0, 0
true_labels, pred_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_test += (preds == labels).sum().item()
        total_test += labels.size(0)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

test_acc = 100 * correct_test / total_test
print(f"Test Accuracy: {test_acc:.2f}%, Loss: {test_loss/len(test_loader):.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.savefig("resnet18/confusion_matrix.png", dpi=300)
plt.show()
print(classification_report(true_labels, pred_labels, target_names=class_labels))
