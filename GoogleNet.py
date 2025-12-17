

# ==============================================================
# GoogLeNet (Inception v1) Fine-Tuning - No Auxiliary Classifiers
# ==============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD DATA =====
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# ===== PREPROCESS =====
# Convert grayscale to RGB if needed
if X_train.shape[-1] == 1:
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Convert to PyTorch tensors (N, C, H, W)
X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
y_test = torch.LongTensor(y_test)

X_train, y_train, X_test, y_test = X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device)

# ===== MODEL =====
# Load GoogLeNet (Inception v1) WITHOUT auxiliary classifiers
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet',
                       pretrained=True, aux_logits=False)

num_classes = 10  # Adjust for your dataset

# Replace final classification layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# OPTIONAL: Freeze early layers and fine-tune deeper ones
for name, param in model.named_parameters():
    if "inception4e" not in name and "inception5" not in name and "fc" not in name:
        param.requires_grad = False  # freeze earlier conv layers

model = model.to(device)

# ===== TRAINING SETUP =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': [p for n, p in model.named_parameters() if "inception4e" in n or "inception5" in n], 'lr': 1e-4},
    {'params': [p for n, p in model.named_parameters() if "fc" in n], 'lr': 1e-3}
])

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

train_losses, train_accuracies = [], []
print("Training GoogLeNet (Inception v1) without auxiliary heads...")

# ===== TRAINING LOOP =====
for epoch in range(20):  # You can increase to 25–30 epochs for better results
    model.train()
    epoch_loss, correct, total = 0, 0, 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)  # returns only main output
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (pred == batch_y).sum().item()

    acc = 100 * correct / total
    train_losses.append(epoch_loss / len(loader))
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1:02d}, Loss: {train_losses[-1]:.4f}, Acc: {acc:.2f}%")

# ===== EVALUATION =====
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, pred = torch.max(outputs, 1)
    acc = (pred == y_test).float().mean()
    print(f"\n Test Accuracy: {acc.item() * 100:.2f}%")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test.cpu(), pred.cpu())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - GoogLeNet (No Aux Heads)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ===== CLASSIFICATION REPORT =====
print("\nClassification Report:")
print(classification_report(y_test.cpu(), pred.cpu(), digits=4))

# ===== TRAINING CURVES =====
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-', linewidth=2)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

