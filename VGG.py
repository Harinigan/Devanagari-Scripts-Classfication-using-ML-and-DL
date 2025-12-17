

# ==============================================================
# Fine-Tuning VGG16 (13 Conv, 5 MaxPool, 3 Dense) - PyTorch
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
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# ===== PREPROCESS =====
# Convert grayscale to 3-channel RGB if needed
X_train = np.repeat(X_train, 3, axis=-1).astype("float32") / 255.0
X_test  = np.repeat(X_test, 3, axis=-1).astype("float32") / 255.0

# Convert to tensors (N, C, H, W)
X_train = torch.FloatTensor(X_train).permute(0, 3, 1, 2).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test  = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(device)
y_test  = torch.LongTensor(y_test).to(device)

# ===== LOAD PRETRAINED VGG16 =====
model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

# Modify final classifier for 10-class classification
num_classes = 10
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

# Move to device
model = model.to(device)

# ===== OPTIMIZER (fine-tune all layers) =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # smaller LR for fine-tuning

# ===== DATALOADER =====
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

train_losses = []
train_accuracies = []

print("Fine-tuning VGG16 (All Layers)...")

# ===== TRAINING LOOP =====
for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    acc = 100 * correct / total
    train_losses.append(epoch_loss / len(loader))
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1:02d} | Loss: {train_losses[-1]:.4f} | Acc: {acc:.2f}%")

# ===== EVALUATION =====
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()

print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test.cpu(), predicted.cpu())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Fine-Tuned VGG16")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===== CLASSIFICATION REPORT =====
print("\nClassification Report:")
print(classification_report(y_test.cpu(), predicted.cpu(), digits=4))

# ===== TRAINING PLOTS =====
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

