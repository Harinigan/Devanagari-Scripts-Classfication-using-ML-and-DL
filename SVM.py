
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

X_train = np.load("/content/drive/MyDrive/processed_data/X_train.npy")
y_train = np.load("/content/drive/MyDrive/processed_data/y_train.npy")
X_test = np.load("/content/drive/MyDrive/processed_data/X_test.npy")
y_test = np.load("/content/drive/MyDrive/processed_data/y_test.npy")

print("Dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# ===== STEP 2: Flatten images for SVM =====
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# ===== STEP 3: Define SVM with RBF kernel =====
n_features = X_train_flat.shape[1]
gamma_value = 1 / (n_features * np.var(X_train_flat))

svm_model = SVC(kernel='rbf', C=10, gamma=gamma_value, probability=True, decision_function_shape='ovr')
svm_model.fit(X_train_flat, y_train)

# ===== STEP 4: Predict =====
y_train_pred = svm_model.predict(X_train_flat)
y_test_pred = svm_model.predict(X_test_flat)

# ===== STEP 5: Evaluate Model =====
# Training Accuracy
train_acc = accuracy_score(y_train, y_train_pred)

# Testing Accuracy
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy : {test_acc * 100:.2f}%")

# Detailed classification report for test set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix (RBF Kernel)')
plt.show()

# Visualize some test samples with predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap='gray')
    ax.set_title(f"True: {y_test[i]}\nPred: {y_test_pred[i]}")
    ax.axis('off')
plt.show()

# ===== STEP 6: ROC AUC Curves (One-vs-Rest) =====
# Binarize the labels for multi-class ROC
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)

# Get decision scores for ROC computation
y_score = svm_model.decision_function(X_test_flat)

plt.figure(figsize=(10,8))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {cls} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for SVM (RBF Kernel, Multi-class)')
plt.legend(loc="lower right")
plt.show()
