
# K-Nearest Neighbours (K-NN) for Digit Classification with HOG Features
# k = 3
# ===========================

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from tqdm import tqdm

# ===== STEP 1: Load Dataset =====
X_train = np.load("/content/processed_data/X_train.npy")
y_train = np.load("/content/processed_data/y_train.npy")
X_test = np.load("/content/processed_data/X_test.npy")
y_test = np.load("/content/processed_data/y_test.npy")

print("Dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# ===== STEP 2: Extract HOG Features =====
def extract_hog_features(images):
    features = []
    for img in tqdm(images, desc="Extracting HOG features"):
        if img.ndim == 3:
            img = img.squeeze() # Ensure grayscale
        hog_feat = hog(img,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys')
        features.append(hog_feat)
    return np.array(features)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

print("Feature shapes:")
print("X_train_hog:", X_train_hog.shape)
print("X_test_hog :", X_test_hog.shape)

# ===== STEP 3: Train KNN with k=3 =====
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train_hog, y_train)

# ===== STEP 4: Predict =====
y_train_pred = knn.predict(X_train_hog)
y_test_pred = knn.predict(X_test_hog)

# ===== STEP 5: Evaluate Model =====
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f" Training Accuracy: {train_acc*100:.2f}%")
print(f" Testing Accuracy : {test_acc*100:.2f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=4))

# ===== STEP 6: Confusion Matrix =====
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix (HOG Features, k=3)')
plt.show()

# ===== STEP 7: Visualize Predictions =====
fig, axes = plt.subplots(2, 5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap='gray')
    ax.set_title(f"True: {y_test[i]}\nPred: {y_test_pred[i]}")
    ax.axis('off')
plt.show()

# ===== STEP 8: ROC AUC Curves (One-vs-Rest) =====
# Binarize the output for multi-class ROC
classes = np.unique(y_train)
y_test_bin = label_binarize(y_test, classes=classes)
y_score = knn.predict_proba(X_test_hog)

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
plt.title('ROC Curves for KNN (HOG Features, k=3)')
plt.legend(loc="lower right")
plt.show()
