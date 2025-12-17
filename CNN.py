

# CNN with Data Augmentation for Digit Classification
# ===========================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# ===== STEP 1: Load Dataset =====
X_train = np.load("/content/processed_data/X_train.npy")
y_train = np.load("/content/processed_data/y_train.npy")
X_test = np.load("/content/processed_data/X_test.npy")
y_test = np.load("/content/processed_data/y_test.npy")

print("Dataset shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test :", X_test.shape, "y_test :", y_test.shape)

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# ===== STEP 2: Data Augmentation =====
datagen = ImageDataGenerator(
    rotation_range=10, # random rotation
    width_shift_range=0.1, # horizontal shift
    height_shift_range=0.1, # vertical shift
    zoom_range=0.1 # zoom
)
datagen.fit(X_train)

# ===== STEP 3: Define CNN Model =====
model = Sequential([
    Input(shape=(32, 32, 1)),

    # Conv block 1
    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Conv block 2
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Fully connected layers
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# ===== STEP 4: Compile Model =====
model.compile(optimizer=Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ===== STEP 5: Callbacks =====
lr_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-6)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ===== STEP 6: Train Model with Augmentation =====
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[lr_reduction, early_stop],
    verbose=1
)

# ===== STEP 7: Evaluate Model =====
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n Testing Accuracy: {test_acc*100:.2f}%")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix with Augmentation")
plt.show()

# ===== STEP 8: Plot Training History =====
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy (Augmentation)")

# Loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss (Augmentation)")

plt.show()

# ===== STEP 9: Visualize Predictions =====
fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].squeeze(), cmap="gray")
    ax.set_title(f"True: {y_test[i]}\nPred: {y_pred_classes[i]}")
    ax.axis("off")
plt.show()
