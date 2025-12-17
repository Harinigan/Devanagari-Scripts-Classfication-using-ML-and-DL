

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# ===== Dataset Paths =====
dataset_17 = r"/content/drive/MyDrive/devnagari_/nhcd"  # [17]
dataset_19 = r"/content/drive/MyDrive/devnagari_/Images"  # [19]

# ===== Preprocessing Function =====
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {img_path}")
        return None
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return img

# ===== Load and Split Dataset [19] =====
def split_dataset_19(dataset_path, test_size_per_class=300):
    X_train, y_train, X_test, y_test = [], [], [], []
    for folder_name in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder):
            continue
        label_str = folder_name.replace("digit_", "")
        if not label_str.isdigit():
            continue
        label = int(label_str)
        img_files = sorted(os.listdir(folder))

        # Split into test (first 300) and train (remaining)
        test_files = img_files[:test_size_per_class]
        train_files = img_files[test_size_per_class:]

        # Load training images
        for img_file in train_files:
            img_path = os.path.join(folder, img_file)
            img = preprocess_image(img_path)
            if img is not None:
                X_train.append(img)
                y_train.append(label)

        # Load testing images
        for img_file in test_files:
            img_path = os.path.join(folder, img_file)
            img = preprocess_image(img_path)
            if img is not None:
                X_test.append(img)
                y_test.append(label)

    return (
        np.array(X_train)[..., np.newaxis],
        np.array(y_train),
        np.array(X_test)[..., np.newaxis],
        np.array(y_test)
    )

# ===== Load Dataset [17] (Training Only) =====
def load_dataset_17(dataset_path):
    X, y = [], []
    for folder_name in os.listdir(dataset_path):
        folder = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder):
            continue
        label_str = folder_name.replace("digit_", "")
        if not label_str.isdigit():
            continue
        label = int(label_str)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(label)
    return np.array(X)[..., np.newaxis], np.array(y)

# ===== Load Data =====
print("Loading datasets... please wait 🕐")
X17, y17 = load_dataset_17(dataset_17)
X19_train, y19_train, X_test, y_test = split_dataset_19(dataset_19, test_size_per_class=300)

# Merge [17] + [19] for training
X_train = np.concatenate((X17, X19_train), axis=0)
y_train = np.concatenate((y17, y19_train), axis=0)

# Shuffle both
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

print(f"Training images: {X_train.shape[0]}")
print(f"Testing images: {X_test.shape[0]} (≈300 per class expected)")

# ===== Data Augmentation =====
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)


# ===== Save for Later Use (Inside a Folder) =====
save_dir = "/content/drive/MyDrive/data"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "X_train.npy"), X_train)
np.save(os.path.join(save_dir, "y_train.npy"), y_train)
np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)

print(f"\n All arrays saved inside: {save_dir}")

