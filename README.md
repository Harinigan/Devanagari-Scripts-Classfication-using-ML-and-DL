# Performance Comparison of Machine Learning Models for Handwritten Devanagari Numerals Classification

## Overview

This project presents a comparative study of machine learning and deep
learning models for automated **Handwritten Devanagari Numerals**
classification. It evaluates five models — **K-Nearest Neighbours (K-NN)**,
**Support Vector Machine (SVM)**, a custom **Convolutional Neural Network
(CNN)**, **VGG-16**, **GoogLeNet (Inception v1)**, and **ResNet-50** — to
determine which architecture, and how much model complexity, is best suited
for classifying digits in the Devanagari script.

The proposed lightweight CNN outperforms the deeper GoogLeNet and ResNet-50
models, achieving **99.522% test accuracy** and an **F1-score of 0.9978**,
the best reported result on this dataset combination.

## Features

-   Automated handwritten Devanagari numeral classification (0-9)
-   Comparison of 6 machine learning / deep learning models
-   HOG feature extraction for K-NN
-   Data augmentation (rotation, shift, zoom) for CNN
-   Transfer learning with pretrained VGG-16, GoogLeNet, and ResNet-50
-   K-Fold cross-validation for the proposed CNN
-   Evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and
    Confusion Matrix

## Digit Classes

-   0
-   1
-   2
-   3
-   4
-   5
-   6
-   7
-   8
-   9

## Tech Stack

-   Python
-   TensorFlow / Keras
-   PyTorch
-   OpenCV
-   NumPy
-   Scikit-learn
-   Scikit-image
-   Matplotlib / Seaborn
-   VGG-16
-   GoogLeNet (Inception v1)
-   ResNet-50

## Project Structure

``` text
Dataset/
├── nhcd/                # Dataset 1 (Pant et al.)
└── Images/              # Dataset 2 (Acharya et al.)
Processed_Data/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
Preprpcessing.py
KNN.py
SVM.py
CNN.py
VGG.py
GoogleNet.py
ResNet.py
README.md
```

## Workflow

1.  Load handwritten Devanagari numeral images from both datasets.
2.  Convert images to grayscale and resize.
3.  Normalize pixel values.
4.  Merge and shuffle the two datasets.
5.  Split into train and test sets.
6.  Apply data augmentation (rotation, width/height shift, zoom).
7.  Train K-NN, SVM, CNN, VGG-16, GoogLeNet, and ResNet-50 models.
8.  Evaluate and compare all models on the same test set.

## Installation

``` bash
git clone https://github.com/your-username/devanagari-numerals-classification.git
cd devanagari-numerals-classification
pip install -r requirements.txt
```

## Run

``` bash
python Preprpcessing.py
python KNN.py
python SVM.py
python CNN.py
python VGG.py
python GoogleNet.py
python ResNet.py
```

## Results

| Model      | Testing Accuracy | Precision | Recall | F1-Score |
|------------|-------------------|-----------|--------|----------|
| K-NN       | 98.266%           | 0.98      | 0.98   | 0.98     |
| SVM        | 99.266%           | 0.99      | 0.99   | 0.99     |
| **CNN (proposed)** | **99.522%** | **0.998** | **0.998** | **0.998** |
| VGG-16     | 94.059%           | 0.976     | 0.950  | 0.963    |
| GoogLeNet  | 97.469%           | 0.992     | 0.990  | 0.991    |
| ResNet-50  | 95.256%           | 0.987     | 0.984  | 0.985    |

## Applications

-   Optical Character Recognition (OCR) for Indian regional languages
-   ID card recognition (Aadhar, PAN cards, Passports)
-   Bank cheque and physical form processing
-   Digitization of institutional and government records
-   Document scanning systems

## Future Enhancements

-   Fine-tuning and hyperparameter optimization for ResNet-50 and GoogLeNet
-   Transfer learning from larger, domain-relevant datasets
-   Ensembling multiple models for higher accuracy
-   Explainable AI (Grad-CAM / feature visualization)
-   Extension to full Devanagari character recognition (beyond numerals)
-   Real-time recognition via live video feed

## Authors

**Agastya Gummaraju**, **Ajitha K. B. Shenoy**, **Smitha N. Pai**

Department of Information and Communication Technology, Manipal Institute
of Technology, Manipal Academy of Higher Education

## License

This project is licensed under a [Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0
License](https://creativecommons.org/licenses/by-nc-nd/4.0/), intended for
educational and research purposes.
