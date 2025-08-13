# 🖼️ Image Recognition – 5 Days Bootcamp Project

## 📌 Overview
This repository contains my complete 5-day hands-on deep learning bootcamp project on **Image Recognition**, built entirely in **Google Colab** using **Python**, **TensorFlow/Keras**, and **Kaggle datasets**.  
The project walks through the **end-to-end image classification workflow** — from data preprocessing to model deployment — using both **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.

## 🎯 Objectives
- Understand and preprocess image datasets.
- Build and train CNN models from scratch.
- Apply advanced techniques such as **Data Augmentation** and **Batch Normalization**.
- Use **Transfer Learning** with pre-trained models.
- Evaluate models with multiple metrics and visualize performance.
- Deploy and predict on new images.

## 📅 Day-by-Day Progress

### **Day 1 – Data Collection & Preprocessing**
- Learned the basics of **Image Recognition**, **Deep Learning**, and datasets.
- Integrated Kaggle API in Colab to download datasets.
- Loaded **MNIST** (handwritten digits) and **CIFAR-10** (object classification).
- Preprocessed:
  - Normalized pixel values (0–1)
  - Reshaped data for CNN input
- Visualized datasets (MNIST digit grid, CIFAR-10 samples).

---

### **Day 2 – Building My First CNN**
- Designed a CNN with:
  - `Conv2D` and `MaxPooling2D` layers
  - Dense fully connected layers
  - Dropout for overfitting prevention
- Trained on **MNIST** for 5 epochs.
- Achieved **~99% accuracy** on MNIST test set.
- Plotted **training vs. validation accuracy**.

---

### **Day 3 – Advanced CNN Techniques**
- Moved to **CIFAR-10** dataset (more challenging due to color and complexity).
- Applied **Data Augmentation** (rotation, shift, flip).
- Added **Batch Normalization** for stable training.
- Evaluated using:
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
- Improved model generalization on CIFAR-10.

---

### **Day 4 – Transfer Learning**
- Introduced pre-trained models: **MobileNetV2**, **ResNet**, **VGG** (ImageNet weights).
- Used **MobileNetV2** for **Cats vs Dogs** classification.
- Steps:
  - Feature extraction with frozen base layers.
  - Fine-tuning with low learning rate.
- Achieved **~84% accuracy** after fine-tuning.
- Generated **ROC curve** for binary classification.

---

### **Day 5 – Predictions & Final Results**
- Predicted on **new uploaded images**.
- Compared model performance across datasets.
- Created bar chart of accuracies for portfolio.
- Discussed deployment readiness and future improvements.

---

## 📊 Final Results

| Dataset      | Accuracy |
|--------------|----------|
| MNIST        | 0.98     |
| CIFAR-10     | 0.75     |
| Cats vs Dogs | 0.84     |

---

## 🛠️ Tech Stack
- **Programming Language:** Python
- **Frameworks & Libraries:** TensorFlow, Keras, Matplotlib, Scikit-learn, Seaborn
- **Environment:** Google Colab
- **Data Source:** Kaggle Datasets (MNIST, CIFAR-10, Cats vs Dogs)

---
