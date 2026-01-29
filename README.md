# RealWaste Classification

A **Convolutional Neural Network (CNN)** based waste image classification project using the **RealWaste dataset**.  
This repository implements both a **custom CNN** and **transfer learning** with pre-trained models (VGG16, ResNet50, etc.) for **multi-class waste classification**.

---

## Overview

This project classifies real-world waste images into **9 categories**:

- Cardboard  
- Food Organics  
- Glass  
- Metal  
- Miscellaneous Trash  
- Paper  
- Plastic  
- Textile Trash  
- Vegetation  

### Features
- Data preprocessing and dataset splitting  
- Custom CNN implementation  
- Fine-tuning of pre-trained models (**VGG16, ResNet50**)  
- Training/validation curves, confusion matrices and detailed performance metrics  
- Full experimentation in **Jupyter Notebook** format  

The code is written in **PyTorch** and runs on **GPU** (tested on Kaggle with CUDA).

---


## Dataset

**RealWaste Dataset**  
Source:  
📌 **UCI Machine Learning Repository – RealWaste**: https://archive.ics.uci.edu/dataset/908/realwaste

- Total images: ~4,752 across 9 classes  
- Moderately imbalanced classes  
  - Plastic: 921  
  - Textile Trash: 318  

### Dataset Handling
- Dataset copied into the working directory during notebook execution  
- Split ratio:
  - 70% Training  
  - 15% Validation  
  - 15% Testing  
- Splitting method: `split-folders`  

### Data Augmentation
- Random flips  
- Random rotations  
- Normalization  

Used to improve generalization.

---

## Techniques & Models

### 1. Custom CNN
- Architecture:
  - Multiple convolutional blocks  
  - ReLU activation  
  - Max pooling  
  - Dropout  
  - Fully connected layers  
- Optimizer experiments:
  - SGD  
  - SGD with momentum  
  - Adam  
- Learning rate scheduling  
- Early stopping  

---

### 2. Transfer Learning (Pre-trained Models)

Fine-tuned models from `torchvision.models`:

- **VGG16 (best performing)**
- ResNet50  
- Others experimented:
  - MobileNetV3  
  - EfficientNet  

#### Training Strategy
- Feature extractor layers frozen initially  
- Only classifier heads trained  
- Full fine-tuning performed later using **differential learning rates**

---

## Key Results

### VGG16 Fine-tuned Model (Best Performer)

- **Test Accuracy:** 90.0%  
- **Macro F1-Score:** 0.8987  
- **Weighted F1-Score:** 0.8962  

---

### Classification Report (VGG16)

| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| Cardboard            | 0.84      | 0.96   | 0.89     | 70      |
| Food Organics        | 0.97      | 0.97   | 0.97     | 63      |
| Glass                | 0.95      | 0.86   | 0.90     | 63      |
| Metal                | 0.88      | 0.89   | 0.89     | 119     |
| Miscellaneous Trash  | 0.78      | 0.83   | 0.80     | 75      |
| Paper                | 0.93      | 0.93   | 0.93     | 75      |
| Plastic              | 0.90      | 0.87   | 0.88     | 139     |
| Textile Trash        | 0.87      | 0.84   | 0.85     | 49      |
| Vegetation           | 1.00      | 0.94   | 0.97     | 66      |

---

### Overall Metrics

- **Macro Precision:** 0.9015  
- **Macro Recall:** 0.8978  
- **Weighted Precision:** 0.8983  
- **Weighted Recall:** 0.8957  

Training/validation loss curves, confusion matrices, and sample predictions are generated in the notebook.

---

## Repository Contents

- `realwaste.ipynb` – Main Kaggle notebook with full code, training, evaluation, and visualizations  
- `documentation.pdf` – Original assignment specification (EN3150 Assignment 03)  
- `.gitignore` – Standard ignores  
- `README.md` – This file  

---

## How to Run

1. Open the notebook on Kaggle (GPU enabled)
2. Attach the RealWaste dataset:  
3. Run all cells sequentially  

## Future Improvements

- **Advanced class imbalance handling**  
  Apply techniques such as class-weighted loss functions, focal loss, or data-level methods like oversampling and augmentation to improve performance on underrepresented classes.

- **Model ensembling**  
  Combine predictions from multiple high-performing models (e.g., VGG16, ResNet50, EfficientNet) to achieve more robust and generalized classification results.

- **Web-based deployment**  
  Deploy the trained model as an interactive web application using **Streamlit** or a lightweight web framework, enabling real-time waste image classification for end users.
