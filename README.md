# Generalising EEG-Based Seizure Prediction 
**A Cross-Patient and Cross-Dataset Study Using CHB-MIT and Siena Scalp EEG**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![MNE](https://img.shields.io/badge/MNE--Python-1.0%2B-green)

## 📌 Executive Summary
This repository contains the codebase for predicting epileptic seizures up to 25 minutes in advance using raw EEG data[cite: 1]. 

Unlike many academic studies that evaluate on heavily overlapping patient-specific splits, this project enforces strict **Leave-One-Patient-Out (LOPO) cross-validation** to test true patient-independent generalization[cite: 1]. Furthermore, it evaluates **zero-shot cross-dataset transfer**, training a 1D-CNN on the CHB-MIT dataset (Boston) and evaluating it on the unseen Siena Scalp EEG dataset (Italy) to measure the impact of clinical domain shift[cite: 1].

**Key Technical Achievements:**
* **Memory-Safe Data Pipeline:** Engineered a highly efficient `.npz` caching and chunking mechanism to process over 31,000 EEG windows across two datasets without OOM errors[cite: 1].
* **Domain Harmonization:** Standardized varying sampling rates (256 Hz vs 512 Hz) and mapped disparate 10-20 electrode configurations into a unified 16-channel input space[cite: 1].
* **Rigorous Evaluation:** Bounded class imbalance (max 80 interictal windows per file) and enforced a strict 2-hour interictal exclusion zone to prevent data leakage[cite: 1].

## 🏗️ Repository Architecture

```text
eeg-seizure-prediction/
├── data/                       # Directory for raw .edf and processed .npz chunks[cite: 1]
├── notebooks/                  # EDA and exploratory visualization[cite: 1]
├── src/                        # Core Python library[cite: 1]
│   ├── config.py               # Centralized hyperparameters & paths[cite: 1]
│   ├── data_utils.py           # Preprocessing, filtering, and windowing logic[cite: 1]
│   ├── dataset.py              # PyTorch Dataset with memory-mapped caching[cite: 1]
│   ├── models/cnn_1d.py        # 1D Convolutional Neural Network architecture[cite: 1]
│   └── evaluation.py           # Metrics and LOPO loop logic[cite: 1]
├── scripts/                    # Entry points for pipelines[cite: 1]
│   ├── 01_preprocess.py       
│   └── 02_train_cnn.py        
├── requirements.txt           [cite: 1]
└── README.md                  [cite: 1]"# eeg-seizure-prediction" 
