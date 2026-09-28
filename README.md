# Federated Learning for Privacy-Preserving Uterine Fibroid Classification from Ultrasound Images

> Using Lightweight Convolutional Neural Networks

This repository contains the full implementation of my undergraduate thesis, which investigates **Federated Learning (FL)** as a privacy-preserving alternative to centralized training for classifying uterine fibroids from ultrasound images.

---

## Overview

Uterine fibroids are among the most common benign gynecological tumors, affecting an estimated 20–25% of women over the age of 30. Ultrasound imaging is the standard first-line diagnostic tool, but manual interpretation is subjective and sonographer-dependent — motivating automated, deep-learning-based classification.

Prior work on this task reports strong accuracy (up to 99.5%), but every existing study assumes **centralized training**, where patient data from multiple hospitals is pooled onto a single server — an assumption that conflicts with real-world data-privacy regulations. This project asks a different question:

> **Can a lightweight CNN classifier be trained under Federated Learning — without patient data ever leaving its originating hospital — while retaining performance close to a centralized model?**

The project is carried out in **two stages**:

1. **Centralized comparison** — four lightweight CNN backbones are compared under two preprocessing strategies to find the best-performing configuration.
2. **Federated Learning** — the best configuration is adopted and trained across three simulated hospital clients, under both homogeneous (IID) and realistic, heterogeneous (Non-IID) data distributions, and compared against a centralized baseline.

---

## Objectives

- Compare four lightweight, ImageNet-pretrained CNN backbones — **EfficientNetB0**, **MobileNetV2**, **MobileNetV3Small**, **NASNetMobile** — under **Full-Image** vs. **Center-ROI** preprocessing.
- Identify the best centralized configuration using accuracy, sensitivity, specificity, precision, F1-score, and AUC.
- Design and implement a **Federated Learning simulation** (3 clients, FedAvg) using the best configuration as the backbone.
- Evaluate the federated model under both **IID** and **Non-IID** (Dirichlet-sampled) client data distributions.
- Quantify the performance trade-off — the *"cost of federation"* and *"cost of heterogeneity"* — relative to a centralized baseline.

---

## Dataset

Ultrasound images labeled into two classes:

| Class | Description |
|---|---|
| **NUF** | Non-Uterine Fibroid |
| **UF** | Uterine Fibroid |

| Split | NUF | UF | Total |
|---|---|---|---|
| Train | 758 | 596 | 1,354 |
| Validation | 134 | 106 | 240 |
| Test | 223 | 173 | 396 |

All images are resized to `224×224`. The train/validation/test split is performed **before** any augmentation, to avoid data leakage.

---

## Methodology

### Stage 1 — Centralized Comparison
- **Preprocessing:** Full-Image vs. Center-ROI (asymmetric crop excluding on-screen calipers/text)
- **Augmentation:** random flip, rotation, zoom, contrast, translation (training split only)
- **Backbones:** EfficientNetB0, MobileNetV2, MobileNetV3Small, NASNetMobile — all frozen, ImageNet-pretrained
- **Classification head:** GAP → BatchNorm → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(2, Softmax)
- **Training:** Adam optimizer, categorical cross-entropy, EarlyStopping + ReduceLROnPlateau
- 8 total configurations (4 backbones × 2 preprocessing strategies) evaluated

Two additional techniques — **CLAHE preprocessing** and a **channel-and-spatial attention module** — were also tested, but excluded from the final pipeline after they were found to *reduce* performance. This negative result is reported explicitly rather than omitted.

### Stage 2 — Federated Learning
- **Framework:** PyTorch (re-implementation of the best centralized config)
- **Clients:** 3 simulated hospitals
- **Partitioning:**
  - **IID** — equal class ratio per client
  - **Non-IID** — Dirichlet sampling (α = 0.5), simulating realistic hospital heterogeneity
- **Local training:** 3 epochs/round, Adam + gradient clipping
- **Aggregation:** Federated Averaging (FedAvg), sample-size-weighted
- **Rounds:** 10 communication rounds (30 epochs-equivalent — matched to the centralized training budget)
- **Baseline:** an equivalent centralized model (same backbone, same budget) trained in PyTorch for fair, controlled comparison

$$w_{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{N} \, w_k$$

---

## Results

### Best Centralized Configuration

**MobileNetV3Small + Full-Image** outperformed all other 7 configurations:

| Metric | Value |
|---|---|
| Accuracy | **90.91%** |
| Sensitivity | **98.27%** |
| Specificity | 85.20% |
| Precision | 83.74% |
| F1-Score | 90.43% |
| AUC | **0.9800** |

Full-Image preprocessing outperformed Center-ROI cropping on **every metric, for all four backbones** (8/8 comparisons) — contrary to the initial hypothesis that cropping out peripheral artifacts would help.

### Federated vs. Centralized Performance

| Setting | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Centralized (baseline) | 90.91% | 93.64% | 88.79% | 0.9718 |
| **FL-IID** (3 clients) | **92.68%** | 91.33% | 93.72% | **0.9727** |
| **FL-Non-IID** (3 clients) | 90.66% | 91.91% | 89.69% | 0.9609 |

- FL-IID slightly **exceeds** the centralized baseline — no meaningful "cost of federation" under homogeneous client data.
- FL-Non-IID stays within **0.25 percentage points** of the centralized accuracy, despite deliberately skewed, heterogeneous clients.
- Sensitivity — the metric most critical for avoiding missed diagnoses — stays **above 91%** in every federated setting.

**Takeaway:** Federated Learning retains diagnostic performance close to centralized training, without any patient ultrasound image ever leaving its originating client.

---

## Future Work

- Detection-based fibroid localization (e.g. YOLO) using annotated bounding-box data
- Explainable AI (XAI) to visualize model decision regions
- More clients / communication rounds; FedProx or similar techniques to further mitigate Non-IID degradation
- Differential privacy for formal, quantifiable privacy guarantees

---
