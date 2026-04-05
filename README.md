# MSDNet: Multi-Scale Dense Network

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation and evaluation of **MSDNet** (Multi-Scale Dense Network) for efficient image classification. This repository features a **custom scratch implementation** compared against an **official-style reference**, focusing on anytime prediction and budgeted batch classification.



---

## 📖 Overview

Deep neural networks often require massive computational budgets. However, not all samples are equally difficult; "easy" samples can be classified by shallow layers, while "hard" samples require deeper processing.

**MSDNet** solves this by integrating three core concepts:
1.  **Multi-scale feature maps:** Fine scales preserve local detail, while coarse scales provide semantic information early on.
2.  **Dense connectivity:** Inspired by DenseNet, this ensures maximum feature reuse and stable gradient flow.
3.  **Early-exit classifiers:** Intermediate classifiers allow "easy" samples to exit the network early, saving computation.

### Inference Modes
* **Anytime Prediction:** The model can be interrupted at any time and return its most recent (best) prediction.
* **Budgeted Batch Classification:** Samples exit early based on a confidence threshold, optimizing the accuracy-to-compute ratio.

---

## 📂 Repository Structure

```text
dl-3rd-assignment/
├── official_msd/           # Official-style reference implementation
│   ├── official_msdnet_implementation.ipynb
│   ├── official_msdnet_best.pth
│   └── picture_file/       # Training and evaluation plots
├── scratch_msd/            # Custom implementation from the ground up
│   ├── scratch_msdnet_implementation.ipynb
│   ├── scratch_msdnet_best.pth
│   └── picture_file/       # Results for the scratch version
├── report/                 # Detailed project documentation
│   ├── report.docx
│   └── report.pdf
├──  README.md
└──  requirement.txt
```

---

## 🏗️ Architecture & Implementation

### 1. Multi-Scale Strategy
Unlike standard CNNs, MSDNet maintains multiple spatial resolutions simultaneously. This allows even the earliest classifiers to access coarse, high-level features.

### 2. Dense Connectivity
Each layer receives input from all preceding layers at the same scale, promoting feature reuse and parameter efficiency.

### 3. Early-Exit Logic
During inference, if the Softmax confidence at an intermediate exit exceeds a threshold ($H$), the prediction is returned immediately:
$$\text{Confidence}(x) = \max \text{Softmax}(f_i(x)) \ge H$$

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch & Torchvision
* NumPy, Matplotlib, Jupyter

### Installation
```bash
git clone https://github.com/your-username/msdnet-project.git
cd msdnet-project
pip install -r requirements.txt
```

### Usage
Launch Jupyter and explore the implementations:
* **Scratch Performance:** `scratch_msd/scratch_msdnet_implementation.ipynb`
* **Official Performance:** `official_msd/official_msdnet_implementation.ipynb`

---

## 📊 Results & Evaluation

Our experiments demonstrate the classic accuracy-efficiency trade-off. By adjusting the confidence threshold, we can navigate the Pareto front between FLOPs and Top-1 Accuracy.

| Metric | Scratch Implementation | Official Reference |
| :--- | :---: | :---: |
| **Training Stability** | Stable (Cosine LR) | Stable (Cosine LR) |
| **Early-Exit Behavior** | Functional | Optimized |
| **FLOP Efficiency** | Moderate | High |

> [!TIP]
> **Key Observation:** Lower thresholds lead to more early exits and lower FLOPs, while higher thresholds prioritize accuracy at the cost of computation.

---

## 🛠️ Training Setup

| Component | Details |
| :--- | :--- |
| **Loss Function** | Cross-entropy with Label Smoothing |
| **Optimizer** | SGD with Cosine Annealing |
| **Supervision** | Multi-exit (Independent loss per classifier) |
| **Metrics** | FLOPs per sample, Exit Distribution, Top-1 Accuracy |

---

## 🔮 Future Work
* **Distillation:** Implementing knowledge distillation between deep and shallow exits.
* **Dynamic Thresholding:** Automatically calibrating thresholds based on a target FLOP budget.
* **Edge Deployment:** Benchmarking on mobile hardware (e.g., CoreML or ONNX).

---

## 📜 References
* **ChatGPT / Claude:**— used for code assistance and report writing
* **Huang et al. (2018):** [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/abs/1703.09844). *ICLR*.

---
*Created for the Deep Learning 3rd Assignment.*
