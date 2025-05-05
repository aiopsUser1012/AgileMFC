# AgileMFC

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.10.0-orange)](https://pytorch.org/) [![Python](https://img.shields.io/badge/Python-%3E%3D3.8.6-green)](https://www.python.org/)

This repository implements **AgileMFC**, a novel framework for failure classification in microservice systems using multi-modal monitoring data (logs, metrics, traces). The method combines modality-aware representation learning with a two-stage training strategy, achieving SOTA performance on three real-world datasets. For details, refer to our paper:
**"To Split or to Merge? Exploring Multi-modal Data Flexibly for Failure Classification in Microservices"** (Internetware '25).

## 📖 Table of Contents
- [✨ Features](#-features)
- [⚙️ Installation](#-installation)
- [📁 Datasets](#-datasets)
- [🚀 Quick Start](#-quick-start)
- [📈 Results](#-results)
- [📚 Citation](#-citation)

## ✨ Features
- **🔀 Multi-modal Fusion**: Processes logs, metrics, and traces using modality-specific expert networks.
- **🎛️ Flexible Gating Mechanism**: Combines modality-specific and shared gates for balanced feature integration.
- **📚 Two-Stage Training**: Decouples representation learning and classifier training for improved generalization.
- **🤖 Transformer-Based Experts**: Leverages self-attention to capture failure-sensitive patterns.
- **🔁 Reproducibility**: Supports full replication of experiments on GAIA, TrainTicket, and SocialNetwork datasets.

## ⚙️ Installation
### Prerequisites
- Python 3.8+
- CUDA 11.3 (for GPU acceleration)
- Conda (recommended)

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/aiopsUser1012/AgileMFC.git
    cd AgileMFC
    ```

2. Create a Conda environment and install dependencies:
    ```bash
    conda env create -f env.yml
    conda activate agilemfc
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

## 📁 Datasets
We evaluate AgileMFC on three datasets. Follow these steps to prepare the data:

| Dataset | Description | Source |
|:---|:---|:---|
| **GAIA** | Simulated environment with 5 failure types (56M metrics, 87M logs, 2.3M traces) | [Download](https://github.com/CloudWise-OpenSource/GAIA-DataSet) |
| **TrainTicket** | Ticketing system with 3 failure types (127M metrics, 6.7M logs, 48K traces) | [Download](https://zenodo.org/records/7615394) |
| **SocialNetwork** | Social platform with 3 failure types (2.4M metrics, 1.4M logs, 126K traces) | [Download](https://zenodo.org/records/7615394) |

## 🚀 Quick Start
Run a demo on the GAIA dataset:
```
python main.py --dataset gaia
```

This will:
1. ✅ Preprocess the data (if not already done).
2. 🧠 Train the representation network.
3. 🎓 Train the classifier and evaluate performance.

## 📈 Results

AgileMFC achieves the following performance (weighted F1-score):

| Dataset | Precision | Recall | F1-score | Improvement over SOTA |
|:---|:---|:---|:---|:---|
| **GAIA** | 0.958 | 0.924 | 0.939 | 🚀 **+ 12.4%** |
| **TrainTicket** | 0.751 | 0.720 | 0.728 | 🚀 **+ 9.1%** |
| **SocialNetwork** | 0.872 | 0.620 | 0.701 | 🚀 **+ 11.8%** |

## 📚 Citation
```bibtex
@inproceedings{tan2025agilemfc,
    author = {Tan, Xiuhong and Yuan, Yuan and Zhou, Tongqing and He, Shiming and Li, Yuqi and Zhang, Jian},
    title = {To Split or to Merge? Exploring Multi-modal Data Flexibly for Failure Classification in Microservices},
    year = {2025},
    booktitle = {Proceedings of the 16th Asia-Pacific Symposium on Internetware},
    series = {Internetware '25}
}
```
