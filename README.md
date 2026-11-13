# DIDA
Official PyTorch implementation of "Diverse Identity-Preserving Diffusion Augmentation for Visible-Infrared Person Re-Identification"
# Diverse Identity-Preserving Diffusion Augmentation for Visible-Infrared Person Re-Identification (DIDA)

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/abs/YOUR_PAPER_ID)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12%2B-orange.svg)](https://pytorch.org/)

</div>

This repository is the official PyTorch implementation of the paper **"Diverse Identity-Preserving Diffusion Augmentation for Visible-Infrared Person Re-Identification"**.

---

## 💡 Abstract

Visible-Infrared Person Re-Identification (VI-ReID) aims to match person identities across visible (RGB) and infrared (IR) camera views. Due to significant cross-modality discrepancies and limited intra-identity diversity, retrieving accurate matches remains challenging. To address these issues, we propose **DIDA** (Diverse Identity-Preserving Diffusion Augmentation), a novel framework that leverages diffusion-based generative models to generate high-fidelity, identity-consistent augmented samples. DIDA effectively mitigates cross-modality gaps and achieves state-of-the-art performance across standard VI-ReID benchmarks.

---

## 🛠️ Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU + CUDA 11.3+
- **Python**: 3.8+
- **PyTorch**: 1.12+

### Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/amslqo180/DIDA.git
cd DIDA
pip install -r requirements.txt
```

---

## 📂 Datasets & Downloads

### 1. Benchmark Datasets
Please download the official original datasets from their respective authors:
- **SYSU-MM01**
- **LLCM**

### 2. Generated Augmented Datasets (Google Drive)
We provide our pre-generated identity-preserving augmented datasets via Google Drive for quick training and reproduction:

| Dataset Name | Target Benchmark | Google Drive Download Link |
| :--- | :---: | :---: |
| **DIDA-SYSU-Aug** | SYSU-MM01 | [[Download Dataset 1](https://drive.google.com/drive/folders/YOUR_GOOGLE_DRIVE_LINK_DATASET_1](https://drive.google.com/file/d/1Ya6vXFzpbcGLTTlYAYSRdCoYZVGoAdEm/view?usp=sharing)) |
| **DIDA-LLCM-Aug** | LLCM | [Download Dataset 2]([https://drive.google.com/drive/folders/YOUR_GOOGLE_DRIVE_LINK_DATASET_2](https://drive.google.com/file/d/1Lj30TnC_8Sx_KM-P6MVDb4rENLxk4n38/view?usp=sharing)) |

> **Note**: Extract the downloaded archives into your specified `./data/` directory and ensure paths are updated in your configuration files.

---

## 📦 Pretrained Models & Benchmark Results

You can download our trained model checkpoints from Google Drive to verify or reproduce the reported performance:

| Model Name | Evaluation Benchmark | Rank-1 (%) | mAP (%) | Checkpoint Link |
| :--- | :---: | :---: | :---: | :---: |
| **SYSU_Market** | SYSU-MM01 | -- | -- | [Download Weights]([https://drive.google.com/file/d/YOUR_GOOGLE_DRIVE_LINK_SYSU_MARKET](https://drive.google.com/drive/folders/1oix0nfvoRyjz_UFW22sO80DhE93t9W8b?usp=sharing)) |
| **LLCM_Market** | LLCM | -- | -- | [Download Weights]([https://drive.google.com/file/d/YOUR_GOOGLE_DRIVE_LINK_LLCM_MARKET](https://drive.google.com/drive/folders/1I4TtgLRgUx_smBfR_wWQQtYhGkJYcirA?usp=sharing)) |

---

## 🚀 Usage Guide

### 1. Training

To train DIDA from scratch on the **SYSU-MM01** or **LLCM** dataset, run:

```bash
# Train on SYSU-MM01 dataset
python train.py --dataset sysu --data-path ./data/sysu --batch-size 64 --gpu 0

# Train on LLCM dataset
python train.py --dataset llcm --data-path ./data/llcm --batch-size 64 --gpu 0
```

### 2. Evaluation / Testing

To evaluate pretrained weights (e.g., `SYSU_Market` or `LLCM_Market`), run:

```bash
# Evaluate SYSU_Market checkpoint
python test.py --dataset sysu --resume path/to/SYSU_Market.pth --gpu 0

# Evaluate LLCM_Market checkpoint
python test.py --dataset llcm --resume path/to/LLCM_Market.pth --gpu 0
```

---

## 📜 Citation

If you find this work, code, or generated datasets helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{dida2026,
  title={Diverse Identity-Preserving Diffusion Augmentation for Visible-Infrared Person Re-Identification},
  author={Your Name and Co-authors},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2026}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
