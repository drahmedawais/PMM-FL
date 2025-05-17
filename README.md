# PMM-FL

Public repository of the proposed work is planned to maintain to facilitate reproducibility and further research. The manuscript is under peer-review at *****. All material will be available for community.

# Personalized Multi-Modal Federated Learning (PMM-FL) for Skin Cancer Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)

A federated learning framework for privacy-preserving multi-modal skin cancer diagnosis, addressing:
- Modality heterogeneity (images + clinical data)
- Missing data imputation
- Client-specific personalization

## Key Features
- **Cross-modal knowledge transfer** via attention-based fusion
- **Learnable missing modality imputation** (68D tabular features)
- **Personalized FedProx** (μ=0.01) with client-specific BN
- **Statistical robustness**: p<0.05 on all critical metrics (Wilcoxon signed-rank)

## Architecture
![Framework Diagram](docs/architecture.png)
*Components:*
1. ResNet18/50 image encoder (512D features)
2. 3-layer MLP tabular encoder (512D)
3. Multi-head cross-modal attention (4 heads)
4. Adaptive output layers per client

## Installation
```bash
conda create -n pmmfl python=3.8
conda activate pmmfl
pip install -r requirements.txt
