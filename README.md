# Communication-Efficient Federated Learning via Clipped Uniform Quantization

This repository contains the code and resources to replicate the simulations presented in the paper:  
**"Communication-Efficient Federated Learning via Clipped Uniform Quantization"**.  
The method integrates clipped uniform quantization with federated learning to enhance communication efficiency without sacrificing accuracy.

---

## Links to the Paper
- Available at: [arXiv](https://arxiv.org/abs/2405.13365)


---

## Abstract
This work presents a novel framework to reduce communication costs in federated learning using clipped uniform quantization. The key contributions include:
- **Optimal Clipping Thresholds**: Balances quantization and clipping noise to minimize information loss.
- **Stochastic Quantization**: Enhances robustness by introducing diversity in client model initialization.
- **Privacy Preservation**: Obviates the need for disclosing client-specific dataset sizes during aggregation.

The proposed method achieves near-full-precision accuracy with significant communication savings, demonstrated through extensive simulations on the MNIST and CIFAR-10 datasets.

---

## Key Features
- **Enhanced Communication Efficiency**:
  - Optimal clipping of model weights before transmission.
  - Stochastic quantization for increased robustness.
- **Privacy-Aware Design**:
  - Avoids sharing dataset sizes with the server.
- **Versatile Aggregation Methods**:
  - Supports both FedAvg and error-weighted aggregation.

---

## Repository Contents
- **Code for MNIST and CIFAR-10 Simulations**:
  - Implements various quantization configurations (e.g., "4-2-2-4", "2-2-2-2").
  - Includes training scripts and evaluation metrics.
- **Clipping Threshold Optimization**:
  - Analytical method for optimal threshold selection.
- **Quantization Strategies**:
  - Deterministic and stochastic quantization examples.

---

## Highlights
- **Communication Savings**: Reduces communication costs.
- **Scalability**
- **Performance Metrics**

---

## Getting Started
### Prerequisites
- Python 3.7 or later
- Required libraries (see `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/ClippedQuantFL.git
   cd ClippedQuantFL
---

### Citation
If you find this repository helpful, please consider citing:

```bibtex
@article{bozorgasl2024clippedquantfl,
  author  = {Zavareh Bozorgasl and Hao Chen},
  title   = {Communication-Efficient Federated Learning via Clipped Uniform Quantization},
  journal = {arXiv preprint arXiv:2405.13365},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.13365}
}

