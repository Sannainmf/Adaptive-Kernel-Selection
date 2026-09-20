# Adaptive Kernel Selection for Convolutional Neural Networks

This repository contains the training scripts for the paper:

> "Adaptive Kernel Selection in Multi-Layer Convolutional Neural Networks" (accepted, IEEE ICMLA 2025)

## Overview

Standard CNNs fix the kernel size of every convolution at design time. We add an Adaptive Kernel Selection (AKS) module after each convolution that lets the network choose its receptive field per input. The module pools the incoming feature map, runs a small MLP to produce softmax weights over 3×3, 5×5, and 7×7 kernels, applies all three convolutions in parallel, and returns their weighted sum. We embed AKS into a 3-layer CNN, a 6-layer CNN, and VGG11, and compare each against its fixed-kernel baseline on CIFAR-10, CIFAR-100, and Fashion-MNIST. Every script trains both the adaptive model and the baseline over several seeds and saves the training curves and per-layer kernel-selection heatmaps.

## Repository Contents

```
3Layer_AKS/                        # 3 conv layers, each followed by an AKS module
├── C10.py                         # CIFAR-10
├── C100.py                        # CIFAR-100
└── FashionMNIST.py                # Fashion-MNIST

6Layer_AKS/                        # 6 conv layers, each followed by an AKS module
├── C10.py
├── C100.py
└── FashionMNIST.py

VGG11_AKS/                         # VGG11 (8 conv layers), each followed by an AKS module
├── C10.py
├── C100.py
└── FashionMNIST.py
```

Each script is self-contained: it defines the AKS module, the adaptive model, and the baseline, downloads the dataset through torchvision, and writes its outputs to a `results/` folder. Requires PyTorch, torchvision, NumPy, and Matplotlib.
