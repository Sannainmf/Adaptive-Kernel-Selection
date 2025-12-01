# Adaptive Kernel Selection for Multi-Scale CNN Feature Extraction

This repository contains code for 3 layer, 6 layer, and 8 layer (VGG11) CNN models across multiple datasets.

> "Adaptive Kernel Selection: Performance Pattern Analysis Across Network Depth Hierarchies and Dataset Complexities"

## Overview

We introduce Adaptive Kernel Selection (AKS), a novel attention-driven dynamic kernel selection architecture enabling CNNs to adaptively select optimal kernel configurations ({3×3, 5×5, 7×7}) for multi-scale feature extraction. This repository provides the experimental framework and trained models to reproduce our results and accelerate research in efficient dynamic convolution methods.

## Repository Contents
```
3Layer_AKS/
├── C10.py                # CIFAR-10
├── C100.py               # CIFAR-100
└── FashionMNIST.py       # Fashion-MNIST

6Layer_AKS/
├── C10.py                # CIFAR-10
├── C100.py               # CIFAR-100
└── FashionMNIST.py       # Fashion-MNIST

VGG11_AKS/
├── C10.py                # CIFAR-10
├── C100.py               # CIFAR-100
└── FashionMNIST.py       # Fashion-MNIST
```
