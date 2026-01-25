# Adaptive Kernel Selection for CNNs

**Areen Patil, Summer 2025**

CNNs with adaptive kernel selection layers. Instead of using fixed kernel sizes, the network learns to pick between 3×3, 5×5, and 7×7 kernels based on the input features.

## Project Structure

```
Adaptive-Kernel-Selection/
├── 3Layer_AKS/
│   ├── C10.py          # CIFAR-10 experiments
│   ├── C100.py         # CIFAR-100 experiments
│   └── FashionMNIST.py # Fashion-MNIST experiments
├── 6Layer_AKS/
│   ├── C10.py          # CIFAR-10 experiments
│   ├── C100.py         # CIFAR-100 experiments
│   └── FashionMNIST.py # Fashion-MNIST experiments
├── VGG11_AKS/
│   ├── C10.py          # CIFAR-10 experiments
│   ├── C100.py         # CIFAR-100 experiments
│   └── FashionMNIST.py # Fashion-MNIST experiments
└── README.md
```

## How It Works

The `AdaptiveKernelSelector` module:
1. Takes feature maps and extracts channel stats using global average pooling
2. Runs a small MLP to predict weights for each kernel size (3×3, 5×5, 7×7)
3. Applies all three convolutions in parallel
4. Combines the outputs using the predicted weights

Each architecture has adaptive layers after the conv layers:
- **3Layer_AKS**: 3 conv layers, 3 adaptive layers
- **6Layer_AKS**: 6 conv layers, 6 adaptive layers  
- **VGG11_AKS**: 8 conv layers, 8 adaptive layers (VGG11-style)

## Setup

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

Just run the scripts:

```bash
cd 6Layer_AKS
python C10.py
```

Each script trains both an adaptive model and a standard baseline, then compares them. Results (training curves, kernel selection heatmaps) get saved to a `results/` folder.

You can tweak the experiment settings in `run_experiment()`:
- `seeds`: random seeds (default: [1, 24, 65])
- `num_epochs`: training epochs
- `dataset`: 'cifar10', 'cifar100', or 'fashionmnist'

Default training uses Adam optimizer (lr=0.001), batch size 128, and standard data augmentation for CIFAR datasets.
