import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

class AdaptiveKernelSelector(nn.Module):
    def __init__(self, in_channels, num_kernels=3):
        super(AdaptiveKernelSelector, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_sizes = [3, 5, 7]
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.selector = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, num_kernels),
            nn.Softmax(dim=1)
        )
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2)
            for k in self.kernel_sizes
        ])
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        stats = self.gap(x).view(batch_size, channels)
        weights = self.selector(stats)
        
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        
        output = torch.zeros_like(x)
        for i, conv_out in enumerate(conv_outputs):
            weight = weights[:, i].view(batch_size, 1, 1, 1)
            output += weight * conv_out
            
        return output, weights


class AdaptiveReceptiveFieldCNN(nn.Module):
    """
    6-layer CNN with 3 adaptive kernel selection modules
    Architecture: conv1 -> adaptive1 -> conv2 -> adaptive2 -> conv3 -> adaptive3
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(AdaptiveReceptiveFieldCNN, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Adaptive layer 1
        self.adaptive1 = AdaptiveKernelSelector(64)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv layer 2 with downsampling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Adaptive layer 2
        self.adaptive2 = AdaptiveKernelSelector(128)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Conv layer 3 with downsampling
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Adaptive layer 3
        self.adaptive3 = AdaptiveKernelSelector(256)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Global average pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
        # Store attention weights for analysis
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Adaptive layer 1
        x, weights1 = self.adaptive1(x)
        x = F.relu(self.bn2(x))
        self.attention_weights.append(weights1)
        
        # Conv layer 2
        x = F.relu(self.bn3(self.conv2(x)))
        
        # Adaptive layer 2
        x, weights2 = self.adaptive2(x)
        x = F.relu(self.bn4(x))
        self.attention_weights.append(weights2)
        
        # Conv layer 3
        x = F.relu(self.bn5(self.conv3(x)))
        
        # Adaptive layer 3
        x, weights3 = self.adaptive3(x)
        x = F.relu(self.bn6(x))
        self.attention_weights.append(weights3)
        
        # Classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class StandardCNN(nn.Module):
    """
    Standard CNN baseline for comparison (6 conv layers, no adaptive)
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(StandardCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def get_data_loaders(batch_size=128, dataset='cifar10'):
    """Get CIFAR-10 or CIFAR-100 data loaders"""
    
    if dataset == 'fashionmnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform_val)
        num_classes = 10
        
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_val)
        num_classes = 100
        
    else:  # cifar10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_val)
        num_classes = 10
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, num_classes


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Training function with validation tracking"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_accuracies


def analyze_kernel_selection(model, val_loader, device='cuda'):
    """Analyze kernel selection patterns"""
    model.eval()
    kernel_selections = {0: [], 1: [], 2: []}  # 3 adaptive layers
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            _ = model(data)
            
            for i, weights in enumerate(model.attention_weights):
                kernel_selections[i].append(weights.cpu().numpy())
    
    # Aggregate results
    results = {}
    for layer_idx in range(3):
        all_weights = np.concatenate(kernel_selections[layer_idx], axis=0)
        avg_weights = np.mean(all_weights, axis=0)
        results[layer_idx] = avg_weights
    
    return results


def plot_results(results, train_losses_adaptive, train_losses_standard, 
                 val_accs_adaptive, val_accs_standard, kernel_selections):
    """Plot comprehensive results"""
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training Loss
    ax1 = axes[0, 0]
    ax1.plot(train_losses_adaptive, label='Adaptive CNN', color='blue')
    ax1.plot(train_losses_standard, label='Standard CNN', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Validation Accuracy
    ax2 = axes[0, 1]
    ax2.plot(val_accs_adaptive, label='Adaptive CNN', color='blue')
    ax2.plot(val_accs_standard, label='Standard CNN', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Kernel Selection Heatmap
    ax3 = axes[1, 0]
    heatmap_data = np.array([kernel_selections[i] for i in range(3)])
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('Layer')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['3×3', '5×5', '7×7'])
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
    ax3.set_title('Kernel Selection Patterns (3 Adaptive Layers)')
    plt.colorbar(im, ax=ax3)
    
    # Add values to heatmap
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center', color='black')
    
    # Bar chart of kernel preferences
    ax4 = axes[1, 1]
    x = np.arange(3)
    width = 0.25
    
    for i in range(3):
        ax4.bar(x + i*width, [kernel_selections[layer][i] for layer in range(3)], 
                width, label=f'{[3,5,7][i]}×{[3,5,7][i]}')
    
    ax4.set_xlabel('Adaptive Layer')
    ax4.set_ylabel('Selection Weight')
    ax4.set_title('Kernel Preferences by Layer')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
    ax4.legend()
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/3_layer_adaptive_results.png', dpi=300)
    plt.show()
    
    print("\nResults saved to 'results/3_layer_adaptive_results.png'")


def run_experiment(seeds=[42, 123, 456], num_epochs=50, dataset='cifar10'):
    """Run multi-seed experiment"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = get_data_loaders(dataset=dataset)
    
    # Determine input channels based on dataset
    input_channels = 1 if dataset == 'fashionmnist' else 3
    
    all_results = {
        'adaptive': {'train_loss': [], 'val_acc': [], 'final_acc': []},
        'standard': {'train_loss': [], 'val_acc': [], 'final_acc': []},
        'kernel_selections': []
    }
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Running seed {seed}")
        print(f"{'='*50}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train Adaptive CNN
        print("\nTraining Adaptive CNN (3 adaptive layers)...")
        adaptive_model = AdaptiveReceptiveFieldCNN(num_classes=num_classes, input_channels=input_channels)
        train_loss_a, val_acc_a = train_model(adaptive_model, train_loader, val_loader, 
                                               num_epochs=num_epochs, device=device)
        all_results['adaptive']['train_loss'].append(train_loss_a)
        all_results['adaptive']['val_acc'].append(val_acc_a)
        all_results['adaptive']['final_acc'].append(val_acc_a[-1])
        
        # Analyze kernel selection
        kernel_sel = analyze_kernel_selection(adaptive_model, val_loader, device)
        all_results['kernel_selections'].append(kernel_sel)
        
        # Train Standard CNN
        print("\nTraining Standard CNN...")
        torch.manual_seed(seed)
        standard_model = StandardCNN(num_classes=num_classes, input_channels=input_channels)
        train_loss_s, val_acc_s = train_model(standard_model, train_loader, val_loader,
                                               num_epochs=num_epochs, device=device)
        all_results['standard']['train_loss'].append(train_loss_s)
        all_results['standard']['val_acc'].append(val_acc_s)
        all_results['standard']['final_acc'].append(val_acc_s[-1])
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY (3-Layer Adaptive CNN)")
    print("="*60)
    
    adaptive_mean = np.mean(all_results['adaptive']['final_acc'])
    adaptive_std = np.std(all_results['adaptive']['final_acc'])
    standard_mean = np.mean(all_results['standard']['final_acc'])
    standard_std = np.std(all_results['standard']['final_acc'])
    
    print(f"Adaptive CNN: {adaptive_mean:.2f}% ± {adaptive_std:.2f}%")
    print(f"Standard CNN: {standard_mean:.2f}% ± {standard_std:.2f}%")
    print(f"Improvement: {adaptive_mean - standard_mean:.2f}%")
    
    # Average kernel selections
    avg_kernel_sel = {}
    for layer in range(3):
        avg_kernel_sel[layer] = np.mean([ks[layer] for ks in all_results['kernel_selections']], axis=0)
    
    print("\nKernel Selection Patterns:")
    for layer in range(3):
        print(f"  Layer {layer+1}: 3×3: {avg_kernel_sel[layer][0]:.3f}, "
              f"5×5: {avg_kernel_sel[layer][1]:.3f}, 7×7: {avg_kernel_sel[layer][2]:.3f}")
    
    # Plot results (using last seed's data for curves, averaged kernel selection)
    plot_results(
        all_results,
        all_results['adaptive']['train_loss'][-1],
        all_results['standard']['train_loss'][-1],
        all_results['adaptive']['val_acc'][-1],
        all_results['standard']['val_acc'][-1],
        avg_kernel_sel
    )
    
    return all_results


if __name__ == "__main__":
    print("="*60)
    print("3-LAYER ADAPTIVE KERNEL SELECTION CNN EXPERIMENT")
    print("="*60)
    print("Architecture: 6 conv layers with 3 adaptive kernel selectors")
    print("Dataset: CIFAR-10")
    print("Seeds: 3")
    print("Epochs: 50")
    print("="*60)
    
    results = run_experiment(seeds=[42, 123, 456], num_epochs=25)
    
