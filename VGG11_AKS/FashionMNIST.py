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


class AdaptiveVGG11(nn.Module):
    """
    VGG11 with 8 adaptive kernel selection layers
    VGG11 config: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    Adaptive layers after each conv layer (8 total)
    """
    def __init__(self, num_classes=10):
        super(AdaptiveVGG11, self).__init__()
        
        # Block 1: 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.adaptive1 = AdaptiveKernelSelector(64)
        self.bn1_adaptive = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 128 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.adaptive2 = AdaptiveKernelSelector(128)
        self.bn2_adaptive = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 256 channels (2 conv layers)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.adaptive3 = AdaptiveKernelSelector(256)
        self.bn3_adaptive = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive4 = AdaptiveKernelSelector(256)
        self.bn4_adaptive = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4: 512 channels (2 conv layers)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.adaptive5 = AdaptiveKernelSelector(512)
        self.bn5_adaptive = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.adaptive6 = AdaptiveKernelSelector(512)
        self.bn6_adaptive = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5: 512 channels (2 conv layers)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.adaptive7 = AdaptiveKernelSelector(512)
        self.bn7_adaptive = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.adaptive8 = AdaptiveKernelSelector(512)
        self.bn8_adaptive = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        
        self.attention_weights = []
        
    def forward(self, x):
        self.attention_weights = []
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x, weights1 = self.adaptive1(x)
        x = F.relu(self.bn1_adaptive(x))
        self.attention_weights.append(weights1)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x, weights2 = self.adaptive2(x)
        x = F.relu(self.bn2_adaptive(x))
        self.attention_weights.append(weights2)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x, weights3 = self.adaptive3(x)
        x = F.relu(self.bn3_adaptive(x))
        self.attention_weights.append(weights3)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x, weights4 = self.adaptive4(x)
        x = F.relu(self.bn4_adaptive(x))
        self.attention_weights.append(weights4)
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.bn5(self.conv5(x)))
        x, weights5 = self.adaptive5(x)
        x = F.relu(self.bn5_adaptive(x))
        self.attention_weights.append(weights5)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x, weights6 = self.adaptive6(x)
        x = F.relu(self.bn6_adaptive(x))
        self.attention_weights.append(weights6)
        x = self.pool4(x)
        
        # Block 5
        x = F.relu(self.bn7(self.conv7(x)))
        x, weights7 = self.adaptive7(x)
        x = F.relu(self.bn7_adaptive(x))
        self.attention_weights.append(weights7)
        
        x = F.relu(self.bn8(self.conv8(x)))
        x, weights8 = self.adaptive8(x)
        x = F.relu(self.bn8_adaptive(x))
        self.attention_weights.append(weights8)
        x = self.pool5(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class StandardVGG11(nn.Module):
    """
    Standard VGG11 baseline for comparison
    """
    def __init__(self, num_classes=10):
        super(StandardVGG11, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_data_loaders(batch_size=128, dataset='cifar10'):
    """Get CIFAR-10 or CIFAR-100 data loaders"""
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform_train)
    val_dataset = torchvision.datasets.FashionMNIST(
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
    """Analyze kernel selection patterns for 8 adaptive layers"""
    model.eval()
    kernel_selections = {i: [] for i in range(8)}  # 8 adaptive layers
    
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            _ = model(data)
            
            for i, weights in enumerate(model.attention_weights):
                kernel_selections[i].append(weights.cpu().numpy())
    
    # Aggregate results
    results = {}
    for layer_idx in range(8):
        all_weights = np.concatenate(kernel_selections[layer_idx], axis=0)
        avg_weights = np.mean(all_weights, axis=0)
        results[layer_idx] = avg_weights
    
    return results


def plot_results(results, train_losses_adaptive, train_losses_standard, 
                 val_accs_adaptive, val_accs_standard, kernel_selections):
    """Plot comprehensive results"""
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training Loss
    ax1 = axes[0, 0]
    ax1.plot(train_losses_adaptive, label='Adaptive VGG11', color='blue')
    ax1.plot(train_losses_standard, label='Standard VGG11', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Validation Accuracy
    ax2 = axes[0, 1]
    ax2.plot(val_accs_adaptive, label='Adaptive VGG11', color='blue')
    ax2.plot(val_accs_standard, label='Standard VGG11', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Kernel Selection Heatmap (8 layers)
    ax3 = axes[1, 0]
    heatmap_data = np.array([kernel_selections[i] for i in range(8)])
    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax3.set_xlabel('Kernel Size')
    ax3.set_ylabel('Layer')
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['3×3', '5×5', '7×7'])
    ax3.set_yticks(range(8))
    ax3.set_yticklabels([f'Layer {i+1}' for i in range(8)])
    ax3.set_title('VGG11 Kernel Selection Patterns (8 Adaptive Layers)')
    plt.colorbar(im, ax=ax3)
    
    # Add values to heatmap
    for i in range(8):
        for j in range(3):
            ax3.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', 
                    color='white' if heatmap_data[i, j] > 0.5 else 'black', fontsize=7)
    
    # Bar chart of kernel preferences
    ax4 = axes[1, 1]
    x = np.arange(8)
    width = 0.25
    
    for i in range(3):
        ax4.bar(x + i*width, [kernel_selections[layer][i] for layer in range(8)], 
                width, label=f'{[3,5,7][i]}×{[3,5,7][i]}')
    
    ax4.set_xlabel('Adaptive Layer')
    ax4.set_ylabel('Selection Weight')
    ax4.set_title('Kernel Preferences by Layer')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([f'L{i+1}' for i in range(8)])
    ax4.legend()
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/vgg11_8_layer_adaptive_results.png', dpi=300)
    plt.show()
    
    print("\nResults saved to 'results/vgg11_8_layer_adaptive_results.png'")


def run_experiment(seeds=[42, 123, 456], num_epochs=50, dataset='cifar10'):
    """Run multi-seed experiment"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = get_data_loaders(dataset=dataset)
    
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
        
        # Train Adaptive VGG11
        print("\nTraining Adaptive VGG11 (8 adaptive layers)...")
        adaptive_model = AdaptiveVGG11(num_classes=num_classes)
        train_loss_a, val_acc_a = train_model(adaptive_model, train_loader, val_loader, 
                                               num_epochs=num_epochs, device=device)
        all_results['adaptive']['train_loss'].append(train_loss_a)
        all_results['adaptive']['val_acc'].append(val_acc_a)
        all_results['adaptive']['final_acc'].append(val_acc_a[-1])
        
        # Analyze kernel selection
        kernel_sel = analyze_kernel_selection(adaptive_model, val_loader, device)
        all_results['kernel_selections'].append(kernel_sel)
        
        # Train Standard VGG11
        print("\nTraining Standard VGG11...")
        torch.manual_seed(seed)
        standard_model = StandardVGG11(num_classes=num_classes)
        train_loss_s, val_acc_s = train_model(standard_model, train_loader, val_loader,
                                               num_epochs=num_epochs, device=device)
        all_results['standard']['train_loss'].append(train_loss_s)
        all_results['standard']['val_acc'].append(val_acc_s)
        all_results['standard']['final_acc'].append(val_acc_s[-1])
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY (VGG11 with 8 Adaptive Layers)")
    print("="*60)
    
    adaptive_mean = np.mean(all_results['adaptive']['final_acc'])
    adaptive_std = np.std(all_results['adaptive']['final_acc'])
    standard_mean = np.mean(all_results['standard']['final_acc'])
    standard_std = np.std(all_results['standard']['final_acc'])
    
    print(f"Adaptive VGG11: {adaptive_mean:.2f}% ± {adaptive_std:.2f}%")
    print(f"Standard VGG11: {standard_mean:.2f}% ± {standard_std:.2f}%")
    print(f"Improvement: {adaptive_mean - standard_mean:.2f}%")
    
    # Average kernel selections
    avg_kernel_sel = {}
    for layer in range(8):
        avg_kernel_sel[layer] = np.mean([ks[layer] for ks in all_results['kernel_selections']], axis=0)
    
    print("\nKernel Selection Patterns:")
    for layer in range(8):
        print(f"  Layer {layer+1}: 3×3: {avg_kernel_sel[layer][0]:.3f}, "
              f"5×5: {avg_kernel_sel[layer][1]:.3f}, 7×7: {avg_kernel_sel[layer][2]:.3f}")
    
    # Plot results
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
    print("VGG11 WITH 8 ADAPTIVE KERNEL SELECTION LAYERS")
    print("="*60)
    print("Architecture: VGG11 with 8 adaptive kernel selectors")
    print("Dataset: CIFAR-10")
    print("Seeds: 3")
    print("Epochs: 50")
    print("="*60)
    
    results = run_experiment(seeds=[42, 123, 456], num_epochs=25)