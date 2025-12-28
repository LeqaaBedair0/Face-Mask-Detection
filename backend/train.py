from model_architecture import EnhancedCNN, OptimizedCNN
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


root = "data"
print(os.listdir(root))
# Define classes
Classes = ['with_mask', 'without_mask']  # Assuming these are the classes

Img_size = (128, 128)
Img_path = []
labels = []

for cls in Classes:
    cls_dir = os.path.join(root, cls)
    print("Loading:", cls_dir)
    for fname in os.listdir(cls_dir):
        fpath = os.path.join(cls_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.resize(img, Img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        Img_path.append(img)
        labels.append(cls)

Img_path = np.array(Img_path)
labels = np.array(labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Label Smoothing Loss
class SmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    

data_dir = 'data'

# Original transforms (kept for basic mode)
train_transform_basic = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform_basic = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Enhanced transforms (for advanced mode)
def get_enhanced_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# Load basic dataset
full_dataset_basic = datasets.ImageFolder(root=data_dir, transform=train_transform_basic)

# Split train/validation
train_size = int(0.8 * len(full_dataset_basic))
val_size = len(full_dataset_basic) - train_size
train_dataset_basic, val_dataset_basic = random_split(full_dataset_basic, [train_size, val_size])

# Basic DataLoaders
batch_size = 128
train_loader_basic = DataLoader(train_dataset_basic, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
val_loader_basic = DataLoader(val_dataset_basic, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_dataset_basic)}")
print(f"Validation samples: {len(val_dataset_basic)}")
print(f"Batch size: {batch_size}")

def setup_training(use_advanced=True):
    """Setup model, loss function, and optimizer for basic/advanced mode"""
    if use_advanced:
        model = OptimizedCNN(num_classes=2).to(device)
        criterion = SmoothCrossEntropy(smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    else:
        model = OptimizedCNN(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    return model, criterion, optimizer, scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:03d}/{num_epochs} [Train]')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='[Validation]'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def train_model(num_epochs, use_advanced):
    """Complete training function for basic/advanced mode"""
    # Setup
    model, criterion, optimizer, scheduler = setup_training(use_advanced)

    # Track metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None

    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader_basic, criterion, optimizer, epoch, num_epochs
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader_basic, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'models/best_model.pth')

            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

        # Print epoch summary
        print(f"\nEpoch {epoch+1:03d}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Best Val Acc so far: {best_val_acc:.2f}%")
        print("-" * 50)

    # Load best model for final testing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accs, val_losses, val_accs, best_val_acc

def train_model_enhanced(num_epochs, use_advanced=True, patience=5):
    """Training with early stopping and enhanced techniques"""
    # Get enhanced transforms
    train_transform, val_transform = get_enhanced_transforms()

    # Reload dataset with enhanced transforms
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Update data loaders
    train_loader = DataLoader(train_dataset, batch_size=64,
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64,
                          shuffle=False, num_workers=2, pin_memory=True)

    # Setup model with more regularization
    model = EnhancedCNN(num_classes=2, dropout_rate=0.4).to(device)

    # Enhanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0005,
                          weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-5
    )

    # Label smoothing loss
    criterion = SmoothCrossEntropy(smoothing=0.15)

    # Tracking
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\nEnhanced Training for {num_epochs} epochs...")
    print(f"Batch size: 64 | Dropout: 0.4 | Learning rate: 0.0005")
    print("-" * 50)

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, num_epochs
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step()

        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'models/best_model_enhanced.pth')

            print(f"✅ New best model! Val Acc: {val_acc:.2f}% (Improved)")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1:03d}/{num_epochs}:")
        print(f"  Train: Loss={train_loss:.4f} | Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc:.2f}%")
        print(f"  Best Val: {best_val_acc:.2f}% | LR: {current_lr:.6f}")
        print(f"  Gap (Train-Val): {train_acc - val_acc:.2f}%")
        print("-" * 50)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, train_accs, val_losses, val_accs, best_val_acc

def test_model(model, test_loader):
    """Test the model on test set"""
    model.eval()
    correct = 0
    total = 0

    print("\n" + "="*50)
    print("TESTING ON TEST SET")
    print("="*50)

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = 100. * correct / total

    print(f"\nTest Results:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Correct: {correct:,} / Total: {total:,}")
    print("="*50)
    return test_acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs, mode_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'training_history_{mode_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print final metrics
    print(f"\nFinal Training Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"Gap (Train - Val): {train_accs[-1] - val_accs[-1]:.2f}%")
    if (train_accs[-1] - val_accs[-1]) > 5:
        print("⚠️  Warning: Possible overfitting (gap > 5%)")
    else:
        print("✅ Good: No significant overfitting")
        
def plot_confusion_matrix(model, data_loader):
    """Plot confusion matrix for detailed analysis"""

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Generating predictions'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    class_names = ['with_mask', 'without_mask']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=class_names))

    return cm

