import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import MNISTCNN

def train_model(epochs=5, batch_size=64, lr=0.001, progress_callback=None, epoch_callback=None, dry_run=False):
    # Setup directories
    os.makedirs('data', exist_ok=True)
    
    # Preprocessing transforms: convert to tensor and normalize
    # MNIST mean is 0.1307, std is 0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download & Load MNIST
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if dry_run:
        # Use a very small subset for dry run to verify code paths
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        test_dataset = torch.utils.data.Subset(test_dataset, range(100))
        epochs = 1
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = MNISTCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Batch progress reporting
            if progress_callback:
                total_batches = len(train_loader) * epochs
                current_batch = epoch * len(train_loader) + batch_idx
                progress_callback(current_batch / total_batches, loss.item())
                
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
        val_loss /= len(test_loader.dataset)
        val_acc = val_correct / val_total
        test_losses.append(val_loss)
        test_accs.append(val_acc)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if epoch_callback:
            epoch_callback(epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc)
            
    # Save model weights
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print("Model saved as mnist_cnn.pth")
    
    # Save training curves plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Val Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', marker='o')
    plt.plot(range(1, epochs + 1), test_accs, label='Val Acc', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print("Training curves saved as training_curves.png")
    
    return model, train_losses, train_accs, test_losses, test_accs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN on MNIST')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--dry-run', action='store_true', help='Quick validation run')
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dry_run=args.dry_run)
