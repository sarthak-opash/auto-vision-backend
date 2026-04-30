# train_resnet50.py

import os
import json
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    # Paths (repo-root relative)
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / 'auto_labeled_cars'  # Root directory with class subdirectories
    checkpoint_dir = BASE_DIR / 'runs' / 'model_identity'
    results_dir = BASE_DIR / 'results' / 'model_identity'
    
    # These will be created during train/val/test split
    train_dir = BASE_DIR / 'auto_labeled_cars_split' / 'train'
    val_dir = BASE_DIR / 'auto_labeled_cars_split' / 'val'
    test_dir = BASE_DIR / 'auto_labeled_cars_split' / 'test'
    
    # Model parameters
    num_classes = 7  # sedan, suv, truck, coupe, hatchback, van, sports
    img_size = 224
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Training parameters
    num_workers = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_every = 5  # Save checkpoint every N epochs
    early_stopping_patience = 10
    
    # Class names (modify based on your categories)
    class_names = ['sedan', 'suv', 'truck', 'coupe', 'hatchback', 'van', 'sports']
    
    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

config = Config()

# ============================================================================
# 2. DATASET CLASS
# ============================================================================

class DamagedCarDataset(Dataset):
    """Custom Dataset for damaged car images"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with subdirectories for each class
            transform (callable, optional): Transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Build list of (image_path, label) tuples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# 3. DATA AUGMENTATION & PREPROCESSING
# ============================================================================

def get_transforms(mode='train'):
    """
    Get image transforms for training/validation
    
    Args:
        mode (str): 'train' or 'val'
    """
    
    if mode == 'train':
        # Aggressive augmentation for training
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(p=0.3),  # Use carefully - some cars are asymmetric
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # Add some blur to simulate poor quality photos
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Simple preprocessing for validation/test
        transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

# ============================================================================
# 4. MODEL DEFINITION
# ============================================================================

class DamagedCarClassifier(nn.Module):
    """ResNet50 based classifier for damaged cars"""
    
    def __init__(self, num_classes, pretrained=True):
        super(DamagedCarClassifier, self).__init__()
        
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Get number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze all ResNet layers except the final classifier"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classifier
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_layer4(self):
        """Unfreeze the last residual block (layer4) for fine-tuning"""
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for full fine-tuning"""
        for param in self.resnet.parameters():
            param.requires_grad = True

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================================================
# 6. MAIN TRAINING LOOP
# ============================================================================

def split_dataset_into_train_val_test():
    """Split images from class subdirectories into train/val/test splits"""
    print("Splitting dataset into train/val/test...")
    
    # Get all class directories
    class_dirs = [d for d in config.data_dir.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all images in this class
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not images:
            continue
        
        # Split into train/val/test (70/15/15)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Create class directories in train/val/test
        (config.train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (config.val_dir / class_name).mkdir(parents=True, exist_ok=True)
        (config.test_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to respective directories (or create symlinks)
        for img in train_imgs:
            dest = config.train_dir / class_name / img.name
            if not dest.exists():
                import shutil
                shutil.copy2(img, dest)
        
        for img in val_imgs:
            dest = config.val_dir / class_name / img.name
            if not dest.exists():
                import shutil
                shutil.copy2(img, dest)
        
        for img in test_imgs:
            dest = config.test_dir / class_name / img.name
            if not dest.exists():
                import shutil
                shutil.copy2(img, dest)
        
        print(f"  {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

def train_model():
    """Main training function"""
    
    print(f"Using device: {config.device}")
    print(f"Training for {config.num_epochs} epochs")
    print("=" * 60)
    
    # Split dataset if not already split
    split_dataset_into_train_val_test()
    
    # Create datasets
    train_dataset = DamagedCarDataset(
        root_dir=str(config.train_dir),
        transform=get_transforms('train')
    )
    
    val_dataset = DamagedCarDataset(
        root_dir=str(config.val_dir),
        transform=get_transforms('val')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = DamagedCarClassifier(num_classes=config.num_classes, pretrained=True)
    model = model.to(config.device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # ========================================================================
    # PHASE 1: Train only classifier head (epochs 1-10)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier head only (backbone frozen)")
    print("=" * 60)
    
    model.freeze_backbone()
    phase1_epochs = 10
    
    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch + 1}/{phase1_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, config.device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.checkpoint_dir / 'best_model_phase1.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # ========================================================================
    # PHASE 2: Fine-tune last residual block (epochs 11-30)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning layer4 (last residual block)")
    print("=" * 60)
    
    model.unfreeze_layer4()
    
    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate * 0.1
    
    phase2_epochs = 20
    
    for epoch in range(phase2_epochs):
        print(f"\nEpoch {phase1_epochs + epoch + 1}/{phase1_epochs + phase2_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': phase1_epochs + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.checkpoint_dir / 'best_model_phase2.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # ========================================================================
    # PHASE 3: Full fine-tuning (remaining epochs)
    # ========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Full fine-tuning (all layers)")
    print("=" * 60)
    
    model.unfreeze_all()
    
    # Even lower learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = config.learning_rate * 0.01
    
    remaining_epochs = config.num_epochs - (phase1_epochs + phase2_epochs)
    epochs_no_improve = 0
    
    for epoch in range(remaining_epochs):
        current_epoch = phase1_epochs + phase2_epochs + epoch + 1
        print(f"\nEpoch {current_epoch}/{config.num_epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, config.device
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, config.checkpoint_dir / 'best_model_final.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Save detailed results for best model
            save_results(val_preds, val_labels, config.class_names)
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered!")
            break
    
    # Save training history
    with open(config.results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    plot_training_curves(history)
    
    print("\n" + "=" * 60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)
    
    return model, history

# ============================================================================
# 7. EVALUATION & VISUALIZATION
# ============================================================================

def save_results(predictions, labels, class_names):
    """Save classification report and confusion matrix"""
    
    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    
    with open(config.results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.results_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Results saved to {config.results_dir}/")

def plot_training_curves(history):
    """Plot training and validation curves"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(config.results_dir / 'training_curves.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Training curves saved to {config.results_dir}/training_curves.png")

# ============================================================================
# 8. INFERENCE FUNCTION
# ============================================================================

def predict_single_image(model, image_path, class_names, device):
    """
    Predict car type for a single damaged car image
    
    Args:
        model: Trained model
        image_path: Path to image
        class_names: List of class names
        device: Device to run inference on
    
    Returns:
        Predicted class and confidence scores
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms('val')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get top-3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_classes = [(class_names[idx.item()], prob.item() * 100) 
                    for idx, prob in zip(top3_indices[0], top3_prob[0])]
    
    return predicted_class, confidence_score, top3_classes

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Train the model
    model, history = train_model()
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print("=" * 60)
    
    # Example inference (optional)
    # Load best model
    # checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_model_final.pth'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # test_image = './test_images/damaged_car.jpg'
    # predicted_class, confidence, top3 = predict_single_image(
    #     model, test_image, config.class_names, config.device
    # )
    # print(f"\nPredicted: {predicted_class} (Confidence: {confidence:.2f}%)")
    # print(f"Top-3 predictions: {top3}")