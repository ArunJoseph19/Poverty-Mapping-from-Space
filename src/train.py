# =============================================================================
# TRAIN CNN AND RIDGE REGRESSION
# Phase 1: Nightlight classification, Phase 2: Feature extraction + Ridge
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from pathlib import Path
from tqdm import tqdm
import warnings
import json
import gc
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING: CNN + Ridge Regression")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Update these paths for your system
data_path = Path('./data/processed')
results_path = Path(f'./results/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
results_path.mkdir(exist_ok=True, parents=True)

for folder in ['models', 'figures', 'predictions', 'metrics']:
    (results_path / folder).mkdir(exist_ok=True)

print(f"\nInput: {data_path}")
print(f"Output: {results_path}")

# =============================================================================
# DATASET CLASS
# =============================================================================

class MemmapDataset(Dataset):
    """Memory-efficient dataset using numpy memmap"""
    
    def __init__(self, images_path, labels, indices, shape):
        self.images = np.memmap(images_path, dtype='float32', mode='r', shape=shape)
        self.labels = labels
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = torch.from_numpy(self.images[real_idx].copy()).float()
        label = self.labels[idx]
        return img, label

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class LightCNN256(nn.Module):
    """
    Lightweight CNN for 256x256 multi-spectral images.
    Approximately 4M parameters.
    """
    
    def __init__(self, n_bands=7, feature_dim=512):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Block 1: 256 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(n_bands, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Block 2: 64 -> 16
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Block 3: 16 -> 4
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(512, feature_dim)
        
        self.nl_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward_backbone(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)
    
    def extract_features(self, x):
        x = self.forward_backbone(x)
        return self.feature_proj(x)
    
    def forward_nl(self, x):
        x = self.forward_backbone(x)
        return self.nl_head(x)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data...")

images_path = data_path / 'satellite_images.npy'
wealth_labels = np.load(data_path / 'wealth_labels.npy')
nightlight_labels = np.load(data_path / 'nightlight_labels.npy')

train_idx = np.load(data_path / 'train_idx.npy')
val_idx = np.load(data_path / 'val_idx.npy')
test_idx = np.load(data_path / 'test_idx.npy')

with open(data_path / 'dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

n_samples = dataset_info['n_images']
n_bands = dataset_info['n_bands']
img_size = dataset_info['image_size']
memmap_shape = (n_samples, n_bands, img_size, img_size)

print(f"Images: ({n_samples}, {n_bands}, {img_size}, {img_size})")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# Labels
train_nl_labels = torch.LongTensor(nightlight_labels[train_idx])
val_nl_labels = torch.LongTensor(nightlight_labels[val_idx])
test_nl_labels = torch.LongTensor(nightlight_labels[test_idx])

train_wealth = torch.FloatTensor(wealth_labels[train_idx])
val_wealth = torch.FloatTensor(wealth_labels[val_idx])
test_wealth = torch.FloatTensor(wealth_labels[test_idx])

# DataLoaders
batch_size = 16

train_dataset_nl = MemmapDataset(images_path, train_nl_labels, train_idx, memmap_shape)
val_dataset_nl = MemmapDataset(images_path, val_nl_labels, val_idx, memmap_shape)
test_dataset_nl = MemmapDataset(images_path, test_nl_labels, test_idx, memmap_shape)

train_loader_nl = DataLoader(train_dataset_nl, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader_nl = DataLoader(val_dataset_nl, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader_nl = DataLoader(test_dataset_nl, batch_size=batch_size, shuffle=False, num_workers=0)

train_dataset = MemmapDataset(images_path, train_wealth, train_idx, memmap_shape)
val_dataset = MemmapDataset(images_path, val_wealth, val_idx, memmap_shape)
test_dataset = MemmapDataset(images_path, test_wealth, test_idx, memmap_shape)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# PHASE 1: NIGHTLIGHT CLASSIFICATION
# =============================================================================

print("\n" + "="*70)
print("PHASE 1: NIGHTLIGHT CLASSIFICATION")
print("="*70)

model = LightCNN256(n_bands=n_bands, feature_dim=512).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

epochs = 20
best_val_acc = 0.0
train_losses, val_accs = [], []
patience, patience_counter = 7, 0

print(f"\nTraining for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader_nl, desc=f'E{epoch+1}/{epochs}', leave=False)
    for Xb, yb in pbar:
        Xb, yb = Xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model.forward_nl(Xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model.forward_nl(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total += len(yb)
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{train_correct/train_total:.3f}'})
        del Xb, yb, logits, loss
    
    scheduler.step()
    avg_loss = train_loss / len(train_loader_nl)
    train_acc = train_correct / train_total
    train_losses.append(avg_loss)
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader_nl:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model.forward_nl(Xb)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_total += len(yb)
            del Xb, yb, logits
    
    val_acc = val_correct / val_total
    val_accs.append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({'model_state_dict': model.state_dict(), 'val_acc': val_acc}, 
                   results_path / 'models' / 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f'E{epoch+1:2d}: Loss={avg_loss:.4f}, Train={train_acc:.3f}, Val={val_acc:.3f}, Best={best_val_acc:.3f}')
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nPhase 1 complete! Best val accuracy: {best_val_acc:.3f}")

# =============================================================================
# PHASE 2: FEATURE EXTRACTION + RIDGE
# =============================================================================

print("\n" + "="*70)
print("PHASE 2: FEATURE EXTRACTION + RIDGE REGRESSION")
print("="*70)

checkpoint = torch.load(results_path / 'models' / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded best model (Val Acc: {checkpoint['val_acc']:.3f})")

def extract_features(loader, desc="Extracting"):
    features = []
    with torch.no_grad():
        for Xb, _ in tqdm(loader, desc=desc, leave=False):
            Xb = Xb.to(device)
            feats = model.extract_features(Xb).cpu().numpy()
            features.append(feats)
            del Xb
    return np.vstack(features)

print("\nExtracting features...")
train_feats = extract_features(train_loader, "Train")
val_feats = extract_features(val_loader, "Val")
test_feats = extract_features(test_loader, "Test")

print(f"Features: Train={train_feats.shape}, Test={test_feats.shape}")

train_feats = np.nan_to_num(train_feats, nan=0.0)
val_feats = np.nan_to_num(val_feats, nan=0.0)
test_feats = np.nan_to_num(test_feats, nan=0.0)

print("\nTraining Ridge Regression...")
ridge = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5, scoring='r2')
ridge.fit(train_feats, train_wealth.numpy())
print(f"Best alpha: {ridge.alpha_:.2e}")

train_preds = ridge.predict(train_feats)
val_preds = ridge.predict(val_feats)
test_preds = ridge.predict(test_feats)

train_r2 = r2_score(train_wealth.numpy(), train_preds)
val_r2 = r2_score(val_wealth.numpy(), val_preds)
test_r2 = r2_score(test_wealth.numpy(), test_preds)

train_mae = np.mean(np.abs(train_wealth.numpy() - train_preds))
val_mae = np.mean(np.abs(val_wealth.numpy() - val_preds))
test_mae = np.mean(np.abs(test_wealth.numpy() - test_preds))

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
print(f"{'Dataset':<12} {'R2':>10} {'MAE':>10}")
print(f"{'-'*32}")
print(f"{'Train':<12} {train_r2:>10.4f} {train_mae:>10.4f}")
print(f"{'Validation':<12} {val_r2:>10.4f} {val_mae:>10.4f}")
print(f"{'Test':<12} {test_r2:>10.4f} {test_mae:>10.4f}")
print(f"{'='*50}")

# =============================================================================
# BASELINES
# =============================================================================

print("\nComputing baselines...")

images_mmap = np.memmap(images_path, dtype='float32', mode='r', shape=memmap_shape)

nl_idx = 4
nl_train = np.array([images_mmap[i, nl_idx].mean() for i in tqdm(train_idx, desc="NL Train", leave=False)]).reshape(-1, 1)
nl_test = np.array([images_mmap[i, nl_idx].mean() for i in tqdm(test_idx, desc="NL Test", leave=False)]).reshape(-1, 1)

ridge_nl = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5)
ridge_nl.fit(nl_train, train_wealth.numpy())
nl_preds = ridge_nl.predict(nl_test)
nl_r2 = r2_score(test_wealth.numpy(), nl_preds)

all_train = np.array([images_mmap[i].mean(axis=(1, 2)) for i in tqdm(train_idx, desc="All Train", leave=False)])
all_test = np.array([images_mmap[i].mean(axis=(1, 2)) for i in tqdm(test_idx, desc="All Test", leave=False)])

ridge_all = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5)
ridge_all.fit(all_train, train_wealth.numpy())
all_preds = ridge_all.predict(all_test)
all_r2 = r2_score(test_wealth.numpy(), all_preds)

del images_mmap
gc.collect()

improvement = test_r2 / nl_r2 if nl_r2 > 0 else float('inf')

print(f"\nCOMPARISON")
print(f"{'-'*40}")
print(f"Nightlights only:     R2 = {nl_r2:.4f}")
print(f"All bands mean:       R2 = {all_r2:.4f}")
print(f"CNN Transfer (ours):  R2 = {test_r2:.4f}")
print(f"{'-'*40}")
print(f"Improvement: {improvement:.2f}x over NL-only")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\nSaving results...")

summary = {
    'method': 'LightCNN256 + Ridge',
    'image_size': f'{img_size}x{img_size}',
    'n_bands': n_bands,
    'n_images': n_samples,
    'phase1_accuracy': float(checkpoint['val_acc']),
    'train_r2': float(train_r2),
    'val_r2': float(val_r2),
    'test_r2': float(test_r2),
    'test_mae': float(test_mae),
    'baseline_nl_r2': float(nl_r2),
    'baseline_all_r2': float(all_r2),
    'improvement_over_nl': float(improvement),
}

with open(results_path / 'metrics' / 'results.json', 'w') as f:
    json.dump(summary, f, indent=2)

pd.DataFrame({
    'true': test_wealth.numpy(),
    'pred': test_preds
}).to_csv(results_path / 'predictions' / 'test_predictions.csv', index=False)

joblib.dump(ridge, results_path / 'models' / 'ridge_model.joblib')

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(test_wealth.numpy(), test_preds, alpha=0.4, s=15, c='steelblue')
lims = [min(test_wealth.min(), test_preds.min()), max(test_wealth.max(), test_preds.max())]
axes[0].plot(lims, lims, 'r--', lw=2)
axes[0].set_xlabel('True Wealth Score')
axes[0].set_ylabel('Predicted Wealth Score')
axes[0].set_title(f'CNN Transfer Learning\nTest R2 = {test_r2:.3f}')
axes[0].grid(True, alpha=0.3)

methods = ['NL Only', 'All Bands', 'CNN Transfer']
r2_vals = [nl_r2, all_r2, test_r2]
colors = ['#e74c3c', '#3498db', '#27ae60']
bars = axes[1].bar(methods, r2_vals, color=colors, edgecolor='black')
axes[1].set_ylabel('R2 Score')
axes[1].set_title('Method Comparison')
for bar, v in zip(bars, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_path / 'figures' / 'results.png', dpi=150)
plt.close()

print(f"\nAll saved to: {results_path}")
print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nFinal R2: {test_r2:.4f}")
print(f"Improvement: {improvement:.1f}x over NL-only")
