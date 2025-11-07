# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from editdatareader import get_data_loaders, NEW_CLASS_NAMES
from CobaResNet18 import ResNet18Custom
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER_TYPE = "Adam"  # bisa "SGD", "RMSprop", atau "Adam"

# --- Persiapan Data ---
train_loader, val_loader, n_classes, n_channels = get_data_loaders(BATCH_SIZE)

# --- Inisialisasi Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18Custom(in_channels=n_channels, num_classes=n_classes).to(device)

# --- Hitung pos_weight (imbalance handling) ---
total_0 = sum(1 for _, labels in train_loader.dataset if labels.item() == 0)
total_1 = sum(1 for _, labels in train_loader.dataset if labels.item() == 1)
pos_weight = torch.tensor([total_0 / total_1], device=device)
print(f"Pos weight digunakan: {pos_weight.item():.2f}")

# --- Loss function ---
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# --- Optimizer ---
if OPTIMIZER_TYPE == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
elif OPTIMIZER_TYPE == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, alpha=0.9, weight_decay=1e-4)
else:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)

# --- Learning Rate Scheduler ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)

# --- Fungsi Training ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_value=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100 * correct / total

# --- Fungsi Validasi ---
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100 * correct / total

# --- Loop Training ---
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("\n--- MULAI TRAINING ---\n")

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
    print(f"  Current LR: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

print("\n--- TRAINING SELESAI ---\n")

# --- Simpan Model ---
torch.save(model.state_dict(), "best_resnet18_binary.pth")
print("Model disimpan ke 'best_resnet18_binary.pth'")

# --- Visualisasi Training ---
plot_training_history(train_losses, val_losses, train_accs, val_accs)

# --- Visualisasi Prediksi ---
visualize_random_val_predictions(model, val_loader, device, NEW_CLASS_NAMES)
