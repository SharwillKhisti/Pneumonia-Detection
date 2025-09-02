import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

def train_model(model, train_loader, val_loader, class_weights, device,
                num_epochs=10, lr=1e-4, checkpoint_path=None,
                log_dir="runs", model_name="model"):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

    best_val_metric = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")

        # 🔄 Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc="🔄 Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        writer.add_scalar("Loss/train", train_loss / total, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        print(f"📊 Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.4f}")

        # 🔍 Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="🔍 Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = correct / total
        val_prec = precision_score(all_labels, all_preds, zero_division=0)
        val_rec = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)

        writer.add_scalar("Loss/val", val_loss / total, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        print(f"📈 Val Loss: {val_loss/total:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"⏱️ Duration: {time.time() - start_time:.2f}s")

        cm = confusion_matrix(all_labels, all_preds)
        print(f"📊 Confusion Matrix:\n{cm}")
        writer.add_text("Confusion Matrix", str(cm), epoch)

        # 🏆 Save Best Model
        if val_f1 > best_val_metric:
            best_val_metric = val_f1
            best_model_wts = model.state_dict()
            print("🏆 New best model found (by F1).")
            if checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_model_wts,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_metric
                }, checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")

        scheduler.step()

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / 1024**2
            print(f"📦 GPU memory used: {mem:.2f} MB")

    writer.close()
    print(f"\n✅ Training complete. Best Val F1: {best_val_metric:.4f}")
    return best_val_metric, model