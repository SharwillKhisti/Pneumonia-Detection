import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import time
import random
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"🔒 Seed set to {seed}")

def get_model(model_name, num_classes=2):
    print(f"🧠 Loading model: {model_name}")
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    print("✅ Model loaded and classifier head replaced.")
    return model

def train_model(model, train_loader, val_loader, class_weights, device,
                num_epochs=10, lr=1e-4, checkpoint_path=None,
                patience=3, log_dir="runs", model_name="model"):

    set_seed()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))

    best_val_acc = 0.0
    best_model_wts = model.state_dict()
    trigger_times = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n📅 Epoch {epoch+1}/{num_epochs} started")
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
        writer.add_scalar("Loss/train", train_loss/total, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        print(f"📊 Train Loss: {train_loss/total:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
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
        writer.add_scalar("Loss/val", val_loss/total, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        print(f"📈 Val Loss: {val_loss/total:.4f}, Val Acc: {val_acc:.4f}")
        print(f"⏱️ Epoch duration: {time.time() - start_time:.2f} seconds")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"📊 Confusion Matrix:\n{cm}")
        writer.add_text("Confusion Matrix", str(cm), epoch)

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            trigger_times = 0
            print("🏆 New best model found and saved.")
            if checkpoint_path:
                torch.save(best_model_wts, checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("🛑 Early stopping triggered.")
                break

        scheduler.step()

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / 1024**2
            print(f"📦 GPU memory used: {mem:.2f} MB")

    writer.close()
    print(f"\n✅ Training complete. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, best_model_wts