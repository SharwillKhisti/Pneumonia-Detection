import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_data_loaders, EpochTracker
from models import get_model
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def format_confusion_matrix(cm, class_names):
    header = "| Actual \\ Predicted | " + " | ".join(class_names) + " |"
    divider = "|--------------------|" + "|".join(["-----------"] * len(class_names)) + "|"
    rows = []
    for i, row in enumerate(cm):
        row_str = f"| {class_names[i]:<18} | " + " | ".join(f"{val:<9}" for val in row) + " |"
        rows.append(row_str)
    formatted = "\n".join([header, divider] + rows)
    return formatted

def evaluate_confusion(model, val_loader, device, class_names, model_name, label):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    formatted_cm = format_confusion_matrix(cm, class_names)

    print(f"\n📊 {model_name} {label} Confusion Matrix:\n{formatted_cm}")
    print(f"\n📈 {model_name} {label} Classification Report:\n{report}")

    os.makedirs("outputs", exist_ok=True)
    report_path = f"outputs/classification_report_{model_name}_{label}.txt"
    cm_path = f"outputs/confusion_matrix_{model_name}_{label}.txt"

    with open(report_path, "w") as f:
        f.write(report)
    with open(cm_path, "w") as f:
        f.write(formatted_cm)

def main():
    parser = argparse.ArgumentParser(description="Evaluate models from checkpoints")
    parser.add_argument("--data_dir", type=str, default="DATA/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    epoch_tracker = EpochTracker()

    checkpoint_dir = "checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    for ckpt_file in checkpoint_files:
        parts = ckpt_file.replace(".pth", "").split("_")
        model_name = parts[0]

        # Handle naming variations
        if "CURLM" in parts or "DIRECT" in parts:
            mode = parts[1]
            phase = parts[-1].replace("phase", "").replace("Phase", "")
            label = f"{mode}_Phase{phase}"
        else:
            mode = "STANDARD"
            phase = parts[-1].replace("phase", "")
            label = f"Phase{phase}"

        print(f"\n🧠 Evaluating {model_name} - {label}")
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)

        try:
            epoch_tracker.current_epoch = int(phase)
        except ValueError:
            print(f"⚠️ Invalid phase format in filename: {ckpt_file}")
            continue

        loaders = get_data_loaders(data_dir=args.data_dir, batch_size=32, epoch_tracker=epoch_tracker)
        val_loader = loaders["val"]
        class_names = loaders["class_names"]

        try:
            model = get_model(model_name, num_classes=len(class_names))
        except ValueError as e:
            print(f"⚠️ {e}")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        evaluate_confusion(model, val_loader, device, class_names, model_name, label)

if __name__ == "__main__":
    main()