import torch
import logging
from sklearn.metrics import accuracy_score
from torchvision import models
from utils import (
    load_checkpoint,
    generate_gradcam,
    visualize_prediction,
    plot_confusion_matrix
)
from data_loader import get_data_loaders

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    logging.info("🚀 Evaluation started...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"🖥️ Using device: {device}")

    # 📁 Load data
    data_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/DATA"
    loaders = get_data_loaders(data_dir, batch_size=1, img_size=224, visualize=False)
    test_loader = loaders["test"]
    class_names = loaders["class_names"]
    logging.info(f"✅ Loaded {len(test_loader.dataset)} test samples.")

    # 🧠 Load model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model = load_checkpoint(model, "checkpoints/best_model.pth", device)
    model.to(device)
    model.eval()
    logging.info("📦 Model checkpoint loaded.")

    # 🔍 Inference & Grad-CAM
    y_true = []
    y_pred = []
    seen_classes = set()

    logging.info("🔍 Running inference...")
    for img, label in test_loader:
        label = label.item()
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()

        if label not in seen_classes:
            target_layer = model.layer4[-1]
            gradcam_img = generate_gradcam(model, img.squeeze(), target_layer, device, class_idx=pred)
            visualize_prediction(img.squeeze(), label, pred, class_names, gradcam_img)
            seen_classes.add(label)

        y_true.append(label)
        y_pred.append(pred)

    # 🎯 Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

    # 📊 Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names, save_path="outputs/confusion_matrix.png")
    logging.info("📊 Confusion matrix saved to outputs/confusion_matrix.png")
    logging.info("✅ Evaluation complete.")

if __name__ == "__main__":
    main()