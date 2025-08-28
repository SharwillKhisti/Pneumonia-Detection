import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from model import get_model, train_model
from utils import set_seed, generate_gradcam, visualize_prediction, plot_confusion_matrix
from dataset import ChestXrayDataset  # You should have this defined
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Pipeline")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "densenet121", "vgg16"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--log_dir", type=str, default="runs/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Datasets and loaders
    train_dataset = ChestXrayDataset(root=args.data_dir, split="train", transform=transform)
    val_dataset = ChestXrayDataset(root=args.data_dir, split="val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Class weights
    class_weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float)

    # Model
    model = get_model(args.model, num_classes=2)

    # Train
    best_acc, best_weights = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=torch.device(args.device),
        num_epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        log_dir=args.log_dir,
        model_name=args.model
    )

    # Load best weights
    model.load_state_dict(best_weights)
    model.to(args.device)
    model.eval()

    # Grad-CAM on one sample
    sample_img, sample_label = val_dataset[0]
    sample_img = sample_img.to(args.device)
    output = model(sample_img.unsqueeze(0))
    _, pred = torch.max(output, 1)

    # Grad-CAM visualization
    target_layer = model.layer4[-1] if "resnet" in args.model else model.features[-1]
    gradcam_img = generate_gradcam(model, sample_img, target_layer, torch.device(args.device))
    visualize_prediction(sample_img, sample_label, pred.item(), class_names=["Normal", "Pneumonia"], gradcam_img=gradcam_img)

if __name__ == "__main__":
    main()