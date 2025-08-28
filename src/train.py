import argparse
import torch
from model import get_model, train_model
from utils import set_seed, generate_gradcam, visualize_prediction
from data_loader import get_data_loaders
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description="Pneumonia Detection Training")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "densenet121", "vgg16"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="DATA/")
    parser.add_argument("--log_dir", type=str, default="runs/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed()

    print("Loading data...")
    loaders = get_data_loaders(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    class_weights = loaders["class_weights"]
    class_names = loaders["class_names"]

    print(f"Using model: {args.model}")
    model = get_model(args.model, num_classes=len(class_names))

    print("Starting training...")
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

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print("Loading best model weights...")
    model.load_state_dict(best_weights)
    model.to(args.device)
    model.eval()

    print("Generating Grad-CAM for one validation sample...")
    sample_img, sample_label = next(iter(val_loader))
    sample_img = sample_img[0].to(args.device)
    sample_label = sample_label[0].item()
    output = model(sample_img.unsqueeze(0))
    _, pred = torch.max(output, 1)

    if "resnet" in args.model:
        target_layer = model.layer4[-1]
    elif "densenet" in args.model:
        target_layer = model.features[-1]
    elif "vgg" in args.model:
        target_layer = model.features[-1]
    else:
        raise ValueError("Unsupported model for Grad-CAM")

    gradcam_img = generate_gradcam(model, sample_img, target_layer, torch.device(args.device))
    visualize_prediction(sample_img, sample_label, pred.item(), class_names=class_names, gradcam_img=gradcam_img)

if __name__ == "__main__":
    main()