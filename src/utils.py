import os
import torch
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def set_seed(seed=42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"🔒 Seed set to {seed}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"📁 Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()

def generate_gradcam(model, input_tensor, target_layer, device, class_idx=None, save_path=None):
    """
    Generate Grad-CAM heatmap for a single image tensor.

    Args:
        model: Trained model
        input_tensor: Image tensor (C, H, W)
        target_layer: Layer to visualize
        device: torch.device
        class_idx: Optional target class index
        save_path: Optional path to save Grad-CAM image

    Returns:
        visualization: Grad-CAM overlay image (numpy array)
    """
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    target = None if class_idx is None else ClassifierOutputTarget(class_idx)
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=None if target is None else [target])

    input_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())  # normalize
    visualization = show_cam_on_image(input_np, grayscale_cam[0], use_rgb=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, visualization)
        logging.info(f"🖼️ Grad-CAM saved to {save_path}")

    return visualization

def visualize_prediction(image_tensor, label, pred, class_names, gradcam_img=None, title=None):
    """Display image with prediction and optional Grad-CAM overlay."""
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    plt.figure(figsize=(5, 5))
    plt.imshow(gradcam_img if gradcam_img is not None else image_np)
    display_title = title if title else f"True: {class_names[label]} | Pred: {class_names[pred]}"
    plt.title(display_title)
    plt.axis("off")
    plt.show()

def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"📦 Loaded model_state_dict from checkpoint: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        logging.info(f"📦 Loaded raw state_dict from checkpoint: {checkpoint_path}")

    return model

def get_target_layer(model, model_name):
    """
    Return the appropriate target layer for Grad-CAM based on model architecture.
    """
    if "resnet" in model_name:
        return model.layer4[-1]
    elif "densenet" in model_name:
        return model.features[-1]
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")