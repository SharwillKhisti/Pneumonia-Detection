import torch
import numpy as np
from captum.attr import IntegratedGradients
from PIL import Image
import matplotlib.cm as cm
from typing import Tuple

def get_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: str = "cpu",
    n_steps: int = 50
) -> np.ndarray:
    """
    Compute Integrated Gradients (IG) attributions for a given input and target class.

    Args:
        model: Trained PyTorch model.
        input_tensor: Input tensor of shape (1, C, H, W).
        target_class: Index of the target class.
        device: "cpu" or "cuda".
        n_steps: Number of integration steps.

    Returns:
        Normalized attributions as a NumPy array of shape (H, W, C).
    """
    model.to(device).eval()
    input_tensor = input_tensor.to(device)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=n_steps)

    attributions = attributions.squeeze().detach().cpu().numpy()
    if attributions.ndim == 3:
        attributions = np.transpose(attributions, (1, 2, 0))  # CHW → HWC
    else:
        raise ValueError("Expected attribution shape (C, H, W), got {}".format(attributions.shape))

    return attributions


def show_ig_overlay(
    attributions: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.5,
    cmap: str = "jet",
    resize: bool = True
) -> Tuple[Image.Image, Image.Image]:
    """
    Overlay Integrated Gradients heatmap on the original image.

    Args:
        attributions: IG attributions of shape (H, W, C).
        original_image: Original PIL image.
        alpha: Transparency for blending.
        cmap: Matplotlib colormap.
        resize: Whether to resize heatmap to match original image.

    Returns:
        Tuple of (heatmap image, blended overlay image).
    """
    if attributions.ndim != 3:
        raise ValueError("Attributions must be 3D (H, W, C), got shape {}".format(attributions.shape))

    heatmap = np.mean(attributions, axis=-1)

    # Normalize safely
    min_val, max_val = heatmap.min(), heatmap.max()
    if max_val > min_val:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    # Apply colormap
    colormap = cm.get_cmap(cmap)(heatmap)[..., :3]  # Drop alpha
    heatmap_img = Image.fromarray((colormap * 255).astype(np.uint8))

    # Resize if needed
    if resize and heatmap_img.size != original_image.size:
        heatmap_img = heatmap_img.resize(original_image.size, resample=Image.BILINEAR)

    blended_img = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)

    return heatmap_img, blended_img