import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Literal

def mask_region(
    input_tensor: torch.Tensor,
    region: Literal["center", "top-left", "top-right", "bottom-left", "bottom-right"] = "center",
    size: int = 50,
    mask_value: float = 0.0
) -> torch.Tensor:
    """
    Mask a square region of the image tensor to simulate counterfactual input.

    Args:
        input_tensor: Tensor of shape (1, C, H, W).
        region: Region to mask.
        size: Size of the square mask.
        mask_value: Value to fill the masked region.

    Returns:
        Masked image tensor.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Expected input tensor of shape (1, C, H, W)")

    img = input_tensor.clone()
    _, _, H, W = img.shape

    # Determine center of region
    region_map = {
        "center": (W // 2, H // 2),
        "top-left": (W // 4, H // 4),
        "top-right": (3 * W // 4, H // 4),
        "bottom-left": (W // 4, 3 * H // 4),
        "bottom-right": (3 * W // 4, 3 * H // 4)
    }

    if region not in region_map:
        raise ValueError(f"Unsupported region: {region}")

    cx, cy = region_map[region]

    # Clamp boundaries
    x1, x2 = max(cx - size // 2, 0), min(cx + size // 2, W)
    y1, y2 = max(cy - size // 2, 0), min(cy + size // 2, H)

    img[:, :, y1:y2, x1:x2] = mask_value
    return img


def visualize_masked_input(
    original_image: Image.Image,
    region: Literal["center", "top-left", "top-right", "bottom-left", "bottom-right"] = "center",
    size: int = 50,
    outline_color: str = "red",
    fill_color: str = None
) -> Image.Image:
    """
    Draw a rectangle on the original image to show masked region.

    Args:
        original_image: PIL image.
        region: Region to visualize.
        size: Size of the mask.
        outline_color: Border color of the rectangle.
        fill_color: Optional fill color inside the rectangle.

    Returns:
        Image with rectangle overlay.
    """
    img = original_image.copy()
    W, H = img.size

    region_map = {
        "center": (W // 2, H // 2),
        "top-left": (W // 4, H // 4),
        "top-right": (3 * W // 4, H // 4),
        "bottom-left": (W // 4, 3 * H // 4),
        "bottom-right": (3 * W // 4, 3 * H // 4)
    }

    if region not in region_map:
        raise ValueError(f"Unsupported region: {region}")

    cx, cy = region_map[region]
    x1, x2 = max(cx - size // 2, 0), min(cx + size // 2, W)
    y1, y2 = max(cy - size // 2, 0), min(cy + size // 2, H)

    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3, fill=fill_color)

    return img