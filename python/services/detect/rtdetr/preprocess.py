"""RT-DETR input preparation."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .config import INPUT_SIZE
from utils.logger import log


def preprocess(image_path: str) -> tuple[np.ndarray, int, int]:
    """Load image -> NCHW tensor (stretch to 640^2, rescale 1/255, no mean/std)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    log.debug(f"RT-DETR preprocess: image {w}x{h}")

    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) * (1.0 / 255.0)
    tensor = arr.transpose(2, 0, 1)[np.newaxis]
    tensor = np.ascontiguousarray(tensor)
    log.debug(f"RT-DETR preprocess: tensor{list(tensor.shape)}")
    return tensor, w, h
