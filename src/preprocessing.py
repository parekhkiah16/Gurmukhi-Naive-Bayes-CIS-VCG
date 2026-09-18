from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .config import IMAGE_SIZE

def preprocess_image(path_or_image):
    """Convert to grayscale, improve contrast, resize, and normalize."""
    if isinstance(path_or_image, (str, Path)):
        image = Image.open(path_or_image)

    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize(IMAGE_SIZE)

    return np.asarray(image, dtype=np.float32) / 255.0
