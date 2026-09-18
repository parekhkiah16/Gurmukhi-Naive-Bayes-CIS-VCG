import numpy as np
from skimage.feature import hog
from .config import (
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
    ZONES,
)
from .preprocessing import preprocess_image

def zoning_features(image, zones=ZONES):
    h, w = image.shape
    rows, cols = zones
    values = []

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*h//rows, (r+1)*h//rows
            x0, x1 = c*w//cols, (c+1)*w//cols
            values.append(float(image[y0:y1, x0:x1].mean()))

    return np.asarray(values, dtype=np.float32)

def extract_features(path):
    image = preprocess_image(path)

    hog_features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    zone_features = zoning_features(image)
    return np.concatenate([hog_features, zone_features]).astype(np.float32)

def build_feature_matrix(paths):
    return np.vstack([extract_features(p) for p in paths])
