from pathlib import Path
import random

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def discover_class_folders(root):
    root = Path(root)
    folders = []
    for d in sorted(root.rglob("*")):
        if d.is_dir():
            if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir()):
                folders.append(d)
    return folders

def collect_image_paths(root, max_per_class=0, seed=42):
    folders = discover_class_folders(root)
    if not folders:
        raise FileNotFoundError(
            f"No class folders containing images found under {root}. "
            "Check the dataset extraction and folder structure."
        )

    rng = random.Random(seed)
    records = []

    for folder in folders:
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_per_class and len(images) > max_per_class:
            images = rng.sample(images, max_per_class)
        records.extend((p, folder.name) for p in images)

    return records

def split_train_test(records, test_size=0.2, seed=42):
    from sklearn.model_selection import train_test_split

    paths = [str(p) for p, _ in records]
    labels = [label for _, label in records]

    return train_test_split(
        paths, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
