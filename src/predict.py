import argparse
from pathlib import Path
import joblib
import numpy as np
from .config import MODEL_DIR
from .features import extract_features

def predict(image_path):
    artifact = joblib.load(MODEL_DIR/"gurmukhi_naive_bayes.joblib")
    model = artifact["model"]

    x = extract_features(image_path).reshape(1, -1)
    probabilities = model.predict_proba(x)[0]
    classes = model.classes_

    order = np.argsort(probabilities)[::-1]
    return [
        (str(classes[i]), float(probabilities[i]))
        for i in order[:5]
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    results = predict(args.image)
    print(f"Predicted character: {results[0][0]}")
    print(f"Confidence: {results[0][1]:.4f}")
    print("\nTop 5 predictions:")
    for label, probability in results:
        print(f"{label}: {probability:.4f}")
