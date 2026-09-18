import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)

from .config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from .data_loader import collect_image_paths, split_train_test
from .features import build_feature_matrix

def main(max_per_class=0):
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = collect_image_paths(DATA_DIR, max_per_class, RANDOM_STATE)
    train_paths, test_paths, y_train, y_test = split_train_test(
        records, TEST_SIZE, RANDOM_STATE
    )

    X_train = build_feature_matrix(train_paths)
    X_test = build_feature_matrix(test_paths)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "num_classes": len(set(y_train)),
        "feature_count": int(X_train.shape[1]),
    }

    with open(OUTPUT_DIR/"metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR/"classification_report.csv"
    )

    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(max(10, len(labels)*0.35), max(8, len(labels)*0.35)))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix - Gurmukhi Naive Bayes")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"confusion_matrix.png", dpi=180)
    plt.show()
    plt.close()

    joblib.dump(
        {"model": model, "labels": labels},
        MODEL_DIR/"gurmukhi_naive_bayes.joblib"
    )

    print("\nEvaluation")
    print("----------")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()
    main(args.max_per_class)
