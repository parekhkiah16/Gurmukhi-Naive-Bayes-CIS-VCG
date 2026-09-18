# Gurmukhi Handwritten Character Classification using Naive Bayes

CIS + VCG project using image preprocessing, HOG features, zoning features and Gaussian Naive Bayes.

## Dataset

This runnable version uses a publicly accessible Gurmukhi handwritten numeral dataset:

https://github.com/siddharthapramanik771/Gurmukhi-Handwritten-Digit-Classification/archive/refs/heads/main.zip

The source repository documents a dataset with separate `train` and `test` folders, 10 Gurmukhi numeral classes (0–9), and 32×32 grayscale images.

The Colab notebook downloads the dataset automatically, combines the labeled train/test folders for exploration, then performs a fresh stratified 80/20 split for the Naive Bayes experiment.

> Note: HWR-Gurmukhi_1.1 is a different 3,500-image/35-class benchmark documented in the literature. A verified direct public download endpoint for that exact benchmark could not be confirmed, so this repository uses the downloadable dataset above instead.

## Colab

Open `notebook/Gurmukhi_Naive_Bayes_CIS_VCG.ipynb` in Google Colab and run the cells from top to bottom.

The notebook:
1. Downloads the dataset automatically.
2. Explores the classes and images.
3. Preprocesses images.
4. Extracts HOG + zoning features.
5. Performs an 80/20 stratified split.
6. Trains Gaussian Naive Bayes.
7. Reports accuracy, precision, recall and F1.
8. Generates a confusion matrix.
9. Demonstrates predictions on unseen test images.
10. Lets you upload a new handwritten Gurmukhi numeral for prediction.
11. Saves the trained model and output figures.

## Local execution

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python -m src.train
python -m src.predict --image path/to/image.png
```

## Project structure

```text
Gurmukhi_Naive_Bayes_CIS_VCG/
├── notebook/
├── src/
├── data/
├── models/
├── outputs/
├── PROJECT_REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```
