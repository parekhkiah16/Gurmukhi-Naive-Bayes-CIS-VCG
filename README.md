### Gurmukhi Handwritten Character Classification using Naive Bayes
### CIS + VCG Project

Team Members:
Kiah Parekh — A027
Sneha Purswani — A034

### Project Overview
This project develops a machine-learning system for classifying handwritten Gurmukhi numeral characters using a Gaussian Naive Bayes classifier.

The system performs dataset exploration, image preprocessing, feature extraction using Histogram of Oriented Gradients (HOG) and 4×4 zoning, model training, evaluation and prediction of unseen handwritten samples.

### Dataset

The project uses a publicly available dataset of handwritten Gurmukhi numeral characters.

Dataset source:
https://github.com/siddharthapramanik771/Gurmukhi-Handwritten-Digit-Classification/archive/refs/heads/main.zip

The dataset contains:

10 numeral classes (0–9)
Separate train and test directories
32×32 grayscale handwritten images

The Colab notebook downloads the dataset automatically. The labeled images are explored and then used in a stratified 80:20 train/test split for the Naive Bayes experiment.

### Methodology

The project follows the following workflow:

Dataset → Exploration → Preprocessing → Feature Extraction → Train/Test Split → Gaussian Naive Bayes → Evaluation → Unseen Sample Prediction

1. Dataset Exploration
The dataset is examined to identify the available classes, image samples and class distribution.

2. Image Preprocessing
The images are:
Converted to grayscale
Contrast-normalized
Resized to 64×64 pixels
Normalized to values between 0 and 1

3. Feature Extraction
Two feature representations are used:

HOG (Histogram of Oriented Gradients): captures local stroke and gradient direction information.
4×4 Zoning: divides the image into 16 regions and calculates regional intensity information.

The HOG and zoning features are concatenated to form the final feature vector.

4. Classification
A Gaussian Naive Bayes classifier is trained using the extracted feature vectors.

5. Evaluation
The classifier is evaluated using:
Accuracy
Macro Precision
Macro Recall
Macro F1-score
Classification Report
Confusion Matrix

6. Unseen Sample Prediction
After training, a separate handwritten Gurmukhi numeral can be uploaded to the system. The trained classifier processes the image and returns the predicted class along with the top class probabilities.

### Results

The evaluated Gaussian Naive Bayes classifier achieved the following results:

Metric	Score
Accuracy	0.9000
Macro Precision	0.9150
Macro Recall	0.9000
Macro F1-score	0.9007

The complete classification report and confusion matrix are available in the executed Colab notebook.

### Google Colab Notebook

The complete executed notebook is available at:
notebook/Gurmukhi_Naive_Bayes_CIS_VCG.ipynb

### The notebook contains:

Dataset download
Dataset exploration
Class distribution analysis
Sample image visualization
Image preprocessing
HOG and zoning feature extraction
Stratified train/test split
Gaussian Naive Bayes training
Classification metrics
Confusion matrix
Prediction inspection
Unseen handwritten numeral prediction
Trained model saving
Local Execution

Create a virtual environment and install the required dependencies:

python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

To train the model:

python -m src.train

### To predict a handwritten image:
python -m src.predict --image path/to/image.png

### Project Structure
└──gurmukhi-naive-bayes-cis-vcg/
    ├── README.md
    ├── PROJECT_REPORT.md
    ├── requirements.txt
    ├── data/
    │   ├── DATASET_SOURCE.md
    │   └── raw/
    │       └── .gitkeep
    └── src/
        ├── __init__.py
        ├── config.py
        ├── data_loader.py
        ├── features.py
        ├── predict.py
        ├── preprocessing.py
        └── train.py

### Technologies Used
Python
Google Colab
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Scikit-image
Pillow
Joblib

### Conclusion
This project demonstrates a complete classical machine-learning pipeline for handwritten Gurmukhi numeral classification. Image preprocessing and feature extraction using HOG and zoning are combined with Gaussian Naive Bayes to classify handwritten samples and evaluate the resulting model using standard classification metrics and a confusion matrix.
