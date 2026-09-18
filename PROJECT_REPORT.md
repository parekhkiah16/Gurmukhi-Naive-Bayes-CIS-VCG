# CIS & VCG Project Report
## Naive Bayes-Based Classification of Handwritten Gurmukhi Characters

### 1. Introduction
Handwritten character recognition is a pattern-recognition problem in which a computer identifies characters from handwritten images. Gurmukhi is an Indic script used for Punjabi. Variation in handwriting makes automatic recognition challenging.

This project develops a classical machine-learning system using Gaussian Naive Bayes to classify isolated handwritten Gurmukhi character images.

### 2. Problem Statement
To develop a Naive Bayes-based system for classifying handwritten Gurmukhi characters using a publicly available Gurmukhi character dataset, including dataset exploration, image preprocessing, feature preparation, unseen-sample prediction and evaluation.

### 3. Objectives
- Explore the dataset and class distribution.
- Preprocess handwritten character images.
- Prepare HOG and zoning features.
- Train a Gaussian Naive Bayes classifier.
- Predict unseen character samples.
- Evaluate accuracy, precision, recall and F1-score.
- Generate and interpret a confusion matrix.
- Publish the Python implementation on GitHub.

### 4. Dataset

The project uses a publicly available dataset of handwritten Gurmukhi numeral characters. The dataset contains 10 classes corresponding to the numerals 0–9 and provides separate training and testing directories containing grayscale character images.

The original images are 32×32 pixels. The dataset is obtained automatically from the public source repository used by the project.

Dataset source:
https://github.com/siddharthapramanik771/Gurmukhi-Handwritten-Digit-Classification

### 5. System Requirements
Hardware: laptop/computer with internet access and sufficient RAM.
Software: Google Colab, Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Scikit-image, Pillow and Joblib.

### 6. Methodology
Dataset → preprocessing → HOG + zoning features → Gaussian Naive Bayes → prediction → evaluation.

### 7. Image Preprocessing
Images are converted to grayscale, contrast-normalized, resized to 64×64 pixels and normalized to values between 0 and 1.

### 8. Feature Extraction
HOG captures local gradient and stroke direction information. Zoning divides each image into a 4×4 grid and calculates regional intensity values. The features are concatenated.

### 9. Naive Bayes Classification
Gaussian Naive Bayes is trained using the prepared feature vectors and class labels.

### 10. Training and Testing
An 80:20 stratified train/test split is used. The random seed is fixed at 42 for reproducibility.

### 11. Unseen Character Prediction
A separate handwritten character image is uploaded after training. The trained model returns its predicted class and the top five class probabilities.

### 12. Evaluation

The trained Gaussian Naive Bayes classifier was evaluated on the test set using standard classification metrics. The following results were obtained:

Metric	Score
Accuracy	0.9000
Macro Precision	0.9150
Macro Recall	0.9000
Macro F1-score	0.9007

A detailed classification report was generated for all 10 numeral classes (0–9), including precision, recall, F1-score and support for each class.
A confusion matrix was also generated to visualize the classification performance across the different Gurmukhi numeral classes.

### 13. Results

The Gaussian Naive Bayes classifier achieved an accuracy of 90.00% on the evaluated test set. The macro precision was 91.50%, the macro recall was 90.00%, and the macro F1-score was 90.07%.
The results show that the trained classifier was able to correctly classify a large majority of the handwritten Gurmukhi numeral samples. The classification report provides class-wise precision, recall, F1-score and support, while the confusion matrix shows the distribution of correct and incorrect predictions across the 10 classes.
The completed Google Colab notebook contains the detailed classification report, confusion matrix, sample predictions and unseen handwritten character prediction generated during the project execution.

### 14. Conclusion
The developed system demonstrates how a classical Naive Bayes classifier can be applied to handwritten Gurmukhi character classification after suitable image preprocessing and feature extraction.

### 15. Future Scope
- Compare Naive Bayes with SVM, Random Forest and k-NN.
- Experiment with additional feature descriptors.
- Increase the amount and diversity of training data.
- Investigate deep-learning approaches such as CNNs for comparison.
- Develop a simple graphical interface for handwritten character recognition.

### 16. References
1. Kumar, M. et al., “Benchmark Datasets for Offline Handwritten Gurmukhi Script Recognition.”
2. Kaur, K., Chaudhuri, B. B., Lehal, G. S., “Gurmukhi Handwritten Character Recognition for Children to Old: A Benchmark Dataset,” SN Computer Science, 2026.
3. Scikit-learn documentation for Gaussian Naive Bayes and classification metrics.
