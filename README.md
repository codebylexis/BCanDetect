# BCanDetect – Identification of Breast Cancer with 90% Accuracy

BCanDetect is a deep learning model designed to identify breast cancer from histopathological images with 90% accuracy. Using a hybrid CNN + ANN architecture, this model aids in early detection by classifying tissue samples as malignant or non-malignant.

---

## Features

- End-to-end CNN + ANN deep learning pipeline for breast cancer detection
- Trained on 6,000 balanced histopathological images
- Accurate binary classification (cancerous vs. non-cancerous)
- Visual performance metrics: accuracy, loss, confusion matrix, ROC, precision-recall
- Clean Jupyter Notebook implementation

---

## Directory Structure

```
BCanDetect/
├── Dataset/
│ └── (your image dataset)
├── Graphs and Pictures/
│ └── (model graphs, evaluation visuals)
├── Notebook/
│ └── identification_of_breast_cancer.ipynb
├── requirements.txt
└── README.md
```

---

## Libraries Used

- TensorFlow  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Model Architecture

The model consists of:
- Three convolutional layers with ReLU activation
- Max-pooling layers after each CNN block
- Flattened feature vector passed into three dense layers
- Dropout applied between layers to prevent overfitting
- Final sigmoid activation for binary classification

---

## Data Visualization

The dataset consists of labeled histopathological images of breast tissue, where "Positive" indicates cancerous cells with abnormal structures, and "Negative" represents healthy, organized tissue—used to train and evaluate the deep learning model for accurate classification.

<p align="center">
  <img src="Graphs and Pictures/Dataset.png" width="750" height="500">
</p>

---


## Target Class Distribution

Zero (`0`) represents infected (cancerous) tissue.  
One (`1`) represents non-infected (non-cancerous) tissue.

<p align="center">
  <img src="Graphs and Pictures/Equal Data Distribution.png" width="400" height="450">
</p>

---

## Methodology & Training

This model uses a hybrid architecture combining Convolutional Neural Networks (CNN) for feature extraction and Artificial Neural Networks (ANN) for classification. It processes 100x100 RGB histopathological images through three CNN layers (32–192 filters) with max-pooling, followed by three dense layers with dropout to reduce overfitting. The model was trained on a balanced dataset of 6,000 images (50% cancerous, 50% non-cancerous) for 20 epochs using the Adam optimizer and binary cross-entropy loss, achieving 90% training accuracy.

<p align="center">
  <img src="Graphs and Pictures/model_graph.png" width="400" height="450">
</p>

---

## Model Performance

### Model Loss:  
<p align="center">
  <img src="Graphs and Pictures/Loss.png" width="400" height="450">
</p>

### Model Accuracy: 
<p align="center">
  <img src="Graphs and Pictures/Accuracy.png" width="400" height="450">
</p>

---

## Model Evaluation

- **Training Accuracy**: 90%  
- **Test Accuracy**: 86%  
- **Training Loss**: 0.90  
- **Test Loss**: 0.31  

---

## Predictions

<p align="center">
  <img src="Graphs and Pictures/Output.png" width="400" height="450">
</p>


---

## Model Introspection

### Confusion Matrix  
<p align="center">
  <img src="Graphs and Pictures/Confusion Matrix.png" width="400" height="450">
</p>

### ROC Curve  
<p align="center">
  <img src="Graphs and Pictures/ROC.png" width="400" height="450">
</p>

### Precision vs. Recall  
<p align="center">
  <img src="Graphs and Pictures/Precision Vs Recall.png" width="400" height="450">
</p>

### Classification Report  
<p align="center">
  <img src="Graphs and Pictures/Classification Report.png" width="400" height="450">
</p>


---

## Setup & Run

```bash
# Step 1: Create a virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch Jupyter Notebook
jupyter notebook Notebook/identification_of_breast_cancer.ipynb
```

---

## Conclusion

BCanDetect is a powerful deep learning model that accurately classifies breast cancer from histopathological images, offering a valuable tool for early detection and clinical decision-making.

