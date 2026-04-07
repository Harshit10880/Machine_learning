# Classification Projects Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Project 1: Iris Flower Classification](#project-1-iris-flower-classification)
3. [Project 2: Wine Classification](#project-2-wine-classification)
4. [Project 3: Breast Cancer Classification](#project-3-breast-cancer-classification)
5. [Project 4: Handwritten Digits Classification](#project-4-handwritten-digits-classification)
6. [Project 5: Wine Classification with SVM](#project-5-wine-classification-with-svm)
7. [Visualization Project](#visualization-project)
8. [Common Concepts](#common-concepts)
9. [How to Run](#how-to-run)

## Introduction

This documentation provides detailed explanations of the classification projects in this repository. Each project demonstrates supervised machine learning concepts using different datasets and algorithms. The projects are designed to be progressive, starting with simple implementations and gradually introducing more complex datasets and techniques.

All projects use Python with scikit-learn, pandas, and other standard ML libraries.

## Project 1: Iris Flower Classification

### Overview
This is the foundational project that introduces basic classification concepts using the famous Iris dataset.

### Dataset Description
- **Source**: Built-in scikit-learn dataset
- **Samples**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Type**: Multiclass classification

### Code Explanation

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Features and target
x = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

# Create KNN classifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, Y_train)

# Make predictions on test data
pred = knn.predict(X_test)

# Calculate accuracy
acc = accuracy_score(Y_test, pred)
```

### Key Steps
1. **Data Loading**: Load iris dataset using `load_iris()`
2. **Data Preparation**: Convert to pandas DataFrame and separate features (X) from target (y)
3. **Train-Test Split**: Split data into 80% training and 20% testing
4. **Model Creation**: Initialize KNN classifier with k=3
5. **Training**: Fit the model on training data
6. **Prediction**: Predict on test data
7. **Evaluation**: Calculate accuracy score

### Expected Output
- Accuracy typically ranges from 0.93 to 1.0 depending on the random split

## Project 2: Wine Classification

### Overview
This project extends the classification concepts to a more complex dataset with chemical features.

### Dataset Description
- **Source**: Built-in scikit-learn dataset
- **Samples**: 178
- **Features**: 13 (alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 of diluted wines, proline)
- **Classes**: 3 (wine cultivars)
- **Type**: Multiclass classification

### Code Explanation

```python
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Features and target
x = df.drop('target', axis=1)
y = df['target']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, Y_train)

# Make predictions
pred = knn.predict(X_test)

# Calculate accuracy
acc = accuracy_score(Y_test, pred)
print(f"Accuracy: {acc:.2f}")

# Optional: Print some predictions
print("Sample predictions:")
for i in range(5):
    print(f"Predicted: {pred[i]}, Actual: {Y_test.iloc[i]}")
```

### Key Differences from Project 1
- Uses `random_state=42` for reproducible results
- Includes print statements for output
- More features (13 vs 4) making it more complex

### Expected Output
- Accuracy typically around 0.70-0.80
- Sample predictions showing predicted vs actual values

## Project 3: Breast Cancer Classification

### Overview
This project demonstrates binary classification using a medical dataset.

### Dataset Description
- **Source**: Built-in scikit-learn dataset
- **Samples**: 569
- **Features**: 30 (computed from digitized images of breast mass)
- **Classes**: 2 (malignant, benign)
- **Type**: Binary classification

### Code Structure
Similar to Project 2, but with breast cancer dataset loaded via `load_breast_cancer()`.

### Key Features
- Binary classification (2 classes vs 3 in previous projects)
- Medical application context
- Higher dimensional feature space

## Project 4: Handwritten Digits Classification

### Overview
This project introduces image classification using the digits dataset.

### Dataset Description
- **Source**: Built-in scikit-learn dataset
- **Samples**: 1797
- **Features**: 64 (8x8 pixel grayscale images flattened)
- **Classes**: 10 (digits 0-9)
- **Type**: Multiclass classification

### Code Structure
Similar to previous projects, using `load_digits()`.

### Key Features
- 10 classes (most complex so far)
- Image data (though pre-processed)
- Larger dataset

## Project 5: Wine Classification with SVM

### Overview
This project demonstrates using a different algorithm (SVM) on the same wine dataset.

### Algorithm
- **Support Vector Machine (SVM)** with linear kernel
- Different from KNN used in previous projects

### Code Explanation

```python
# Create SVM classifier
svm = SVC(kernel='linear', random_state=42)

# Train the model
svm.fit(X_train, Y_train)

# Make predictions
pred = svm.predict(X_test)
```

### Key Differences
- Uses `SVC` instead of `KNeighborsClassifier`
- Introduces different ML algorithm
- Same dataset for comparison

## Visualization Project

### Overview
Located in `visulize_prob_1.py`, this file demonstrates data visualization techniques.

### Features
- Uses matplotlib and seaborn
- Likely includes plots for the classification datasets
- Helps in understanding data distribution and relationships

## Common Concepts

### Supervised Learning
All projects use supervised learning where we have labeled training data.

### Train-Test Split
- Training data: Used to train the model
- Testing data: Used to evaluate performance
- Prevents overfitting

### K-Nearest Neighbors (KNN)
- Distance-based algorithm
- k=3 means considers 3 nearest neighbors
- Simple but effective for small datasets

### Accuracy Score
- Measures percentage of correct predictions
- Formula: (correct predictions) / (total predictions)

### Random State
- Ensures reproducible results
- Same random_state gives same train-test split

## How to Run

1. Ensure all dependencies are installed:
   ```bash
   pip install scikit-learn pandas matplotlib seaborn numpy
   ```

2. Navigate to the classification folder:
   ```bash
   cd Machine_learning/classification
   ```

3. Run any project:
   ```bash
   python classification_1.py
   ```

4. View the output in the terminal.

## Next Steps

After understanding these projects, you can:
- Experiment with different values of k in KNN
- Try other algorithms (Decision Trees, Random Forest, etc.)
- Add more evaluation metrics (precision, recall, F1-score)
- Implement cross-validation
- Add feature scaling
- Create your own visualizations

This documentation provides a foundation for understanding the code and concepts. Each project builds upon the previous one, creating a learning progression in machine learning classification.