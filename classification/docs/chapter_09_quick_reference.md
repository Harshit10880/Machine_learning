# Chapter 9: Quick Reference Guide

## 9.1 Essential Code Snippets

### Data Loading Template
```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Quick exploration
print(df.head())
print(df.describe())
print(df['target'].value_counts())
```

### Model Training Template
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

### Cross-Validation Template
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

## 9.2 Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv ml_env
ml_env\Scripts\activate  # Windows

# Install packages
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

### Jupyter Notebook
```bash
# Start Jupyter
jupyter notebook

# Common imports in notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

---

**[← Back to Main Guide](../classification_guide.md)** | **[End of Guide](../classification_guide.md)**