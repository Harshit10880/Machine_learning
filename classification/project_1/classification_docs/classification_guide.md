# Classification: Comprehensive Guide

## What is Classification?

Classification is a type of supervised machine learning task where the goal is to assign input examples to one of two or more discrete classes. In contrast to regression, which predicts continuous values, classification predicts categorical labels such as pass/fail, spam/not spam, or disease/no disease.

## Key Points

- **Supervised learning**: Trained on labeled data.
- **Discrete target**: Output variable takes categorical values.
- **Binary and multi-class**: Can handle two classes or more.
- **Evaluation metrics**: Accuracy, precision, recall, F1-score, confusion matrix.
- **Common classifiers**: Decision trees, random forests, K-nearest neighbors, support vector machines.

## Advantages

- Directly solves decision-making problems.
- Easy to interpret for many models.
- Works well with labeled datasets.
- Provides clear performance metrics.

## Disadvantages

- Requires good feature engineering.
- May overfit if model is too complex.
- Some algorithms do not scale well for large datasets.
- Performance depends on data quality.

## Applications

- Student pass/fail prediction.
- Email spam detection.
- Medical diagnosis.
- Loan approval.
- Fraud detection.
- Image recognition.

## Example 1: Student Pass/Fail Classification

This example uses `data.csv` inside the `project_1` folder. The model predicts whether a student passes based on:
- `hours_studied`
- `attendance`
- `assignment_score`

### Code Explanation for `python.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data.csv")

# Select features and target
X = df[['hours_studied', 'attendance', 'assignment_score']]
y = df['pass']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Step-by-step explanation:**
1. Load the student dataset from `data.csv`.
2. Choose the input features and the target label.
3. Split data into training and testing sets to evaluate generalization.
4. Use a `DecisionTreeClassifier` to learn patterns from the training examples.
5. Predict on unseen test data and measure accuracy, precision, recall, and F1-score.

## Example 2: Customer Purchase Classification

This example uses `data_2.csv` and builds a classification model to predict if a customer will make a purchase. Features include:
- `age`
- `income`
- `browsing_time`

### Code Explanation for `python_project.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic classification data
np.random.seed(42)
n_samples = 300
age = np.random.randint(18, 66, n_samples)
income = np.random.randint(20000, 100001, n_samples)
browsing_time = np.random.randint(1, 61, n_samples)
purchase = ((age > 30) & (income > 50000) & (browsing_time > 10)).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'browsing_time': browsing_time,
    'purchase': purchase
})

# Save synthetic dataset to CSV
df.to_csv('data_2.csv', index=False)

# Select features and target
X = df[['age', 'income', 'browsing_time']]
y = df['purchase']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Step-by-step explanation:**
1. Generate a synthetic customer dataset and save it to `data_2.csv`.
2. Use age, income, and browsing time as classification features.
3. Scale numeric values before training.
4. Split into training and testing data.
5. Train a decision tree classifier.
6. Evaluate using accuracy and class-based performance metrics.

## Conclusion

This guide now focuses on classification problems and workflows. The two example scripts inside `project_1` demonstrate classification using real and synthetic datasets, with clear evaluation metrics showing how well the model performs.
