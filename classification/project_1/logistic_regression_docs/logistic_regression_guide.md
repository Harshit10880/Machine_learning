# Logistic Regression: Comprehensive Guide

## What is Logistic Regression?

Logistic regression is a statistical method used for binary classification problems. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability of an event occurring (e.g., pass/fail, yes/no, spam/not spam). It uses the logistic (sigmoid) function to map the output to a probability between 0 and 1.

The logistic function is:  
σ(z) = 1 / (1 + e^(-z))  

Where z is the linear combination of input features: z = w0 + w1*x1 + w2*x2 + ... + wn*xn

## Key Points

- **Binary Classification**: Designed for problems with two possible outcomes.
- **Probabilistic Output**: Provides probabilities, not just class labels.
- **Linear Decision Boundary**: Assumes a linear relationship between features and the log-odds of the outcome.
- **Maximum Likelihood Estimation**: Uses MLE to estimate parameters.
- **Regularization**: Can include L1 (Lasso) or L2 (Ridge) regularization to prevent overfitting.

## Advantages

- Simple and interpretable: Easy to understand and implement.
- Efficient: Fast training and prediction, especially for small to medium datasets.
- Probabilistic predictions: Outputs probabilities, useful for decision-making.
- Feature importance: Coefficients indicate feature importance.
- Handles multicollinearity: With regularization.
- No assumptions about feature distributions.

## Disadvantages

- Assumes linear relationship between features and log-odds.
- Sensitive to outliers.
- May underperform on complex, non-linear problems.
- Requires feature scaling for better performance.
- Not suitable for multi-class problems without modifications (e.g., One-vs-Rest).

## Applications

- Medical diagnosis (disease prediction)
- Credit scoring (loan approval)
- Spam detection
- Customer churn prediction
- Marketing (response prediction)
- Fraud detection
- Quality control (defect prediction)

## Example: Student Pass/Fail Prediction

Using the dataset in `data.csv`, we predict whether a student will pass based on hours studied, attendance percentage, and assignment score.

### Code Explanation (from python.py)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Features and target
X = df[['hours_studied', 'attendance', 'assignment_score']]
y = df['pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
```

**Step-by-step explanation:**
1. **Import libraries**: pandas for data handling, sklearn for ML.
2. **Load data**: Read CSV into DataFrame.
3. **Select features and target**: X are predictors, y is binary outcome.
4. **Split data**: 80% train, 20% test.
5. **Initialize model**: LogisticRegression from sklearn.
6. **Train model**: Fit on training data.
7. **Predict**: On test data.
8. **Evaluate**: Calculate accuracy.

### New Example: Predicting Customer Purchase Intent

For this new example, we'll create a synthetic dataset `data_2.csv` with features: age, income, browsing_time, and target: purchase (1 if purchased, 0 otherwise).

Dataset generation:
- Age: 18-65
- Income: 20000-100000
- Browsing_time: 1-60 minutes
- Purchase: Based on simple rules (e.g., if age > 30 and income > 50000 and browsing_time > 10, likely purchase)

### Code Explanation (from python_project.py)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
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

# Save to CSV
df.to_csv('data_2.csv', index=False)

# Features and target
X = df[['age', 'income', 'browsing_time']]
y = df['purchase']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Coefficients
print("Coefficients:", model.coef_)
```

**Step-by-step explanation:**
1. **Generate data**: Create synthetic dataset with numpy random.
2. **Save data**: To data_2.csv.
3. **Select features/target**: Age, income, browsing_time as X; purchase as y.
4. **Scale features**: Use StandardScaler for better performance.
5. **Split data**: 80/20 split.
6. **Model**: LogisticRegression.
7. **Train and predict**.
8. **Evaluate**: Accuracy and detailed report.
9. **Inspect coefficients**: To understand feature importance.

## Conclusion

Logistic regression is a fundamental algorithm in machine learning for binary classification. It provides interpretable results and works well on linearly separable data. For more complex problems, consider advanced models like random forests or neural networks.