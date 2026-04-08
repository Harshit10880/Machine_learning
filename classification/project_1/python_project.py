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

# Features and target
X = df[['age', 'income', 'browsing_time']]
y = df['purchase']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Classification model
model = DecisionTreeClassifier(random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))