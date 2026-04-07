import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
df = pd.DataFrame(data=digits.data)
df['target'] = digits.target

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