    import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.linear_model import LinearRegression
#from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

obj = fetch_california_housing()

df = pd.DataFrame(obj.data, columns=obj.feature_names)
df['price'] = obj.target

x = df['AveBedrms']
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Reshape x_train and x_test to be 2D arrays
x_train_reshaped = x_train.values.reshape(-1, 1)
x_test_reshaped = x_test.values.reshape(-1, 1)

model = LinearRegression()

model.fit(x_train_reshaped, y_train)

prediction = model.predict(x_test_reshaped)

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
# print(predictions.head())

plt.scatter(x_test_reshaped, y_test, color='blue', label='Actual')

# Plot the regression line
plt.plot(x_test_reshaped, prediction, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Average Number of Bedrooms (AveBedrms)')
plt.ylabel('House Price ($100,000s)')
plt.title('Simple Linear Regression: Average Bedrooms vs. House Price')
plt.legend()
plt.show()