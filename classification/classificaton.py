import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
#print(df.head())
x = df.drop('target', axis=1)
y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)
pred = knn.predict(X_test)

acc = accuracy_score(Y_test, pred)