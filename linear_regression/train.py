#!/usr/bin/python3
"""Trying to get the relationship between petal's length(y) and sepal's length(x) of a flower"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('iris.csv', header=None)
df = df.dropna()
print(df.describe())
print(df.shape)

# Selecting petal length (column index 2) and sepal length (column index 0) as features
X = df.iloc[:, [0]].values # Sepal length
y = df.iloc[:, 2].values  # Petal length

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predicting petal length using the sepal length from the test set
predictions = model.predict(X_test)
predict = model.predict([[6.3]])
print(f"prediction of 6.3 is {predict}")

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"R2 Score: {r2}")
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, predictions)
plt.xlabel('Actual petal length')
plt.ylabel('Predicted pretal length')
plt.title('Actual petal vs predicted petal')
plt.show()
