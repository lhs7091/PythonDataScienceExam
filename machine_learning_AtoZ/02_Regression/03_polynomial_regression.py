# Polynomial Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('./data/Position_Salaries.csv')
X = dataset.drop(['Position','Salary'], axis=1).values
y = dataset['Salary'].values

# Splitting the dataset into the Training set and Test set
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# Fitting polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=4)
X_poly = pr.fit_transform(X)

lr_2 = LinearRegression()
lr_2.fit(X_poly, y)

# Visualising the Linear Regression results
'''plt.scatter(X, y, color='r')
plt.plot(X, lr.predict(X), color='b')
plt.title('truth or bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')'''

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='r')
plt.plot(X, lr_2.predict(pr.fit_transform(X)), color='b')
plt.title('truth or bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()
