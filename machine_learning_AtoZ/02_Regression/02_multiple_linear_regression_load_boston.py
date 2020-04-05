"""
Boston house prices dataset
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()
columns = boston.feature_names
# check Description
#print(boston.DESCR)

# importing the dataset
X = pd.DataFrame(boston.data, columns=columns)
X = X.values
y = pd.DataFrame(boston.target)
y = y.values
#print(X.info())
#print(X.isnull().sum())

# splitting the dataset into the Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

score = lr.score(X_test, y_test)
print(score)

import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((506,1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,4,5,6,8,9,10,11,12,13]]
#regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
#print(regressor_OLS.summary())

'''X_opt = X[:, [0,1,2,3,4]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())'''


# splitting the dataset into the Training and test set by X_opt
from sklearn.model_selection import train_test_split
X_opt_train, X_opt_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.20)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_opt_train, y_train)
#y_opt_pred = lr.predict(X_opt_test)

score = lr.score(X_opt_test, y_test)
print(score)