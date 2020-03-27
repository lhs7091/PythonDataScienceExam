"""
Support Vector Regression
Support Vector Machines suppprt linear and nonlinear regression
 that we can refer to as SVR.
Instead of trying to fit the largest possible street between two classes while limiting margin violations,
 SVR tries to fit as many instances as possible on the street while limiting margin violations.

1. Collect a training set X, Y
2. Choose a kernel and it's parameters as well as any regularization needed
    -> Kernel: Gaussian(RBF),, Regularization: Noise
3. Form the correlation matrix, K
4. Train your machine, exactly or approximately, to get contraction coefficients
5. Use those coefficients, create your estimator

"""
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


# importing the dataset
dataset = pd.read_csv('./data/SVR.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

# SVR needs feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the dataset
svr = SVR(kernel='rbf')
print(svr.fit(X, y))

# Predicting a new result
y_pred = sc_y.inverse_transform(svr.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)

'''plt.scatter(X, y, color='r')
plt.plot(X, svr.predict(X), color='b')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()'''

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, svr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()