"""
Decision Tree
"""

# Importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./data/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)

# Fitting the Decision Tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X, y)
print(dtr)

# Predicting a new result
y_pred = dtr.predict([[6.5]])
print(y_pred)

# Visualising the Regression results(for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, dtr.predict(X_grid), color='b')
plt.title('Truth or Bluff(Decision Tree Regression')
plt.show()