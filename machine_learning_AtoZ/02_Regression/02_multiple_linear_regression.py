'''
Multiple Linear Regression
 y = b0 + b1*x1 + b2*x2 +....+ bn*xn

we want to predict that they have some exracts from their profits
and loss statements from their income by 5 4olumns
Colums
  R&D Spend ($)
  Administration ($)
  Marketing Spend ($)
  State (region)
    -> we have to change int values by dummy function(one-hot encoding)
    -> then now it has 2 values in State, New York, California
    -> if you adapt dummy variables, columns should be increased.
    -> so we can consider of p-value

Prediction
  Profit

'''
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('./data/50_Startups.csv')
X = dataset.drop(['Profit'], axis=1).values
y = dataset['Profit'].values

# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummy variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor)

# Predicting the Test set
y_pred = regressor.predict(X_test)
print(y_pred)





