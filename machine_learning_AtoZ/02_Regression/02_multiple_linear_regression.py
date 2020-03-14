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

# Building the optimal model using backward Elimination
# Statsmodels which we will use to compute the values
# and evaluate the statistical significance(통계적 유의성)

'''
1. Select a significance level to stay in the model.(e.g. SL = 0.05)
2. Filt the full model with all possible predictors
3. Consider the predictor with the highest P-value. If P > SL, go to Step4, otherwise go to FIN
4. Remove the predictor
5. Fit model without this variable*
'''
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())




