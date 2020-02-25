# Data Preprocessing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing the dataset
dataset = pd.read_csv('./data/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print(y)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)








