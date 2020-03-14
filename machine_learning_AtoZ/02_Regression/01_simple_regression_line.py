import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/Salary_Data.csv')

# importing the dataset
X = dataset.drop(['Salary'], axis=1).values
y = dataset['Salary'].values

# splitting the dataset into the Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#plt.scatter(dataset['YearsExperience'], dataset['Salary'])
#plt.show()

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

'''# Visualising the Training set results
plt.scatter(X_train, y_train, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()'''

# Visualising the Test set results
plt.scatter(X_test, y_test, color='r')
plt.plot(X_test, y_pred, color='b')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

