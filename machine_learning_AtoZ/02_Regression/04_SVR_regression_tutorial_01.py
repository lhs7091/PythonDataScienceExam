# Linear SVR
import numpy as np
import matplotlib.pyplot as plt

# make a dataset following Gaussian distribution
# np.random.rnadn -> standard normal distribution
np.random.seed(42)
m = 50
X = 2* np.random.rand(m, 1)
y = (4+3*X+np.random.randn(m, 1)).ravel()

'''print('X{} : {}'.format(str(X.shape), X))
print('y{} : {}'.format(str(y.shape), y))'''

'''plt.scatter(X, y, color='r')
plt.show()'''

from sklearn.svm import LinearSVR
svm_lsvr = LinearSVR(epsilon=1.5, random_state=42)
svm_lsvr.fit(X, y)
print(svm_lsvr)

"""
check the differences in each case
1. basic model
2. large margin(epsilon=1.5)
3. small margin(epsilon=0.5)
"""
svm_lsvr2 = LinearSVR(epsilon=1.5, random_state=42)
svm_lsvr3 = LinearSVR(epsilon=0.5, random_state=42)
svm_lsvr2.fit(X, y)
svm_lsvr3.fit(X, y)

"""
Define support vector
off_margin : difference of Absolute value between real y value and predict value
np.argwhere return True value which is in matrix
"""


def find_support_vectors(svm_lsvr, X, y):
    y_pred = svm_lsvr.predict(X)
    off_margin = (np.abs(y-y_pred)>=svm_lsvr.epsilon)
    return np.argwhere(off_margin)


svm_lsvr2.support_ = find_support_vectors(svm_lsvr2, X, y)
svm_lsvr3.support_ = find_support_vectors(svm_lsvr3, X, y)

eps_x1 = 1
eps_y_pred = svm_lsvr2.predict([[eps_x1]])


"""
display plot with support vector
"""


def plot_svm_regression(svm_lsvr, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_lsvr.predict(x1s)
    plt.plot(x1s, y_pred, 'k-', linewidth=2, label='y^')
    plt.plot(x1s, y_pred+svm_lsvr.epsilon, 'k--')
    plt.plot(x1s, y_pred-svm_lsvr.epsilon, 'k--')
    plt.scatter(X[svm_lsvr.support_], y[svm_lsvr.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, 'bo')
    plt.xlabel('x1', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.axis(axes)


plt.figure(figsize=(9,4))
plt.subplot(121)
plot_svm_regression(svm_lsvr2, X, y, [0,2,3,11])
plt.title('€={}'.format(svm_lsvr2.epsilon), fontsize=18)
plt.ylabel('y', fontsize=18, rotation=0)

'''plt.annotate('', xy=(eps_x1, eps_y_pred), xycoords='data',
             xytext=(eps_x1, eps_y_pred, svm_lsvr2.epsilon),
             textcoords='data', arrowprops={'arrowstyle':'<->', 'linewidth':1.5})'''

plt.text(0.91, 5.6, '€', fontsize=20)
plt.subplot(122)
plot_svm_regression(svm_lsvr3, X, y, [0,2,3,11])
plt.title('€={}'.format(svm_lsvr3.epsilon), fontsize=18)
plt.show()

# Conclusionly, this model is insensitive by epsilon value

