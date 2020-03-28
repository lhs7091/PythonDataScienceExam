"""
Non-Linear SVR
C : 규제강도
gamma :
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 100
X = 2*np.random.rand(m, 1)-1
y = (0.2+0.1*X+0.5*X**2+np.random.randn(m,1)/10).ravel()

from sklearn.svm import SVR
svm_poly_svr = SVR(kernel='poly', gamma='auto', degree=2, C=10, epsilon=0.1)
svm_poly_svr.fit(X,y)
print(svm_poly_svr)

svm_poly_svr2 = SVR(kernel='poly', gamma='auto', degree=2, C=100, epsilon=0.1)
svm_poly_svr3 = SVR(kernel='poly', gamma='auto', degree=2, C=0.01, epsilon=0.1)
svm_poly_svr2.fit(X, y)
svm_poly_svr3.fit(X, y)


import svrFunction as srt
plt.figure(figsize=(9,4))
plt.subplot(121)
srt.plot_svm_regression(svm_poly_svr2, X, y, [-1,1,0,1])
plt.title('degree={}, C={}, Ε={}'.format(svm_poly_svr2.degree, svm_poly_svr2.C, svm_poly_svr2.epsilon), fontsize=18)
plt.subplot(122)
srt.plot_svm_regression(svm_poly_svr3, X, y, [-1, 1, 0, 1])
plt.title('degree={}, C={}, Ε={}'.format(svm_poly_svr3.degree, svm_poly_svr3.C, svm_poly_svr3.epsilon), fontsize=18)
plt.show()






