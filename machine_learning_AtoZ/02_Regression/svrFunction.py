import numpy as np
import matplotlib.pyplot as plt


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


def find_support_vectors(svm_lsvr, X, y):
    y_pred = svm_lsvr.predict(X)
    off_margin = (np.abs(y-y_pred)>=svm_lsvr.epsilon)
    return np.argwhere(off_margin)

