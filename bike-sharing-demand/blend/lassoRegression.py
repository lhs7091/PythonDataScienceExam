import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from rmsle import rmsle


def getLassoRegression(X, y, kfolds):
    lasso_m_ = Lasso()
    alpha = 1/np.array([0.01, 0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
    lasso_params_ = {'max_iter':[3000], 'alpha':alpha}

    rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
    y_train_log = np.log1p(y)

    grid_lasso_m = GridSearchCV(lasso_m_,
                              lasso_params_,
                              scoring = rmsle_scorer,
                              cv=5)

    #grid_lasso_m.fit(X, y_train_log )
    grid_lasso_m.fit(X, y_train_log)
    preds = grid_lasso_m.predict(X)
    print (grid_lasso_m.best_params_)
    score = rmsle(np.exp(y_train_log),np.exp(preds), False)
    print ("RMSLE Value For Lasso Regression: ", score)

    return score