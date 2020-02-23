import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rmsle import rmsle


def randomForest(X_train, y_train):
    rfModel = RandomForestRegressor(n_estimators=100)

    y_train_log = np.log1p(y_train)
    rfModel.fit(X_train, y_train_log)

    preds = rfModel.predict(X_train)
    score = rmsle(np.exp(y_train_log), np.exp(preds), False)
    print("RMSLE Value for Random Forest: {:.5f}".format(score))

    return score
