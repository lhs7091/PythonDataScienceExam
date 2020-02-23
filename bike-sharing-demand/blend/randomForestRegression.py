import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rmsle import rmsle
from sklearn.metrics import mean_squared_log_error


def randomForest(X_train, y_train):
    '''rfModel = RandomForestRegressor(n_estimators=100)

    score = np.sqrt(mean_squared_log_error(y, rfModel.fit(X, y).predict(X)))
    print("RMSLE Value for Random Forest: {:.5f}".format(score))'''

    rfModel = RandomForestRegressor(n_estimators=100)

    y_train_log = np.log1p(y_train)
    rfModel.fit(X_train, y_train_log)

    preds = rfModel.predict(X_train)
    score = rmsle(np.exp(y_train_log), np.exp(preds), False)
    print("RMSLE Value for Random Forest: {:.5f}".format(score))

    return score

'''    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(12, 5)

    plt.xticks(rotation=30, ha='right')
    sns.distplot(y_train, ax=ax1, bins=50)

    plt.xticks(rotation=30, ha='right')
    sns.distplot(np.exp(rfModel.predict(X_test)), ax=ax2, bins=50)'''



'''    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(12, 5)
    
    plt.xticks(rotation=30, ha='right')
    sns.distplot(y_train, ax=ax1, bins=50)
    
    plt.xticks(rotation=30, ha='right')
    sns.distplot(np.exp(rfModel.predict(X_test)), ax=ax2, bins=50)'''

