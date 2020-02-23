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



    '''
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    
    df = pd.DataFrame(grid_lasso_m.cv_results_)
    df["alpha"] = df["params"].apply(lambda x:x["alpha"])
    df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)
    
    plt.xticks(rotation=30, ha='right')
    sns.pointplot(data=df, x='alpha', y='rmsle', ax=ax)'''




"""
def getLassoRegression(X, y, kfolds):
    alphas = 1 / np.array([0.01, 0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
    #lasso_params_ = {'max_iter': [3000], 'alpha': alphas}
    y = np.log1p(y)

    train_scores = []
    test_scores = []

    for alpha in alphas:
        model = Lasso(alpha=alpha)
        #train_score = -mean_squared_error(y, model.fit(X, y).predict(X))
        train_score = np.sqrt(mean_squared_log_error(y, model.fit(X, y).predict(X)))
        test_score = np.mean(cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))
        train_scores.append(train_score)
        test_scores.append(test_score)

    optimal_alpha = alphas[np.argmax(test_scores)]
    optimal_score = np.max(sum(test_scores)/len(test_scores))
    #print("최적 alpha:", optimal_alpha)
    #print("최적 score:", optimal_score)

    lasso = make_pipeline(RobustScaler(),
                          LassoCV(max_iter=1e7, alphas=alphas,
                                  random_state=42, cv=kfolds))

    print(datetime.now(), 'lasso')
    lasso_model_full_data = lasso.fit(X, y)

    return optimal_alpha, optimal_score, lasso_model_full_data



'''    plt.plot(alphas, test_scores, "-", label="검증 성능")
    plt.plot(alphas, train_scores, "--", label="학습 성능")
    plt.axhline(optimal_score, linestyle=':')
    plt.axvline(optimal_alpha, linestyle=':')
    plt.scatter(optimal_alpha, optimal_score)
    plt.title("최적 정규화")
    plt.ylabel('성능')
    plt.xlabel('정규화 가중치')
    plt.legend()
    plt.show()'''

"""