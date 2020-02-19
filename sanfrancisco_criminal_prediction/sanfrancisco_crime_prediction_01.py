import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

train = pd.read_csv('./train.csv', parse_dates=['Dates'])
test = pd.read_csv('./test.csv', parse_dates=['Dates'])
full_data = [train, test]

'''
Quick Preprocessing & Feature Engineering
'''

# Drop the Resolution Column:
train.drop('Resolution', axis=1, inplace=True)

'''
Dummy Encoding of 'PdDistrict':¶
pd.get_dummies() :: One-Hot Encoding
'''
train = pd.get_dummies(train, columns=['PdDistrict'])
test = pd.get_dummies(test, columns=['PdDistrict'])

# Engineer a feature to indicate
# whether the crime was commited by day or by night
train['IsDay'] = 0
train.loc[(train['Dates'].dt.day>=6) & (train['Dates'].dt.hour <20), 'IsDay'] = 1

test['IsDay'] = 0
test.loc[(test['Dates'].dt.hour>=6) & (test['Dates'].dt.hour <20), 'IsDay'] = 1


days_to_int_dic = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7,
}

# Encode 'DayOfWeek' to Integer
train['DayOfWeek'] = train['DayOfWeek'].map(days_to_int_dic)
test['DayOfWeek'] = test['DayOfWeek'].map(days_to_int_dic)

# Create hour, month and year columns
train['Hour'] = train['Dates'].dt.hour
test['Hour'] = test['Dates'].dt.hour
train['Month'] = train['Dates'].dt.month
test['Month'] = test['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
test['Year'] = test['Dates'].dt.year

# Deal with the cyclic characteristic of Months and Days of Week:
train['HourCos'] = np.cos((train['Hour'] * 2 * np.pi) / 24)
train['DayOfWeekCos'] = np.cos((train['DayOfWeek'] * 2 * np.pi) / 7)
train['MonthCos'] = np.cos((train['Month'] * 2 * np.pi) / 12)
test['HourCos'] = np.cos((test['Hour'] * 2 * np.pi) / 24)
test['DayOfWeekCos'] = np.cos((test['DayOfWeek'] * 2 * np.pi) / 7)
test['MonthCos'] = np.cos((test['Month'] * 2 * np.pi) / 12)


'''
Label Encoding of 'Category':
change type categorical data to numeric data
'''
category_label = LabelEncoder()
train['CategoryInt'] = pd.Series(category_label.fit_transform(train['Category']))

# separate Section, block
train['InIntersection'] = 1
train.loc[train['Address'].str.contains('Block'), 'InIntersection'] = 0

test['InIntersection'] = 1
test.loc[test['Address'].str.contains('Block'), 'InIntersection'] = 0


'''
Feature Selection
Now let's get our dataset ready for training !
'''

feature_cols = ['X', 'Y', 'IsDay', 'DayOfWeek', 'Month', 'Hour', 'Year', 'InIntersection',
                'PdDistrict_BAYVIEW', 'PdDistrict_CENTRAL', 'PdDistrict_INGLESIDE',
                'PdDistrict_MISSION', 'PdDistrict_NORTHERN', 'PdDistrict_PARK',
                'PdDistrict_RICHMOND', 'PdDistrict_SOUTHERN', 'PdDistrict_TARAVAL', 'PdDistrict_TENDERLOIN']
target_cols = 'CategoryInt'

train_x, val_x, train_y, val_y = train_test_split(train[feature_cols], train[target_cols], test_size=0.5, shuffle=True)
# train_x = train[feature_cols]
# train_y = train[target_cols]

test_id = test['Id']
test_x = test[feature_cols]

'''
Modeling
1. Validation set
    - 90% of training data are learned(trained) and 
    - Vailidate by the others 10% training data
2. k-fold cross validation
    - define round for training.
    - devide trainig data by round
    - 
'''
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

'''
지도 학습 모델
(1) k-최근접 이웃 모델
KNN(K-Nearest Neighbors)알고리즘은 앞에서도 잠깐 설명하였습니다. 
이는 가장 간단한 머신러닝 알고리즘으로 학습 데이터셋을 그냥 저장하는 것이 모델을 만드는 과정의 전부입니다. 
새로운 데이터를 예측할 때는 훈련 데이터셋에서 가장 가까운 데이터 포인트인 '최근접 이웃'을 찾습니다.

'''

'''
# 21.98
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold, n_jobs=1, scoring=scoring)
print(round(np.mean(score)*100, 2)) 
'''

'''
# accuracy : 0.33070933398933317
clf.fit(train_x, train_y)
acc = clf.score(train_x, train_y, cross_val_score(cv=k_fold, n_jobs=1))
print('accuracy : {}'.format(acc))
'''

'''
Random Forest
    - small diverse decision trees voting for one decision
    - boosting : biased data collection 
      -> decision tree is easy to over fitting 
    - random selection of feature set
    - aggregating of result(voting)

'''
'''
from sklearn.ensemble import RandomForestClassifier

# Score :
clf = RandomForestClassifier(n_estimators=10)
# estimators = 1, Score = 0.21520
# estimators = 10, Score = 0.28487
# estimators = 50, Score = 0.30399
# estimators = 100, Score = 0.30747
scoring = 'accuracy'
score = cross_val_score(clf, train_x, train_y, cv=k_fold,
                        n_jobs=1, scoring='accuracy')
score = score.mean()
# 0에 근접할수록 좋은 데이터
print("Score = {0:.5f}".format(score))


clf.fit(train_x, train_y)
result = pd.DataFrame(clf.predict_proba(test_x), index=test_x.index, columns=clf.classes_)
result.to_csv('./submit_randomforest.csv')
print(result)
'''



param = {
    'max_depth':[2,3,4],
    'n_estimators':range(550,700,50),
    'colsample_bytree':[0.5,0.7,1],
    'colsample_bylevel':[0.5,0.7,1],
}
model = xgb.XGBRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)

grid_search.fit(train_x, train_y)
print(grid_search.best_params_)
print(grid_search.best_estimator_)

pred_train = grid_search.predict(train_x)
pred_val = grid_search.predict(val_x)

print('train mae score: ', metrics.log_loss(train_y, pred_train))
print('val mae score:', metrics.log_loss(val_y, pred_val))


train_xgb = xgb.DMatrix(train_x, label=train_y)
test_xgb  = xgb.DMatrix(test_x)

print('Fitting Model ...')
m = xgb.train(param, train_xgb, 10)
res = m.predict(test_xgb)
cols = ['Id'] + category_label.classes_
submission = pd.DataFrame(res, columns=category_label.classes_)
submission.insert(0, 'Id', test_id)
submission.to_csv('submission.csv', index=False)
print('Done Outputing !')
print(submission.sample(3))