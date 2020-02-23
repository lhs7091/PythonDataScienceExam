#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False


# In[4]:


train = pd.read_csv('./train.csv', parse_dates=["datetime"])
train.shape


# In[5]:


test = pd.read_csv('./test.csv', parse_dates=["datetime"])
test.shape


# ## Feature Engineering

# In[6]:


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek
train.shape


# In[7]:


test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek
test.shape


# In[13]:


fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count', title='train windspeed')
sns.countplot(data=train, x='windspeed', ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count', title='test windspeed')
sns.countplot(data=test, x='windspeed', ax=axes[1])


# In[14]:


# many data of windspeed input 0, we guess it's valueless
# so we can input windspeed values by estimating machine learning instead of 0


# In[15]:


trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed']!=0]
trainWind0.shape, trainWindNot0.shape


# In[18]:


from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed']!=0]
    
    wCol = ['season', 'weather', 'humidity', 'month', 'temp', 'year', 'atemp']
    
    dataWindNot0['windspeed'] = dataWindNot0['windspeed'].astype('str')
    
    rfModel_wind = RandomForestClassifier()
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0['windspeed'])
    
    wind0Values = rfModel_wind.predict(X=dataWind0[wCol])
    
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0
    
    predictWind0['windspeed'] = wind0Values
    
    data = predictWindNot0.append(predictWind0)
    
    data['windspeed'] = data['windspeed'].astype('float')
    
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data


# In[19]:


train = predict_windspeed(train)

fig, ax1 = plt.subplots()
fig.set_size_inches(18, 6)

plt.sca(ax1)
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='count', title='Train windspeed')
sns.countplot(data=train, x='windspeed', ax=ax1)


# ## Feature Selection

# In[22]:


categorical_feature_names = ['season', 'holiday', 'workingday', 'weather', 'dayofweek', 'month', 'year', 'hour']

for var in categorical_feature_names:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')


# In[23]:


feature_names = ['season', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
                'year', 'hour', 'dayofweek', 'holiday', 'workingday']
feature_names


# In[24]:


X_train = train[feature_names]

print(X_train.shape)
X_train.head()


# In[25]:


X_train.info()


# In[26]:


X_test = test[feature_names]

print(X_test.shape)
X_testlabel_name = "count"
y_train = train[label_name]

print(y_train.shape)
y_train.head().head()


# In[33]:


label_name = "count"
y_train = train[label_name]

print(y_train.shape)
y_train.head()


# In[27]:


X_test.info()


# ## Score
# 
# ### RMSLE

# In[29]:


from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values):
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    difference = np.log(predicted_values+1)-np.log(actual_values+1)
    difference = np.square(difference)
    score = np.sqrt(difference.mean())
    
    return score
rmsle_scorer = make_scorer(rmsle)
rmsle_scorer


# In[31]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ## RandomForest

# In[35]:


from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=1000,
                             n_jobs=1,
                             random_state=0)
model


# In[36]:


get_ipython().run_line_magic('time', 'score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)')
score = score.mean()
print("Score={0:.5f}".format(score))


# ## Train

# In[37]:


model.fit(X_train, y_train)


# In[43]:


predictions = model.predict(X_test)

print(predictions.shape)
predictions[0:10]


# In[46]:


fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12,5)

sns.distplot(y_train, ax=ax1, bins=50)
ax1.set(title="train")
sns.distplot(predictions, ax=ax2, bins=50)
ax2.set(title="test")


# ## Submit

# In[47]:


submission = pd.read_csv('./sampleSubmission.csv')
submission

submission['count'] = predictions

submission.head()
                         


# In[48]:


submission.to_csv('./Score_{0:.5f}_submission.csv'.format(score), index=False)


# In[ ]:




