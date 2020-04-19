import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

pd.set_option('display.max_columns' , None)
pd.set_option('display.max_rows', None)

train = pd.read_csv('./covid19_forcasting_4weeks/train.csv', parse_dates=['Date'])
test = pd.read_csv('./covid19_forcasting_4weeks/test.csv', parse_dates=['Date'])
submission = pd.read_csv('./covid19_forcasting_4weeks/submission.csv')


def fillState(state, country):
    if state == EMPTY_VAL: return country
    return state


train.rename(columns={'Country_Region':'Country'}, inplace=True)
test.rename(columns={'Country_Region':'Country'}, inplace=True)

EMPTY_VAL = "EMPTY_VAL"

train.rename(columns={'Province_State':'State'}, inplace=True)
train['State'].fillna(EMPTY_VAL, inplace=True)
train['State'] = train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

test.rename(columns={'Province_State':'State'}, inplace=True)
test['State'].fillna(EMPTY_VAL, inplace=True)
test['State'] = test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

#print(train.isnull().sum())
#print(test.isnull().sum())

groupByCountry = train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max()\
    .reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

#print(groupByCountry.head())
print(groupByCountry.loc[lambda df: df['Country']=='Korea, South'])

import plotly.express as px

countries = groupByCountry.Country.unique().tolist()
plot = train.loc[(train.Country.isin(countries[:10])) & (train.Date>='2020-03-11'),
                 ['Date', 'Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country', 'State'])\
                 .max().reset_index().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False)\
                 .reset_index()
fig = px.bar(plot, x='Date', y='ConfirmedCases', color='Country', barmode='stack')
fig.update_layout(title='Rise of Confirmed Cases around top 10 countries', annotations=[dict(x='2020-03-21', y=150, xref='x', yref='y', text='Coronas Rise exponentially from here', showarrow=True, arrowhead=1, ax=-150, ay=-150)])
#fig.show()

train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().nlargest(15, 'ConfirmedCases').style.background_gradient(cmap='nipy_spectral')

plot = train.loc[:, ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()
plot.loc[:, 'Date'] = plot.Date.dt.strftime("%Y-%m-%d")
plot.loc[:, 'Size'] = np.power(plot['ConfirmedCases']+1, 0.3)-1#np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*300)
fig = px.scatter_geo(plot,
                     locations='Country',
                     locationmode='country names',
                     hover_name='Country',
                     color='ConfirmedCases',
                     animation_frame='Date',
                     size='Size',
                     #projection='natural earth',
                     title='Rize of COVID-19 Confirmed Cases')
#fig.show()

countries = groupByCountry.Country.unique().tolist()
plot = train.loc[(train.Country.isin(countries[:10])), ['Date', 'Country', 'ConfirmedCases']]\
    .groupby(['Date', 'Country']).max().reset_index()
fig = px.line(plot, x='Date', y='ConfirmedCases', color='Country')
fig.update_layout(title='Number of Confirmed Cases per Day for Top 10 Countries',
                  xaxis_title='Date',
                  yaxis_title='Number of Confirmed Cases')
#fig.show()


MIN_TEST_DATE = test.Date.min()
df_train = train.loc[train.Date<MIN_TEST_DATE, :]
y1_train = train.iloc[:, -2]
y2_train = train.iloc[:, -1]


def extractDate(df, colName='Date'):
    """
    This function does extract the date feature into multiple features
    - week, day, month, year, dayofweek
    :param df:
    :param colName:
    :return df:
    """
    assert colName in df.columns
    df = df.assign(week = df.loc[:, colName].dt.week,
                   month = df.loc[:, colName].dt.month,
                   day = df.loc[:, colName].dt.day,
                   dayofweek = df.loc[:, colName].dt.dayofweek,
                   dayofyear = df.loc[:, colName].dt.dayofyear)
    return df


def createNewDataset(df):
    """
    This function does create a new dataset for modelling
    """

    df_New = df.copy()

    df_New.loc[:, 'Date_Int'] = (df_New.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')
    df_New.drop(columns=['Date'], axis=1, inplace=True)

    return df_New


X_train = createNewDataset(train)
X_test = createNewDataset(test)
#print(X_train[X_train.Country == 'Japan'].tail())
days = range(1,11)


def getDaysShift(df):
    newDCols = []
    for day in days:
        newDCol = f'D_{day}'
        df['C'+newDCol] = df.groupby(['Country', 'State'])['LConfirmedCases'].shift(day)
        df['F'+newDCol] = df.groupby(['Country', 'State'])['LFatalities'].shift(day)
        newDCols.append(newDCol)
    return df


days_change = [1,2,3,5,7,10]


def getChangeGrowth(df):
    newCCols = []
    newGCols = []
    for day in days_change:
        newCCol = f'C_{day}'
        df['C'+newCCol] = df['LConfirmedCases'] - df[f'CD_{day}']
        df['F'+newCCol] = df['LFatalities'] - df[f'FD_{day}']
        newCCols.append(newCCol)
        newGCol = f'G_{day}'
        df['C'+newGCol] = df['C'+newCCol] / df[f'CD_{day}']
        df['F'+newGCol] = df['F'+newCCol] / df[f'FD_{day}']
        newGCols.append(newGCol)

    df.fillna(0, inplace=True)
    return df


windows = [1,2,3,5,7]


def getMA(df):
    newCMACols = []
    newGMACols = []
    for window in windows:
        for day in days_change:
            newCMACol = f'CMA_{day}_{window}'
            df['C'+newCMACol] = df[f'CC_{day}'].rolling(window).mean()
            df['F'+newCMACol] = df[f'FC_{day}'].rolling(window).mean()
            newCMACols.append(newCMACol)
            newGMAcol = f'GMA_{day}_{window}'
            df['C'+newGMAcol] = df[f'CG_{day}'].rolling(window).mean()
            df['F'+newCMACol] = df[f'FG_{day}'].rolling(window).mean()
            newGMACols.append(newGMAcol)
    df.fillna(0, inplace=True)
    return df


cases = [1,50,100,500,1000,5000,35000,75000,100000]


def getCDSC(df):
    newCDSCCols = []
    for case in cases:
        newDSCCol = f'{case}_CDSC'
        df.loc[df.CD_1 == 0, newDSCCol] = 0
        df.loc[df.CD_1 >= case, newDSCCol] = df[df.CD_1 >= case].groupby(['Country', 'State']).cumcount()
        newCDSCCols.append(newDSCCol)
    df.fillna(0, inplace=True)
    return df


deaths = [1,50,100,500,1000,5000,35000]


def getFDSC(df):
    newFDSCCols = []
    for death in deaths:
        newDSCCol = f'{death}_FDSC'
        df.loc[df.FD_1 == 0, newDSCCol] = 0
        df.loc[df.FD_1 >= death, newDSCCol] = df[df.FD_1 >= death].groupby(['Country', 'State']).cumcount()
        newFDSCCols.append(newDSCCol)
    df.fillna(0, inplace=True)
    return df


df = pd.concat([train, test[test.Date > train.Date.max()]], axis=0, sort=False, ignore_index=True)
df['LConfirmedCases'] = np.log1p(df['ConfirmedCases'])
df['LFatalities'] = np.log1p(df['Fatalities'])
df.loc[(df.Date >= test.Date.min()) &
       (df.Date <= train.Date.max()), 'ForecastId'] = test.loc[(test.Date >= test.Date.min()) & (test.Date <= train.Date.max()), 'ForecastId'].values
df = getDaysShift(df)
df = getChangeGrowth(df)
df = getMA(df)
df = getCDSC(df)
df = getFDSC(df)

df['CSID'] = df.groupby(['Country', 'State']).cumcount()
df.loc[df.ForecastId > 0, 'ForecastId'].nunique()

disp = df[df.Country == 'Japan'][['ConfirmedCases', 'CD_1', 'CC_1']][70:95]
print(disp)

from sklearn.preprocessing import LabelEncoder
cLEncoder = LabelEncoder()
sLEncoder = LabelEncoder()

df.loc[:, 'Country'] = cLEncoder.fit_transform(df.loc[:, 'Country'])
df.loc[:, 'State'] = cLEncoder.fit_transform(df.loc[:, 'State'])

X_Train = df[df.Date <= train.Date.max()]
X_Train.loc[:, 'Date_Int'] = (X_Train.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')

yC_Train = X_Train.CC_1
yF_Train = X_Train.FC_1
X_Train = X_Train.drop(columns=['Id', 'ForecastId', 'Date', 'ConfirmedCases', 'Fatalities', 'CC_1', 'FC_1'])
print(X_Train.shape, yC_Train.shape, yF_Train.shape)
print(X_Train.tail())


