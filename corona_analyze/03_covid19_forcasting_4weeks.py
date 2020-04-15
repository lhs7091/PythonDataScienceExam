import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
fig.show()

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
fig.show()







