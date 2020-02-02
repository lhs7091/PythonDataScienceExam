'''
crawling many comments
Analyze many comments on web site
https://www.inflearn.com/pages/newyear-event-20200102
1. please check robots.txt (ex. www.inflearn.com/robots.txt)
    User-agent: *
    Disallow: /carts
    Disallow: /orders
    Disallow: /course/*/edit
    Disallow: /api
    Disallow: /assets
    Disallow: /auth
    Disallow: /oauth2_email_not_validated
    Disallow: /email_validation
    Disallow: /signout
    Disallow: /hello
    Disallow: /password
    Disallow: /email
    Disallow: /admin
    Sitemap: https://www.inflearn.com/sitemap.xml
'''
# load the contents of web site into small web browser
import requests
# parsing of web site
from bs4 import BeautifulSoup as bs
# data frame for results of crawling
import pandas as pd
# status and check of crawling progress
from tqdm import trange

base_url = 'https://www.inflearn.com/pages/newyear-event-20200102'
response = requests.get(base_url) # -> <Response [200]>
# print(response.text)
soup = bs(response.text, 'html.parser')
# main > section > div > div > div.chitchats > div.chitchat-list > div:nth-child(835) > div > div.body.edit-chitchat
content = soup.select("#main > section > div > div > div.chitchats > div.chitchat-list > div")
# print(content[-1])

# load only one chitchat
chitchat = content[-1].select('div.body.edit-chitchat')[0].get_text(strip=True)
# print(chitchat)

# but we need all of chitchat
# so we just test loading 5 chitchats first.
'''for i in range(5):
    print('-'*20)
    chitchat = content[i].select('div.body.edit-chitchat')[0].get_text(strip=True)
    print(chitchat)'''

# get all of chitchat
# print(len(content)) # -> 2436 chitchats
content_count = len(content)
events = []
for i in trange(content_count): # trange : display progress
    chitchat = content[i].select('div.body.edit-chitchat')[0].get_text(strip=True)
    events.append(chitchat) # 100%|██████████| 2436/2436 [00:00<00:00, 9167.09it/s]

# make DataFrame by pandas
df = pd.DataFrame({"text": events})
# save by csv file
df.to_csv("/Users/lhs/PycharmProjects/PythonDataScienceExam/event_comments_analyes/events.csv", index=False)
# load csv file
print(pd.read_csv("/Users/lhs/PycharmProjects/PythonDataScienceExam/event_comments_analyes/events.csv").head())