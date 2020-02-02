'''
Analyze events chitchats
    what is the key words interested
    how to get the prequency each key word
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
'''
# setting fond on mac
plt.rc("font", family="AppleGothic")
# setting fond on window
# plt.rc("font", family="Malgun Gothic")
plt.rc('axes', unicode_minus=False)

# load csv file
df = pd.read_csv("/Users/lhs/PycharmProjects/PythonDataScienceExam/event_comments_analyes/events.csv")

# remove duplicate chitchats caused by network error or else.
# print(df.shape) # (2436, 1)
df = df.drop_duplicates(["text"], keep="last") # remain only last one
# print(df.shape) # (2399, 1)

# save the original text at first
df["origin_text"] = df["text"]

# every word change lower case for data filtering
df["text"] = df["text"].str.lower()

# make words or languages by one
df["text"] = df["text"].str.replace(
    "python", "파이썬").str.replace( # python -> 파이썬
    "pandas", "판다스").str.replace(
    "javascript", "자바스크립트").str.replace(
    "java", "자바").str.replace(
    "react", "리액트"
)

# add class column from text column
df["course"] = df["text"].apply(lambda x: x.split("관심강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강의")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심강좌")[-1])
df["course"] = df["course"].apply(lambda x: x.split("관심 강좌")[-1])
df["course"] = df["course"].str.replace(":", "")

#print(df[["text", "course"]].tail(10))

# extract key word from "text"
search_keyword=['머신러닝', '딥러닝', '파이썬', '데이터분석', '크롤링', '공공데이터']

for keyword in search_keyword:
    df[keyword] = df["course"].str.contains(keyword)
#print(df.head())
df_python = df[df["text"].str.contains("파이썬|공공데이터|판다스")].copy()
# print(df.shape) # (2399, 9)
# print(df_python.shape) # (429, 9)

#print(df[search_keyword].sum().sort_values(ascending=False))
'''파이썬      405
머신러닝     132
크롤링       56
딥러닝       52
데이터분석     24
공공데이터     12'''

# i want to read specific data(공공데이터) in text column
'''
text = df.loc[(df["공공데이터"] == True), "text"]
for i in text:
    print(i)

#관심강의: 프로그래밍 시작하기 : 파이썬 입문, 공공데이터로 파이썬 데이터 분석 시작하기
파이썬의 고수가 되고싶어요
자바기반 웹 개발자입니다. 데이터 분석에 많이 쓰이는 파이썬이 궁금합니다.
#관심강의: 프로그래밍 시작하기 : 파이썬 입문, 공공데이터로 파이썬 데이터 분석 시작하기
올해 안에 원하는 공부 다 끝내보려고요. 내년이면 수능이라..    
'''


"""
put key words on token inside of the BOW back
sklearn : separate key words based on each spaces
"""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = 'word', # vectorize based on character
                             tokenizer = None, # call of specific tokenizer
                             preprocessor = None, # implement of preprocessor
                             stop_words = None, # no permission word
                             min_df = 2, # the number of documents at least with token(separate mistyping or special words)
                             ngram_range=(3, 6), # range of BOW ['aaa','bbb','ccc'] ~ ['aaa','bbb','ccc','aaa','bbb','ccc']
                             max_features = 2000 # the number of words to make
                            )
feature_vector = vectorizer.fit_transform(df['course'])
# print(feature_vector.shape) # (2399, 2000) 2000 features

vocab = vectorizer.get_feature_names()
# print(len(vocab)) # 2000
# print(vocab[:10]) # ['12개 만들면서 배우는', '12개 만들면서 배우는 ios', '12개 만들면서 배우는 ios 아이폰', .....]

# check the word frequency in each review
pd.DataFrame(feature_vector[:10].toarray(), columns=vocab).head()

# but to deal with is difficult because it is (2399, 2000) metrics
# so all of data in feature_vector sum group by vocab
dist = np.sum(feature_vector, axis=0)
df_freq = pd.DataFrame(dist, columns=vocab)
# print(df_freq)

# change the row and axis by ascending
df_freq.T.sort_values(by=0, ascending=False)
# print(df_freq.T.sort_values(by=0, ascending=False).head(20))

'''
TF-IDF 로 가중치를 주어 벡터화
TfidfTransformer()
norm='l2' 각 문서의 피처 벡터를 어떻게 벡터 정규화 할지 정합니다.
L2 : 벡터의 각 원소의 제곱의 합이 1이 되도록 만드는 것이고 기본 값(유클리디안거리)
L1 : 벡터의 각 원소의 절댓값의 합이 1이 되도록 크기를 조절(맨하탄거리)
smooth_idf=False
피처를 만들 때 0으로 나오는 항목에 대해 작은 값을 더해서(스무딩을 해서) 피처를 만들지 아니면 그냥 생성할지를 결정
sublinear_tf=False
use_idf=True
TF-IDF를 사용해 피처를 만들 것인지 아니면 단어 빈도 자체를 사용할 것인지 여부
'''
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
# print(transformer) # TfidfTransformer(norm='l2', smooth_idf=False, sublinear_tf=False, use_idf=True)
feature_tfidf = transformer.fit_transform(feature_vector)
'''
%%time 
feature_tfidf.shape
'''
tfidf_freq = pd.DataFrame(feature_tfidf.toarray(), columns=vocab)

df_tfidf = pd.DataFrame(tfidf_freq.sum())
df_tfidf_top = df_tfidf.sort_values(by=0, ascending=False)
# print(df_tfidf_top.head())

'''
Clustering
    - K-Means
    - MiniBatchKMeans
https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
'''
from sklearn.cluster import KMeans
from tqdm import trange

'''
# check how many clusters we need.
inertia = []

start = 10
end = 50


for i in trange(start, end):
    cls = KMeans(n_clusters=i, random_state=42)
    cls.fit(feature_vector)
    inertia.append(cls.inertia_)
plt.plot(range(start,end), inertia)
plt.title("KMeans clustering")
plt.show()
'''
n_clusters = 60
cls = KMeans(n_clusters=n_clusters, random_state=42)
cls.fit(feature_vector)
predict = cls.predict(feature_vector)
df["cluster"] = predict
'''
print(df["cluster"].value_counts().head(10))
0     1672
33      85
22      44
31      41
44      34
5       33
30      30
14      29
1       29
11      28
Name: cluster, dtype: int64
'''
# clustering by MiniBatchKMeans
from sklearn.cluster import MiniBatchKMeans
'''
b_inertia = []
# check how many clusters we need.
start = 10
end = 50
for i in trange(start, end):
    cls = MiniBatchKMeans(n_clusters=i, random_state=42)
    cls.fit(feature_vector)
    b_inertia.append(cls.inertia_)
plt.plot(range(start,end), b_inertia)
plt.title("KMeans clustering")
plt.show()
'''
cls = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
cls.fit(feature_vector)
predict = cls.predict(feature_vector)
df["bcluster"] = predict
# print(df["bcluster"].value_counts())

# pre-check
# print(df[df["bcluster"] == 15].head(5))
# print(df.loc[df["bcluster"] == 15, ["bcluster", "cluster", "course"]].head(10))

from wordcloud import WordCloud

# we want to except specific words
stopwords = ["관심 강의", "관심강의", "관심", "강의", "강좌", "강의를",
             "올해", "올해는", "열심히", "공부를", "합니다", "하고", "싶어요",
             "있는", "있습니다", "싶습니다", "2020년"]


def displayWordCloud(data=None, backgroundcolor='white', width=1280, height=768):
    wordcloud = WordCloud(
        font_path = '/Library/Fonts/NanumGothicExtraBold.otf',
        stopwords = stopwords,
        background_color = backgroundcolor,
        width = width, height = height).generate(data)

    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


course_text = " ".join(df["course"])
displayWordCloud(course_text)
