import pandas as pd

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from modelHelper import *

plt.show()

data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
print(airline_tweets.head())

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
nBModel = modelHelper("random forest", 5, labels)
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = nBModel.create_features(features, vectorizer)
nBModel.train_and_test_model()
accuracy = nBModel.get_accuracy()
print('naive bayes results:')
print(accuracy[0])
print(accuracy[1])
print(accuracy[2])

a = 1
