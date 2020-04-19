import pandas as pd

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from modelHelper import *
import random
import json

plt.show()

'''
data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
print(airline_tweets.head())

airline_features = airline_tweets.iloc[:, 10].values
airline_labels = airline_tweets.iloc[:, 1].values
'''

positive_tweets = []
for line in open("twitter_samples/positive_tweets.json"):
    tweet = json.loads(line)
    tweet["label"] = 1
    positive_tweets.append(tweet)

negative_tweets = []
for line in open("twitter_samples/negative_tweets.json"):
    tweet = json.loads(line)
    tweet["label"] = -1
    negative_tweets.append(tweet)

neg_pos_tweets = positive_tweets + negative_tweets
random.shuffle(neg_pos_tweets)

neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

# labels = airline_tweets.columns.values.tolist().index['label']
labels = neg_pos_tweets.iloc[:, 24].values
# j_features = neg_pos_tweets.iloc[:, neg_pos_tweets.columns != 'label'].values
features = neg_pos_tweets.iloc[:, 2].values
ids = neg_pos_tweets.iloc[:, 23].values

nBModel = modelHelper("random forest", 5, labels, ids)
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = nBModel.create_features(features, vectorizer)
nBModel.train_and_test_model()
accuracy = nBModel.get_accuracy()
print('naive bayes results:')
print(accuracy[0])
print(accuracy[1])
print(accuracy[2])
results = nBModel.pred_vs_ytest_comp()
print(results)
results.to_csv("random_forest_results.csv")

a = 1
