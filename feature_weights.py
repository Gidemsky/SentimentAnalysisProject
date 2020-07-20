import json
import random
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from modelHelper import *


fPos = "twitter_samples/negative_tweets1.json"
fNeg = "twitter_samples/positive_tweets1.json"
is_trans = True
positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)

neg_pos_tweets = positive_tweets + negative_tweets
random.shuffle(neg_pos_tweets)

neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

labels = neg_pos_tweets.iloc[:, 24].values

features = neg_pos_tweets.iloc[:, 2].values
ids = neg_pos_tweets.iloc[:, 23].values


nBModel = modelHelper("svm", 5, labels, ids)
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = nBModel.create_features(features, vectorizer)
nBModel.scores = (nBModel.fet_ids, cross_val_score(nBModel.model, nBModel.processed_features, nBModel.labels, cv=5))
nBModel.model.fit(nBModel.processed_features, nBModel.labels)
fet_Weights = nBModel.model.coef_[0]
featureNames = vectorizer.get_feature_names()
names_and_weight = pd.DataFrame(data={'feature': featureNames, 'weight': fet_Weights})

a = 1