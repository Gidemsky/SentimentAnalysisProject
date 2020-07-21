import pandas as pd

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import model_utils as mUtils
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

fPos = "twitter_samples/negative_tweets1.json"
fNeg = "twitter_samples/positive_tweets1.json"
is_trans = False
positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)

neg_pos_tweets = positive_tweets + negative_tweets
random.shuffle(neg_pos_tweets)

neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

# labels = airline_tweets.columns.values.tolist().index['label']
labels = neg_pos_tweets.iloc[:, 24].values
# j_features = neg_pos_tweets.iloc[:, neg_pos_tweets.columns != 'label'].values
tweets = neg_pos_tweets.iloc[:, 2].values
ids = neg_pos_tweets.iloc[:, 23].values
#vocabulary = mUtils.get_vocabulary()

nBModel = modelHelper("naive bayes", 5, labels, ids)
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_tweets = nBModel.process_tweets(tweets)
nBModel.create_features(processed_tweets, vectorizer)
nBModel.train_and_test_model()
accuracy = nBModel.get_accuracy()

print('Naive Bayes results:')
print("Cross Validation Accuracy: ", accuracy)
print("Accuracy: ", nBModel.accuracy_score)
results = nBModel.get_predictions()
df = pd.DataFrame(data={"id": results[0], "prediction": results[1], "label": results[2]})
df['correct_prediction'] = df.prediction == df.label
'''
print(accuracy[0])
print(accuracy[1])
print(accuracy[2])
results = nBModel.pred_vs_ytest_comp()
print(results)
'''

mUtils.save_results(df, "naive_bayes", is_trans)

a = 1
