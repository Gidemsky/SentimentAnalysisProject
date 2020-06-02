import random

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import model_utils as mUtils


def set_data(nBModel):
    plt.show()

    fPos = "twitter_samples/negative_tweets1.json"
    fNeg = "twitter_samples/positive_tweets1.json"

    positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)

    neg_pos_tweets = positive_tweets + negative_tweets
    random.shuffle(neg_pos_tweets)

    neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

    labels = neg_pos_tweets.iloc[:, 24].values
    features = neg_pos_tweets.iloc[:, 2].values
    ids = neg_pos_tweets.iloc[:, 23].values

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    filtered_data = nBModel.filter_data(features, vectorizer)

    return nBModel.get_train_and_test_sets(ids, filtered_data, labels)


def run_model(isTest, nBModel, modelName, fet_ids, train_X, train_Y, test_ids ,test_X, test_Y):
    nBModel.create_model(modelName)
    nBModel.train_model(modelName, fet_ids, train_X, train_Y)
    is_trans = True
    results = nBModel.test_model(modelName, test_ids, test_X)
    accuracy = nBModel.get_accuracy()

    print(modelName + 'Results:')
    print(" Cross Validation Accuracy: ", accuracy[1])

    if isTest:
        print(" Accuracy: ", nBModel.get_test_accuracy(results[1], test_Y))

        df = pd.DataFrame(data={"id": results[0], "prediction": results[1], "label": test_Y})
        df['correct_prediction'] = df.prediction == df.label

        mUtils.save_results(df, modelName, is_trans)


