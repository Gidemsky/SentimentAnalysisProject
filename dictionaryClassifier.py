import pandas as pd
import model_utils as mUtils
import random
import numpy as np
import re


def num_of_words_in_vec(a, ls_b):
    ls_a = a.split(' ')
    while "" in ls_a:
        ls_a.remove("")
    count = 0
    for it in ls_a:
        if it in ls_b:
            count += 1
    return count


def create_tweet_words(features):
    processed_features = []
    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)
    return processed_features


def classify(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def is_correct(x, y):
    if x == 0:
        return 0
    if x == y:
        return 1
    else:
        return -1

fPos = "twitter_samples/negative_tweets1.json"
fNeg = "twitter_samples/positive_tweets1.json"
is_trans = False
positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)

neg_pos_tweets = positive_tweets + negative_tweets
random.shuffle(neg_pos_tweets)

neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

if is_trans:
    pos_wrds_fname = 'vocabulary_words/positive_words_he.txt'
    neg_wrds_fname = 'vocabulary_words/negative_words_he.txt'
else:
    pos_wrds_fname = 'vocabulary_words/positive_words.txt'
    neg_wrds_fname = 'vocabulary_words/negative_words.txt'

with open(pos_wrds_fname) as f:
    pos_words = f.read().split('\n')

with open(neg_wrds_fname) as f1:
    neg_words = f1.read().split('\n')

tweet_text = neg_pos_tweets.iloc[:, 2].values
ids = neg_pos_tweets.iloc[:, 23].values
labels = neg_pos_tweets.iloc[:, 24].values

tweet_words = create_tweet_words(tweet_text)
tweet_wrds_df = pd.DataFrame(data={'id': ids, 'tweet_words': tweet_words, 'label': labels})
tweet_wrds_df['pos_words_count'] = tweet_wrds_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], pos_words),
                                                       axis=1)
tweet_wrds_df['neg_words_count'] = tweet_wrds_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], neg_words),
                                                       axis=1)
tweet_wrds_df['vocab_words_dif'] = tweet_wrds_df.pos_words_count - tweet_wrds_df.neg_words_count
tweet_wrds_df['classifier'] = tweet_wrds_df.vocab_words_dif.apply(classify)
tweet_wrds_df['clas_correct'] = tweet_wrds_df.apply(lambda x: is_correct(x['classifier'], x['label']), axis=1)
clas_series = tweet_wrds_df.pivot_table(index=['clas_correct'], aggfunc='size')
clas_df = pd.DataFrame(data={'is_correct': clas_series.index.values, 'count': clas_series})
clas_df.reset_index(drop=True, inplace=True)
clas_df.loc[clas_df['is_correct'] == 1, 'is_correct'] = 'correct'
clas_df.loc[clas_df['is_correct'] == -1, 'is_correct'] = 'incorrect'
clas_df.loc[clas_df['is_correct'] == 0, 'is_correct'] = 'uncovered'
clas_df.to_csv('vocab_classifier_res.csv')
a = 1
