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

        ls_fet = processed_feature.split(' ')
        while "" in ls_fet:
            ls_fet.remove("")

        processed_features.append(ls_fet)

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


def create_df_and_vocab_ls():
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
    ids = neg_pos_tweets.loc[:, 'id_str'].values
    labels = neg_pos_tweets.loc[:, 'label'].values
    tweet_words = create_tweet_words(tweet_text)
    tweet_wrds_df = pd.DataFrame(data={'id': ids, 'tweet_words': tweet_words, 'label': labels})
    return tweet_wrds_df, pos_words, neg_words


def classify_by_vocab(tweets_df, pos_words, neg_words):
    tweets_df['pos_words_count'] = tweets_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], pos_words),
                                                   axis=1)
    tweets_df['neg_words_count'] = tweets_df.apply(lambda x: num_of_words_in_vec(x['tweet_words'], neg_words),
                                                   axis=1)
    tweets_df['vocab_words_dif'] = tweets_df.pos_words_count - tweets_df.neg_words_count
    tweets_df['classifier'] = tweets_df.vocab_words_dif.apply(classify)
    tweets_df['clas_correct'] = tweets_df.apply(lambda x: is_correct(x['classifier'], x['label']), axis=1)
    clas_series = tweets_df.pivot_table(index=['clas_correct'], aggfunc='size')
    clas_df = pd.DataFrame(data={'is_correct': clas_series.index.values, 'count': clas_series})
    clas_df.reset_index(drop=True, inplace=True)
    return tweets_df


def run_classification():
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls()
    clas_df = classify_by_vocab(tweets_df, pos_words, neg_words)
    clas_df.loc[clas_df['is_correct'] == 1, 'is_correct'] = 'correct'
    clas_df.loc[clas_df['is_correct'] == -1, 'is_correct'] = 'incorrect'
    clas_df.loc[clas_df['is_correct'] == 0, 'is_correct'] = 'uncovered'
    clas_df.to_csv('vocab_classifier_res.csv')


def words_in_vec(a, ls_b):
    ls_a = a.split(' ')
    while "" in ls_a:
        ls_a.remove("")
    words_ls = []
    for it in ls_a:
        if it in ls_b:
            words_ls.append(it)
    return words_ls


def add_vocab_columns(tweets_df, pos_words, neg_words):
    tweets_df['pos_words'] = tweets_df.apply(lambda x: words_in_vec(x['tweet_words'], pos_words),
                                             axis=1)
    tweets_df['neg_words'] = tweets_df.apply(lambda x: words_in_vec(x['tweet_words'], neg_words),
                                             axis=1)
    return tweets_df


def create_tweets_w_vocab_df():
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls()
    # tweets_df = add_vocab_columns(tweets_df, pos_words, neg_words)
    pos_dict, neg_dict = has_multi_words(tweets_df, pos_words, neg_words)
    return pos_dict, neg_dict


def has_multi_words(df, pos_words, neg_words):
    pos_dict = {}
    neg_dict = {}
    for p in pos_words:
        df['w'] = p
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True]
        pos_dict[p] = a
    for n in neg_words:
        df['w'] = n
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True]
        pos_dict[n] = a

    for p in pos_dict.keys():
        if pos_dict[p].empty:
            removekey(pos_dict, p)

    return pos_dict, neg_dict

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

if __name__ == "__main__":
    # run_classification()
    df = create_tweets_w_vocab_df()
a = 1
