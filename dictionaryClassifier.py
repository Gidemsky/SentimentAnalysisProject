import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import model_utils as mUtils
import random
import numpy as np
import re

from datetime import datetime

datetime_object = datetime.now()
dt = datetime_object.strftime("%d_%m_%H_%M")


def num_of_words_in_vec(a, ls_b):
    count = 0
    for it in a:
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
    fPos = "twitter_samples/positive_tweets1.json"
    fNeg = "twitter_samples/negative_tweets1.json"
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
    # tweets_df['clas_correct'] = tweets_df.apply(lambda x: is_correct(x['classifier'], x['label']), axis=1)
    tweets_df['clas_correct'] = (tweets_df.label == tweets_df.classifier).astype(int)
    tweets_df['clas_correct'].loc[tweets_df['clas_correct'] == 0] = -1
    tweets_df['clas_correct'].loc[tweets_df['classifier'] == 0] = 0

    return tweets_df


def run_classification(fout_name):
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls()
    res_df = classify_by_vocab(tweets_df, pos_words, neg_words)
    clas_series = res_df.pivot_table(index=['clas_correct'], aggfunc='size')
    clas_df = pd.DataFrame(data={'is_correct': clas_series.index.values, 'count': clas_series})
    clas_df.reset_index(drop=True, inplace=True)
    clas_df.loc[clas_df['is_correct'] == 1, 'is_correct'] = 'correct'
    clas_df.loc[clas_df['is_correct'] == -1, 'is_correct'] = 'incorrect'
    clas_df.loc[clas_df['is_correct'] == 0, 'is_correct'] = 'uncovered'
    clas_df.to_csv(f'{fout_name}_{dt}.csv')


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
    pos_df, neg_df = has_vocab_words(tweets_df, pos_words, neg_words)
    return pos_df, neg_df, pos_words, neg_words


def has_vocab_words(df, pos_words=None, neg_words=None):
    pos_df = pd.DataFrame()
    neg_df = pd.DataFrame()

    for p in pos_words:
        df['w'] = p
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True].loc[df.label == 1]
        if a.empty:
            continue
        a.drop(columns=['has_word'], inplace=True)
        pos_df = pos_df.append(a)
    for n in neg_words:
        df['w'] = n
        df['has_word'] = [c in l for c, l in zip(df['w'], df['tweet_words'])]
        a = df.loc[df.has_word == True].loc[df.label == -1]
        if a.empty:
            continue
        a.drop(columns=['has_word'], inplace=True)
        neg_df = neg_df.append(a)
    pos_df.reset_index(drop=True, inplace=True)
    neg_df.reset_index(drop=True, inplace=True)
    return pos_df, neg_df


def print_words_df(df):
    pd.set_option('display.max_colwidth', -1)
    for column in df.columns:
        s = df.loc[df[column].notna()][column]
        print(column, ":", s)


def get_most_frequent(df, vocab1, vocab2):
    df['tweet_words'] = df['tweet_words'].astype(str)
    # creating tfidf vecotr
    stop_words = set(stopwords.words('english'))
    cv = CountVectorizer(max_df=0.85, stop_words=stop_words)
    word_count_vector = cv.fit_transform(df['tweet_words'])
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    words = cv.get_feature_names()
    # creating df with word count
    word_count = word_count_vector.toarray().sum(axis=0)
    word_count_df = pd.DataFrame(data={'word': words, 'word_count': word_count})
    # finding words that are already in vocab1
    word_count_df['to_remove'] = word_count_df['word'].isin(vocab1)
    word_count_df.drop(word_count_df.loc[word_count_df.to_remove == True].index, inplace=True)
    # finding words that are already in vocab2
    word_count_df['to_remove'] = word_count_df['word'].isin(vocab2)
    # removing words from either vocab
    word_count_df.drop(word_count_df.loc[word_count_df.to_remove == True].index, inplace=True)
    word_count_df.drop(columns=['to_remove'], inplace=True)
    word_count_df.sort_values(by='word_count', inplace=True, ascending=False)
    word_count_df.reset_index(inplace=True, drop=True)
    return word_count_df


def make_special_words(pos_df, neg_df):
    pos_v = pos_df['word'].to_list()
    neg_v = neg_df['word'].to_list()
    pos_df['is_special'] = ~pos_df['word'].isin(neg_v)
    neg_df['is_special'] = ~neg_df['word'].isin(pos_v)
    # was better without this and just took new special words by hand
    # finding 'almost special words' i.e appear 3.5 times more in one than the other
    pos_neg_words = pos_df.loc[pos_df['word'].isin(neg_v)]
    pos_neg_words.rename(columns={'word_count': 'pos_word_count'}, inplace=True)
    pos_neg_words.drop(columns=['is_special'], inplace=True)
    pos_neg_words = pos_neg_words.merge(neg_df, on='word')
    pos_neg_words.rename(columns={'word_count': 'neg_word_count'}, inplace=True)
    pos_neg_words.drop(columns=['is_special'], inplace=True)
    pos_neg_words['pos_special'] = pos_neg_words['pos_word_count'] >= pos_neg_words['neg_word_count'] * 3.5
    pos_neg_words['neg_special'] = pos_neg_words['neg_word_count'] >= pos_neg_words['pos_word_count'] * 3.5
    # keeping only special words from original special words df
    pos_df = pos_df[pos_df['is_special'] == True]
    neg_df = neg_df[neg_df['is_special'] == True]
    new_pos_special = pos_neg_words.loc[pos_neg_words['pos_special'] == True]
    new_pos_special.drop(columns=['neg_word_count', 'neg_special'], inplace=True)
    new_pos_special.rename(columns={'pos_word_count': 'word_count', 'pos_special': 'is_special'}, inplace=True)
    pos_df = pos_df.append(new_pos_special)
    new_neg_special = pos_neg_words.loc[pos_neg_words['neg_special'] == True]
    new_neg_special.drop(columns=['pos_word_count', 'pos_special'], inplace=True)
    new_neg_special.rename(columns={'neg_word_count': 'word_count', 'neg_special': 'is_special'}, inplace=True)
    neg_df = neg_df.append(new_neg_special)
    return pos_df, neg_df


def get_bad_word(df, vocab):
    bad_words_ls = []
    for index, row in df.iterrows():
        words = row.tweet_words
        added_word = False
        for w in words:
            if w in vocab:
                bad_words_ls.append(w)
                added_word = True
                break
        if not added_word:
            bad_words_ls.append(None)
        # will only get here if it doesn't break
    df['bad_word'] = bad_words_ls
    return df


def remove_bad_words():
    tweets_df, pos_words, neg_words = create_df_and_vocab_ls()
    res_df = classify_by_vocab(tweets_df, pos_words, neg_words)
    wrong_class = res_df.loc[res_df['clas_correct'] == -1]
    one_wrong_word_df = wrong_class.loc[wrong_class['label'] == 1].loc[wrong_class['neg_words_count'] == 1]
    one_wrong_word_df = one_wrong_word_df
    one_wrong_word_df = one_wrong_word_df.append(
        wrong_class.loc[wrong_class['label'] == -1].loc[wrong_class['pos_words_count'] == 1])
    pos_one_wrong_word_df = one_wrong_word_df.loc[one_wrong_word_df['label'] == 1]
    neg_one_wrong_word_df = one_wrong_word_df.loc[one_wrong_word_df['label'] == -1]
    pos_one_wrong_word_df = get_bad_word(pos_one_wrong_word_df, neg_words)
    neg_one_wrong_word_df = get_bad_word(neg_one_wrong_word_df, pos_words)
    check_df = res_df.loc[res_df['vocab_words_dif'] == 1]
    pos_check_df = check_df.loc[check_df['label'] == 1].loc[check_df['classifier'] == 1]
    neg_check_df = check_df.loc[check_df['label'] == -1].loc[check_df['classifier'] == -1]
    # the bad words are in apposite lists, that's why they are bad
    neg_bad_words = pos_one_wrong_word_df['bad_word'].to_list()
    pos_bad_words = neg_one_wrong_word_df['bad_word'].to_list()
    # checking whether there are 'bad' word that are crucial for other classifications,
    # i.e without them a tweet will be uncovered
    pos_check_df = get_bad_word(pos_check_df, pos_bad_words)
    neg_check_df = get_bad_word(neg_check_df, neg_bad_words)
    pos_check_list = pos_check_df['bad_word'].to_list()
    neg_check_list = neg_check_df['bad_word'].to_list()
    for w in pos_bad_words:
        if w in pos_check_list:
            pos_bad_words.remove(w)
    for w in neg_bad_words:
        if w in neg_check_list:
            neg_bad_words.remove(w)
    remove_words_from_file(pos_bad_words, pos_words, 'vocabulary_words/positive_words_clean.txt')
    remove_words_from_file(neg_bad_words, neg_words, 'vocabulary_words/negative_words_clean.txt')


def remove_words_from_file(words_ls, vocab, fname):
    for w in words_ls:
        if w in vocab:
            vocab.remove(w)
    with open(fname, 'w+') as filehandle:
        for word in vocab:
            filehandle.write('%s\n' % word)


if __name__ == "__main__":
    # remove_bad_words()

    run_classification('vocab_classifier_results/vocab_classifier_res__after_clean')
    # pos_words_df, neg_words_df, pos_words, neg_words = create_tweets_w_vocab_df()
    # new_pos_words = get_most_frequent(pos_words_df, pos_words, neg_words)
    # new_neg_words = get_most_frequent(neg_words_df, neg_words, pos_words)
    # new_pos_words, new_neg_words = make_special_words(new_pos_words, new_neg_words)
    # new_pos_words.to_csv(f'vocab_classifier_results/new_pos_words_{dt}.csv')
    # new_neg_words.to_csv(f'vocab_classifier_results/new_neg_words_{dt}.csv')

a = 1
