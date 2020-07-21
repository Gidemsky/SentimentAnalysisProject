import json
import random
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from modelHelper import *


def remove_none_vocab_words(processed_tweets, vocab1, vocab2):
    tweets_in_vocab = []
    for t in processed_tweets:
        t_ls = t.split(' ')
        keep = [x for x in t_ls if x in vocab1 or x in vocab2]
        keep_str = ' '.join(keep)
        tweets_in_vocab.append(keep_str)
    return tweets_in_vocab


def create_vocabs(p_wrds_fname, n_wrds_fname):
    pos_wrds_fname = f'vocab_classifier/vocabularies/{p_wrds_fname}'
    neg_wrds_fname = f'vocab_classifier/vocabularies/{n_wrds_fname}'

    with open(pos_wrds_fname, encoding="utf8") as f:
        pos_words = f.read().split('\n')

    with open(neg_wrds_fname, encoding="utf8") as f1:
        neg_words = f1.read().split('\n')
    return pos_words, neg_words


def create_feature_weights():
    fPos = "twitter_samples/negative_tweets1.json"
    fNeg = "twitter_samples/positive_tweets1.json"
    is_trans = True
    pos_vocab, neg_vocab = create_vocabs('positive_words_clean.txt', 'negative_words_clean.txt')
    positive_tweets, negative_tweets = mUtils.get_tweets(fPos, fNeg)

    neg_pos_tweets = positive_tweets + negative_tweets
    random.shuffle(neg_pos_tweets)

    neg_pos_tweets = pd.DataFrame(neg_pos_tweets)

    labels = neg_pos_tweets.iloc[:, 24].values

    tweets = neg_pos_tweets.iloc[:, 2].values
    ids = neg_pos_tweets.iloc[:, 23].values

    nBModel = modelHelper("svm", 5, labels, ids)
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_tweets = nBModel.process_tweets(tweets)
    processed_tweets = remove_none_vocab_words(processed_tweets, pos_vocab, neg_vocab)
    nBModel.create_features(processed_tweets, vectorizer)
    nBModel.model.fit(nBModel.processed_features, nBModel.labels)
    fet_Weights = nBModel.model.coef_[0]
    featureNames = vectorizer.get_feature_names()
    names_and_weights = pd.DataFrame(data={'feature': featureNames, 'weight': fet_Weights})
    return names_and_weights


if __name__ == "__main__":
    create_feature_weights()
    a = 1
