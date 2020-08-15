import json
from support.Utils import get_json_tweet_list
from datetime import datetime
import pandas as pd

TRAIN_FILE = "C:\\Users\\t-orahar\PycharmProjects\SentimentAnalysis\SentimentAnalysisProject\Models\Data\\bootstrapped_train_set.json"
TEST_FILE = "C:\\Users\\t-orahar\PycharmProjects\SentimentAnalysis\SentimentAnalysisProject\support\Temp files\oriya_trans.json"


def save_results(df, nm, is_trans):
    """
    saves results for test session in csv file
    :param df: data frame contains the predictions
    :param nm: model name
    :param is_trans: bool, is data translated
    """
    # get current date
    datetime_object = datetime.now()
    dt = datetime_object.strftime("%d_%m_%H_%M")
    fname = "results/" + nm + "_results_" + dt
    if is_trans:
        fname += "_tr"
    df.to_csv(fname + ".csv")


def get_tweets(pos_f, neg_f):
    """
    deserialize tweets for test session
    :param pos_f: name of the positive tweets file
    :param neg_f: name of the negative tweets file
    :return: positive and negative tweets
    """
    with open(pos_f, 'r', encoding="utf-8") as pos_json_file:
        pos_d = json.load(pos_json_file)
    with open(neg_f, 'r', encoding="utf-8") as neg_json_file:
        neg_d = json.load(neg_json_file)
    positive_tweets = pos_d['tweets']
    negative_tweets = neg_d['tweets']
    for itemP in positive_tweets:
        itemP.update({"label": 1})

    for itemN in negative_tweets:
        itemN.update({"label": -1})

    return positive_tweets, negative_tweets


def get_train_test_tweets():
    train_set = get_json_tweet_list(TRAIN_FILE)
    test_set = get_json_tweet_list(TEST_FILE)

    return train_set, test_set


def get_vocabulary():
    """
    deserialize vocabulary file
    :return: list of words
    """
    with open("positive-words.txt", 'r') as file:
        vocabulary = file.read().split('\n')[:1000]
    with open("negative-words.txt", 'r') as file:
        vocabulary += file.read().split('\n')[:1000]
    return list(set(dict.fromkeys(vocabulary)))


def separate_data(data):
    """
    separate data to features and labels
    :param data: original data
    :return: separated data
    """
    polarity = None
    subjectivity = None
    features = []
    df_data = pd.DataFrame(data)
    if 'label' in df_data.columns:
        df_data['label'] = df_data['label'].astype(str)
        df_data['polarity'] = df_data['label'].str.slice(16, 17)
        df_data.polarity = df_data.polarity.astype(int)
        df_data['subjectivity'] = df_data['label'].str.slice(41, -2)
        df_data['is_topic'] = df_data['subjectivity'] == 'topic'
        df_data.is_topic = df_data.is_topic.astype(int)
        polarity = list(df_data.polarity)
        subjectivity = list(df_data.is_topic)
    ids = df_data.iloc[:, 2].values
    for _, item in df_data.iterrows():
        if type(item['extended_tweet']) is not float:
            features.append(item['extended_tweet']['full_text'][0]['input'])
        else:
            features.append(item['text'][0]['input'])
    return ids, features, polarity, subjectivity