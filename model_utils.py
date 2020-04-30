import json
from datetime import datetime


def save_results(df, nm, is_trans):
    # get current date
    datetime_object = datetime.now()
    dt = datetime_object.strftime("%d_%m_%H_%M")
    fname = "results/" + nm + "_results_" + dt
    if is_trans:
        fname += "_tr"
    df.to_csv(fname + ".csv")
    df.to_excel(fname + ".xlsx")


def get_tweets(pos_f, neg_f):
    '''
    positive_tweets = []
    for line in open(pos_f):
        tweet = json.loads(line)
        tweet["label"] = 1
        positive_tweets.append(tweet)

    negative_tweets = []
    for line in open(neg_f):
        tweet = json.loads(line)
        tweet["label"] = -1
        negative_tweets.append(tweet)
    '''
    pos_d = json.load(open(pos_f))
    neg_d = json.load(open(neg_f))
    positive_tweets = pos_d['tweets']
    negative_tweets = neg_d['tweets']
    for itemP in positive_tweets:
        itemP.update({"label": 1})

    for itemN in negative_tweets:
        itemN.update({"label": -1})

    return positive_tweets, negative_tweets
