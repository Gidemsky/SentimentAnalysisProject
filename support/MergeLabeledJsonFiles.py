import os

from support.Utils import script_opener, get_json_tweet_list, check_tweets_number
from support.Utils import create_json_dict_file

def json_files_collector():
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk('Temp files/backup/old_labelers_action'):
        for file in f:
            if '.json' in file:
                files.append(os.path.join(r, file))

    return files


def marge_all_json_file():

    counter_all_lists = 0

    for f in json_files_list:
        cur_json = get_json_tweet_list(f)
        initial_merged_json.extend(cur_json)
        counter_all_lists += check_tweets_number(cur_json)

    # return counter_all_lists


def average_tweet_calc(tweet, positivity, relativity):
    number_of_labelers = tweet["JSON-Manager"]["labelers"]
    previous_label_positivity = tweet["label"]["positivity"]
    previous_label_relativity = tweet["label"]["relative subject"]
    prev_val_positivity = previous_label_positivity * number_of_labelers
    prev_val_relativity = previous_label_relativity * number_of_labelers
    tweet["JSON-Manager"]["labelers"] = number_of_labelers + 1
    tweet["label"]["positivity"] = (prev_val_positivity + positivity) / (number_of_labelers + 1)
    tweet["label"]["relative subject"] = (prev_val_relativity + relativity) / (number_of_labelers + 1)
    return tweet


def change_tweet_relative_label(tweet_to_refactor):
    try:
        tweet_to_refactor["label"]["positivity"] = float(tweet_to_refactor["label"]["positivity"])
        if tweet_to_refactor["label"]["relative subject"] == "person":
            tweet_to_refactor["label"]["relative subject"] = 1.0
        else:
            tweet_to_refactor["label"]["relative subject"] = 0.0
        tweet_to_refactor["JSON-Manager"] = {"labelers": 1,
                                             "relative-number": 0.0}
        return tweet_to_refactor
    except:
        print("this is unlabeled tweet")
        return None


def refactor_labels_back(new_fixed_tweets):
    for tweet in new_fixed_tweets:
        tweet["label"]["positivity"] = int(tweet["label"]["positivity"])
        if tweet["label"]["relative subject"] >= 0.5:
            tweet["label"]["relative subject"] = "person"
        else:
            tweet["label"]["relative subject"] = "topic"
    return new_fixed_tweets


if __name__ == '__main__':
    script_opener("Labeled JSON merger")

    initial_merged_json = list()

    new_fixed_json = list()

    # path = 'c:\\projects\\hc2\\'

    json_files_list = json_files_collector()

    marge_all_json_file()
    counter = check_tweets_number(initial_merged_json)

#    create_json_dict_file(initial_merged_json, "C:\\Users\\t-orahar\PycharmProjects\SentimentAnalysis\SentimentAnalysisProject\support\Temp files\\backup.labeled_tweets_collection.json")
    for tweet in initial_merged_json:
        checking_id = tweet["id"]
        fixed_tweet = change_tweet_relative_label(initial_merged_json[0])
        del initial_merged_json[0]
        if fixed_tweet is None:
            continue
        # initial_merged_json.remove(0)
        for cmp_tweet, i in zip(initial_merged_json, range(initial_merged_json.__len__())):
            if cmp_tweet["id"] != checking_id:
                continue
            cmp_tweet = change_tweet_relative_label(cmp_tweet)
            if cmp_tweet is None:
                del initial_merged_json[i]
                continue
            fixed_tweet = average_tweet_calc(fixed_tweet, cmp_tweet["label"]["positivity"], cmp_tweet["label"]["relative subject"])
            print("removeing tweet number: " + str(cmp_tweet["id"]))
            del initial_merged_json[i]
            # initial_merged_json.remove(i)
        new_fixed_json.append(fixed_tweet)

    print(check_tweets_number(new_fixed_json))
    new_fixed_json = refactor_labels_back(new_fixed_json)
