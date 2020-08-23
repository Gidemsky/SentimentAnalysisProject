"""
This script is charge for calculate and create new JSON with the final and average numbers
This takes all the JSONs stored and checks the average of the labels
"""
from support.JsonManager import JsonManager
from support.Utils import script_opener, json_files_collector, marge_all_json_file, check_duplicate_tweet, \
    dir_checker_creator, retweet_checker

LABELED_JSONS_TEMP = 'Temp files/backup/old_labelers_action'
LABELED_JSONS = 'Temp files/labeled files'
UNLABELED_JSON = 'Temp files/unlabeled JSON'


def average_tweet_calc(tweet, positivity, relativity):
    """
    calculates the real average of the label
    :param tweet: the tweet we want to work with
    :param positivity: number
    :param relativity: number
    :return: the tweet with the new labels value
    """
    number_of_labelers = tweet["JSON-Manager"]["labelers"]
    previous_label_positivity = tweet["label"]["positivity"]
    previous_label_relativity = tweet["label"]["relative subject"]

    # calc the value of the labels that where before
    prev_val_positivity = previous_label_positivity * number_of_labelers
    prev_val_relativity = previous_label_relativity * number_of_labelers

    # writes the new average label number
    tweet["JSON-Manager"]["labelers"] = number_of_labelers + 1
    tweet["label"]["positivity"] = (prev_val_positivity + positivity) / (number_of_labelers + 1)
    tweet["label"]["relative subject"] = (prev_val_relativity + relativity) / (number_of_labelers + 1)

    return tweet


def change_tweet_relative_label(tweet_to_refactor):
    """
    changes the format from name label to number
    :param tweet_to_refactor: some tweet. this is a dictionary
    :return:
    """
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
        # print("this is unlabeled tweet") TODO: change the exception to condition
        return None


def refactor_labels_back(new_fixed_tweets):
    """
    refactor the value back to the model using format
    :param new_fixed_tweets: the final list
    :return: the final list with the refactor values
    """
    for tweet in new_fixed_tweets:
        tweet["label"]["positivity"] = int(tweet["label"]["positivity"])
        if tweet["label"]["relative subject"] >= 0.5:
            tweet["label"]["relative subject"] = "person"
        else:
            tweet["label"]["relative subject"] = "topic"
    return new_fixed_tweets


def main_json_merger():

    new_fixed_json = list()
    # for every tweet check by the id the sum of all the next tweets
    for tweet in initial_merged_json:
        checking_id = tweet["id"]
        # during the loop it refactors the label format for average sum purpose
        fixed_tweet = change_tweet_relative_label(initial_merged_json[0])
        del initial_merged_json[0]

        if fixed_tweet is None:
            continue

        # next for every tweet we calculate the average with all the suitable tweets with the same id use
        for cmp_tweet, i in zip(initial_merged_json, range(initial_merged_json.__len__())):
            if cmp_tweet["id"] != checking_id:
                continue

            cmp_tweet = change_tweet_relative_label(cmp_tweet)
            if cmp_tweet is None:
                del initial_merged_json[i]
                continue

            # calculate the average
            fixed_tweet = average_tweet_calc(fixed_tweet, cmp_tweet["label"]["positivity"],
                                             cmp_tweet["label"]["relative subject"])
            del initial_merged_json[i]

        # adds the new tweet to the final list
        new_fixed_json.append(fixed_tweet)

    new_fixed_json = refactor_labels_back(new_fixed_json)
    return new_fixed_json


def labeled_and_unlabeled_json_creator():

    # creates the main labeled list
    print("\nCollecting labeled tweets...")
    json_files_list = json_files_collector(path=LABELED_JSONS)
    initial_merged_json = marge_all_json_file(file_list=json_files_list)
    labeled_total_list = check_duplicate_tweet(initial_merged_json)

    # creates the main unlabeled list
    print("\nCollecting labeled tweets...")
    json_files_list = json_files_collector(path=UNLABELED_JSON)
    initial_merged_json = marge_all_json_file(file_list=json_files_list)
    unlabeled_total_list = retweet_checker(check_duplicate_tweet(initial_merged_json))
    # unlabeled_total_list = manager.create_json_with_quotes()

    manager = JsonManager(unlabeled_total_list)
    manager.remove_double_tweets(labeled_total_list)
    manager.save_new_json_manager_file(list_to_be_saved=labeled_total_list, name='new total labeled')
    manager.summarize_labeled_tweets(json_to_summarize=labeled_total_list)


if __name__ == '__main__':
    script_opener("JSON merger")
    new_json = list()

    user_choose = input("\nPlease choose your action:\nFor regular file merge - press 1\n"
                        "For creating new labeled and unlabeled JSONs - press 2\n")

    if user_choose == '1':
        # gathering all the tweets together
        json_files_list = json_files_collector(path=LABELED_JSONS_TEMP)
        initial_merged_json = marge_all_json_file(file_list=json_files_list)
        new_json = main_json_merger()

    elif user_choose == '2':
        labeled_and_unlabeled_json_creator()
        print("New unlabeled and labeled file has been created\n")

    else:
        print("bad input")

    print('DONE - bye')

