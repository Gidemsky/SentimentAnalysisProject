from shutil import copyfile

from support.Utils import get_json_tweet_list, create_json_dict_file, separate_debug_print_big


def initialize_data():
    """
    initialize all the lists: the main tweets data, the labeled and those
    the user couldn't label
    :return: the 3 list mentioned above
    """
    return get_json_tweet_list('Temp files/translated_tweets_5000.json'),\
           get_json_tweet_list('Temp files/labled_tweets.json'),\
           get_json_tweet_list('Temp files/problem_tweets.json')


def print_tweet_data(cur_tweet):
    """
    print the data line by line
    :param cur_tweet: a single data tweet
    :return:
    """
    data = []
    if 'extended_tweet' in cur_tweet:
        data.append(cur_tweet['extended_tweet']['full_text'][0])
    else:
        data.append(cur_tweet['text'][0])
    print("The tweet data:\n")
    for d in data[0]:
        print(d, ':', data[0][d])  # TODO: check about more fields
    print("\n")


if __name__ == '__main__':
    unlabeled, labeled, problematic_tweets = initialize_data()

    for i in range(len(unlabeled)):

        if i % 25 == 0 and not i == 0:
            dst = 'Temp files/backup/labeled_tweets_until_' + str(i) + '.json'
            copyfile('Temp files/labled_tweets.json', dst)

        if input("Do you want to label a tweet?\nPlease press 0 - for no, 1 - for yes\n") == '1':
            separate_debug_print_big("label number " + str(i + 1))
            t = unlabeled[i]
            print_tweet_data(t)
            if input("Do you want to label this tweet or skip to consult with the teammates?\n"
                     "Please press 0 - for collect to teammates, 1 - for continue labeling\n") == '0':
                problematic_tweets.append(t)
            else:
                neg_pos_lab = input("What is the level of positivity/negativity?\n"
                                    "Please press 1 (negative) up to 5 (positive)\n")
                if input("Is the tweet relative to a person or topic?\n"
                         "Please press 0 - for person, 1 - for topic\n") == '1':
                    pers_top = "topic"
                else:
                    pers_top = "person"
                cur_label = {'positivity': neg_pos_lab, 'relative subject': pers_top}
                t["label"] = cur_label
                labeled.append(t)
        else:
            print("Saving data...\n")
            unlabeled = unlabeled[i:]
            create_json_dict_file(unlabeled, 'Temp files/translated_tweets_5000.json')
            create_json_dict_file(labeled, 'Temp files/labled_tweets.json')
            create_json_dict_file(problematic_tweets, 'Temp files/problem_tweets.json')
            print("data saved!\ngoodbye")
            exit(0)
    print("here will be email")
