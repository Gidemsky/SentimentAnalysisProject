import re

from support.Utils import get_json_tweet_list, create_json_dict_file, separate_debug_print_big, send_report_by_email, \
    script_opener, separate_debug_print_small

TRANSLATED_JSON = 'translated_tweets_5000.json'
BACKUP_RATIO = 25
NAME = 'Gidi'


def initialize_data():
    """
    initialize all the lists: the main tweets data, the labeled and those
    the user couldn't label
    :return: the 3 list mentioned above
    """
    print("Loading json's data...\n")
    return get_json_tweet_list('Temp files/' + TRANSLATED_JSON), \
           get_json_tweet_list('Temp files/labeled_tweets.json'), \
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


def finalize_json_data(src_json, tweet_number_reached, labeled_final_data, problem_tweet):
    """
    Summarize all the data that collected to JSONs files
    :param src_json: the main json list we want to label
    :param tweet_number_reached: the iteration number of the loop
    :param labeled_final_data: the tweet's list I labeled
    :param problem_tweet: the tweet's list I hasn't labeled and signed as problematic
    :return:
    """
    print("Saving data...")
    # saves the tweets we didn't passed yet
    src_json = src_json[tweet_number_reached:]
    create_json_dict_file(src_json, 'Temp files/' + TRANSLATED_JSON)
    create_json_dict_file(labeled_final_data, 'Temp files/labeled_tweets.json')
    create_json_dict_file(problem_tweet, 'Temp files/problem_tweets.json')
    print("data saved!\n")


def labeling_report(total_signed_tweets):
    """
    The report stage to the group
    :param total_signed_tweets: the number of the all tweets we passed and signed
    :return:
    """
    mail_body = "{0} tweets total\n{1} labeled tweets and {2} problematic tweet. :-)".format(
        str(total_signed_tweets), str(len(labeled)), str(len(problematic_tweets)))
    send_report_by_email(mail_subject=NAME + "'s label progress", body_text=mail_body)


def relative_subject_labeler():
    """
    The relative stage. we choose here what kind of the tweet
    :return: topic / person as string
    """
    # user input for the relativity label
    relativity_type = input("Is the tweet relative to a person or topic?\n"
                            "Please press 0 - for person, 1 - for topic\n")
    # checks if the input is legal
    if re.search('^[0-1]', relativity_type) and relativity_type.__len__() == 1:
        if relativity_type == '1':
            return "topic"
        else:
            return "person"
    # run the input function again until the input is legal
    else:
        print("you pressed wrong number, please try again")
        return relative_subject_labeler()


def tweet_pos_neg_labeler():
    """
    The positive or negative labeling stage.
    we choose here the sign of the tweet from 1 up to 5
    :return: positivity label as string
    """
    neg_pos = input("What is the level of positivity/negativity?\n"
                    "Please press 1 (negative) up to 5 (positive)\n")
    # checks if the input is legal
    if re.search('^[1-5]', neg_pos) and neg_pos.__len__() == 1:
        if int(neg_pos) >= 1 or int(neg_pos) <= 5:
            return neg_pos
    # run the input function again until the input is legal
    else:
        print("you typed wrong input, please try again")
        return tweet_pos_neg_labeler()


if __name__ == '__main__':

    script_opener("Tweet Labeler")
    unlabeled, labeled, problematic_tweets = initialize_data()
    print("Loading completed!\nTotal tweets you has labeled: " + str(len(labeled)) +
          "\nTotal tweets you mark as problematic: " + str(len(problematic_tweets)) +
          "\nTotal tweets you left the label: " + str(len(unlabeled)) +
          "\nPlease follow the instruction and good luck\n")

    i = 0
    # run main while loop as far as the is unlabeled tweets
    while i < len(unlabeled):

        # the backup staging every BACKUP_RATIO constant
        if i % BACKUP_RATIO == 0 and not i == 0:
            separate_debug_print_small("Backuping the files now")
            labeled_backup_dst = 'Temp files/backup/labeled_tweets_until_' + str(len(labeled)) + '.json'
            problem_backup_dst = 'Temp files/backup/problem_tweets_until_' + str(len(problematic_tweets)) + '.json'
            create_json_dict_file(labeled, labeled_backup_dst)
            create_json_dict_file(problematic_tweets, problem_backup_dst)
            separate_debug_print_small("Backup done")

        user_main_choose = input("\nDo you want to label a tweet?\nPlease press 0 - for no, 1 - for yes\n")
        if user_main_choose == '1':
            separate_debug_print_big("label number " + str(i + 1))
            t = unlabeled[i]
            print_tweet_data(t)

            label_user_action = input("Do you want to label this tweet or skip to consult with the teammates?\n"
                                      "Please press 0 - for collect to teammates, 1 - for continue labeling\n")
            if label_user_action == '0':
                problematic_tweets.append(t)
            elif label_user_action == '1':
                # creating the label dictionary we want to append to the translated json
                t["label"] = {'positivity': tweet_pos_neg_labeler(),
                              'relative subject': relative_subject_labeler()}
                labeled.append(t)
            else:
                # subs the i in order to run the same tweet again
                print("You entered wrong input!\nStarting the labeling again\nPlease run again\n")
                i -= 1
        elif user_main_choose == '0':
            # sends report in order the user want
            if input("Do you want to share your progress?\nPlease press 0 - for no, 1 - for yes\n") == '1':
                labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets))
            finalize_json_data(unlabeled, i, labeled, problematic_tweets)
            print("Goodbye Chiquititas!!!")
            exit(0)

        else:
            print("You entered wrong input!\nPlease run again\n")
            # sub the i in order to run the same tweet again
            i -= 1
        i += 1
    # in case the translated file ended
    labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets))
