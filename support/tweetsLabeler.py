import re

from support.Utils import get_json_tweet_list, create_json_dict_file, separate_debug_print_big, send_report_by_email, \
    script_opener, separate_debug_print_small, dir_checker_creator

TRANSLATED_JSON = 'gidi_trans.json'
BACKUP_RATIO = 20
NAME = ''


def retweet_checker(json_list_to_check):
    """
    This function runs all over the source list and deletes all the retweets
    The deleted tweets will be saved as new file
    :param json_list_to_check: the source list t check
    :return: new list without retweets at all
    """
    print("Please wait, checking if there are retweets to delete...")
    deleted_tweets = 0
    deleted_list_tweets = list()
    new_tweets_list = list()
    for t in json_list_to_check:
        if 'retweeted_status' not in t:
            new_tweets_list.append(t)
        else:
            deleted_list_tweets.append(t)
            deleted_tweets += 1
    print("All tweets are checked!")
    if deleted_tweets > 0:
        create_json_dict_file(deleted_list_tweets, 'Temp files/temp_deleted_retweets.json')
        print("{0} retweeted tweets has been removed from your tweet list\n".format(str(deleted_tweets)))
    else:
        print("No tweets hs been removed. The list is OK!\n")
    return new_tweets_list


def initialize_data():
    """
    initialize all the lists: the main tweets data, the labeled and those
    the user couldn't label
    :return: the 3 list mentioned above
    """
    print("Loading json's data...\n")
    # create the directories for the program
    dir_checker_creator("Temp files/Backup")

    main_json_list = retweet_checker(get_json_tweet_list('Temp files/' + TRANSLATED_JSON))
    labeled_list = get_json_tweet_list('Temp files/labeled_tweets.json')
    problems_list = get_json_tweet_list('Temp files/problem_tweets.json')

    print("Loading completed!\nTotal tweets you has labeled: " + str(len(labeled_list)) +
          "\nTotal tweets you mark as problematic: " + str(len(problems_list)) +
          "\nTotal tweets you left the label: " + str(len(main_json_list)) +
          "\nPlease follow the instruction and good luck\n")

    return main_json_list, labeled_list, problems_list


def print_tweet_data(cur_tweet):
    """
    print the data line by line
    :param cur_tweet: a single data tweet
    :return:
    """
    data = []
    text_type = ""

    if 'retweeted_status' not in cur_tweet or 'quoted_status' not in cur_tweet:
        # a simple tweet
        if 'extended_tweet' in cur_tweet:
            data.append(cur_tweet['extended_tweet']['full_text'][0])
            text_type = "Full text - \n"
        else:
            data.append(cur_tweet['text'][0])
            text_type = "Short text - \n"

    else:
        # a complicated tweet
        data.append(cur_tweet['text'][0])
        text_type = "Short text - \n"

    # prints the tweet's text
    print(text_type)
    try:
        for d in data[0]:
            print(d, ':', data[0][d])  # TODO: check about more fields
    except TypeError:
        print("It seems like you have json format issue\n"
              "Please write down the iteration number {0}.\nExit the program now!\n".format(i))
        finalize_json_data()
        exit(1)


def finalize_json_data():
    """
    Summarize all the data that collected to JSONs files
    :return:
    """
    print("Saving data...")
    # saves the tweets we didn't passed yet
    new_unlabeled_list = unlabeled[i:]
    create_json_dict_file(new_unlabeled_list, 'Temp files/' + TRANSLATED_JSON)
    create_json_dict_file(labeled, 'Temp files/labeled_tweets.json')
    create_json_dict_file(problematic_tweets, 'Temp files/problem_tweets.json')
    print("data saved!\n")


def labeling_report(total_signed_tweets, is_finish_labeling=False):
    """
    The report stage to the group
    :param total_signed_tweets: the number of the all tweets we passed and signed
    :param is_finish_labeling: the case of sending the mail in order to know the mail's subject
    :return:
    """
    mail_body = "{0} tweets total\n{1} labeled tweets and {2} problematic tweet. :-)".format(
        str(total_signed_tweets), str(len(labeled)), str(len(problematic_tweets)))
    print("Sending email to the teammates...\n")
    if not is_finish_labeling:
        send_report_by_email(mail_subject=NAME + "'s label progress", body_text=mail_body)
    else:
        send_report_by_email(mail_subject=NAME + "'s label progress - DONE!", body_text=mail_body)


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


def backup_files():
    """
    backup the file we writing
    :return:
    """
    separate_debug_print_small("Backup the files now")
    labeled_backup_dst = 'Temp files/backup/labeled_tweets_until_' + str(len(labeled)) + '.json'
    problem_backup_dst = 'Temp files/backup/problem_tweets_until_' + str(len(problematic_tweets)) + '.json'
    unlabeled_backup_dst = 'Temp files/backup/unlabeled_tweets_until_' + str(len(unlabeled) - i) + '.json'
    create_json_dict_file(labeled, labeled_backup_dst)
    create_json_dict_file(problematic_tweets, problem_backup_dst)
    create_json_dict_file(unlabeled[i:], unlabeled_backup_dst)
    separate_debug_print_small("Backup done")


def main_labeler(t):
    labeler_status = True
    separate_debug_print_small("starting tweet's labeler")
    print_tweet_data(t)
    while labeler_status:
        user_action = input("\nDo you want to label this tweet or skip to consult with the teammates?\n"
                            "Please press:\n0 - for collect to teammates\n"
                            "1 - for continue labeling\n2 - see the text again\n3 - skip this tweet")
        if user_action == '0':
            problematic_tweets.append(t)
            labeler_status = False
        elif user_action == '1':
            # creating the label dictionary we want to append to the translated json
            t["label"] = {'positivity': tweet_pos_neg_labeler(),
                          'relative subject': relative_subject_labeler()}
            labeled.append(t)
            labeler_status = False
        elif user_action == '2':
            print_tweet_data(t)
        elif user_action == '3':
            print("Bye bye, you unuseful tweet, TFIEE!\n")
            labeler_status = False
        else:
            print("You entered wrong input!\nYou should enter 0, 1 or 2\nPlease try again\n")


if __name__ == '__main__':
    script_opener("Tweet Labeler")
    unlabeled, labeled, problematic_tweets = initialize_data()

    if len(unlabeled) == 0:
        print("You have empty json!\nPlease Check your tweet's file")

    i = 0
    # run main while loop as far as the is unlabeled tweets
    while i < len(unlabeled):
        # the backup staging every BACKUP_RATIO constant
        if i % BACKUP_RATIO == 0 and not i == 0:
            backup_files()

        user_main_choose = input("\nDo you want to label a tweet?\nPlease press:\n0 - for no\n1 - for yes\n")

        if user_main_choose == '1':
            separate_debug_print_big("label number " + str(i + 1))
            main_labeler(unlabeled[i])

        elif user_main_choose == '0':
            # sends report in order the user want
            if input("Do you want to share your progress?\n"
                     "Please press 1 - for yes, or anything else - for no\n") == '1':
                if NAME == "":
                    NAME = input("\nYou didn't entered your name. Please enter your name now:\n")
                labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets))
            finalize_json_data()
            print("Goodbye Chiquititas!!!")
            exit(0)

        else:
            print("You entered wrong input!\nPlease run again\n")
            # sub the i in order to run the same tweet again
            i -= 1

        i += 1

    # in case the translated file ended
    if i != 0:
        if NAME == "":
            NAME = input("\nYou didn't entered your name. Please enter your name now:\n")
        labeling_report(total_signed_tweets=len(labeled) + len(problematic_tweets))
