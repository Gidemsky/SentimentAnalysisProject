import json

from GoogleTranslate.GoogleTranslateAPI import GoogleTranslateAPI
from Utils import separate_debug_print_small, send_report_by_email, create_json_dict_file, get_json_list

LOGGING = False
TWEETS_LIMIT = 5000

MAIN_JSON_FILE = 'tweets.json'

# creates the google access API
google = GoogleTranslateAPI()


# convert list of json to list
def temp_convert_json_to_list(json_name):
    tweets = []
    for line in open(json_name):
        tweet = json.loads(line)
        tweet["label"] = 1
        tweets.append(tweet)
    return tweets


def json_translation(json_list):
    """
    Json's fields translation function.
    This function translates the following fields (if there are exist):
    + "text"
    + "user" -> "description"
    + "retweeted_status" -> "text"
    + "retweeted_status" -> "extended_tweet" -> "full_text"
    + "extended_tweet" -> "full_text"
    :param json_list: list we want to translate
    :return:
    """
    i = 0
    try:
        for tweet in enumerate(json_list):
            # in case we want to print the results
            i += 1
            if LOGGING:
                separate_debug_print_small('The ' + str(i) + ' translation')
            # "text"
            tweet[1]['text'] = google.translate(tweet[1]['text'], 'en', text_language='iw', level=2)
            if tweet[1]['user']['description'] is not None:
                # "user" -> "description"
                tweet[1]['user']['description'] = google.translate(tweet[1]['user']['description'], 'en',
                                                                   text_language='iw', level=2)
            if 'retweeted_status' in tweet[1]:
                # "retweeted_status" -> "text"
                tweet[1]['retweeted_status']['text'] = google.translate(tweet[1]['retweeted_status']['text'], 'en',
                                                                        text_language='iw', level=2)
                if 'extended_tweet' in tweet[1]['retweeted_status']:
                    # "retweeted_status" -> "extended_tweet" -> "full_text"
                    tweet[1]['retweeted_status']['extended_tweet']['full_text'] = google.translate(
                        tweet[1]['retweeted_status']['extended_tweet']['full_text'], 'en', text_language='iw', level=2)
            if 'extended_tweet' in tweet[1]:
                # "extended_tweet" -> "full_text"
                tweet[1]['extended_tweet']['full_text'] = google.translate(tweet[1]['extended_tweet']['full_text'],
                                                                           'en',
                                                                           text_language='iw', level=2)
            if i % 2500 == 0:
                send_report_by_email(mail_subject="Translated Tweets",
                                     body_text=str(i) + ' tweets has successfully translated!')
        send_report_by_email(mail_subject="Translated Tweets",
                             body_text=str(i) + ' tweets has successfully translated!')
    except Exception:
        temp_result_json = 'translated_' + MAIN_JSON_FILE
        print(str(i))
        create_json_dict_file(json_list, temp_result_json)


if __name__ == '__main__':
    list_to_translate = get_json_list('../Twitter_Api/' + MAIN_JSON_FILE)
    # list_to_translate = list_to_translate[TWEETS_LIMIT:]

    json_translation(list_to_translate)

    result_json_name = 'translated_' + MAIN_JSON_FILE

    create_json_dict_file(list_to_translate, result_json_name)
