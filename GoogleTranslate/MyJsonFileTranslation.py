import json
import Utils as Tool

from GoogleTranslate.GoogleTranslateAPI import GoogleTranslateAPI

LOGGING = True
TWEETS_LIMIT = 75

JSON_NAME1 = 'negative_tweets.json'
JSON_NAME2 = 'positive_tweets.json'

google = GoogleTranslateAPI()


def temp_convert_json_to_list(json_name):
    tweets = []
    for line in open(json_name):
        tweet = json.loads(line)
        tweet["label"] = 1
        tweets.append(tweet)
    return tweets


def json_translation(json_list):
    i = 0
    for single in enumerate(json_list):
        if LOGGING:
            i += 1
            Tool.separate_debug_print_small('The ' + str(i) + ' translation')
        single[1]['text'] = google.translate(single[1]['text'], 'iw', text_language='en')
        if single[1]['user']['description'] is not None:
            single[1]['user']['description'] = google.translate(single[1]['user']['description'], 'iw',
                                                                text_language='en')
        if 'retweeted_status' in single[1]:
            single[1]['retweeted_status']['text'] = google.translate(single[1]['retweeted_status']['text'], 'iw',
                                                                     text_language='en')
            if 'extended_tweet' in single[1]['retweeted_status']:
                single[1]['retweeted_status']['extended_tweet']['full_text'] = google.translate(
                    single[1]['retweeted_status']['extended_tweet']['full_text'], 'iw', text_language='en')
        if 'extended_tweet' in single[1]:
            single[1]['extended_tweet']['full_text'] = google.translate(single[1]['extended_tweet']['full_text'], 'iw',
                                                                        text_language='en')


# json_all_list = json_all_list[:20]
# for tweet in enumerate(json_all_list):
#     for value in tweet[1].values():
#         if isinstance(value, str) or isinstance(value, list):
#             result = google.language_detection(value)


if __name__ == '__main__':

    # json_all_list = Tool.get_json_list('negative_tweets.json')
    list_to_translate = temp_convert_json_to_list(JSON_NAME2)
    list_to_translate = list_to_translate[:TWEETS_LIMIT]

    json_translation(list_to_translate)

    result_json_name = 'translated_result ' + JSON_NAME2

    Tool.create_json_dict_file(list_to_translate, result_json_name)

    email_msg = "There is translated limited json file for now"
    Tool.send_report_by_email(mail_subject="Translated JSON test file", body_text=email_msg,
                              file_path=result_json_name)
