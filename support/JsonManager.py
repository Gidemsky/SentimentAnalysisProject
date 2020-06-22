from support.Utils import get_json_tweet_list, create_json_dict_file


class JsonManager(object):

    def __init__(self, json_file_name):
        self.json_list = get_json_tweet_list(json_file_name)

    def create_json_with_quotes(self):
        quotes_ids = list()
        new_tweets_list = list()
        new_quoted_list = list()
        added_tweets = 0
        total_quoted_status = 0
        for t in self.json_list:
            if 'quoted_status' not in t:
                new_tweets_list.append(t)
            else:
                new_tweets_list.append(t)
                total_quoted_status += 1
                if t['quoted_status']['id'] in quotes_ids:
                    continue
                new_quoted_list.append(t['quoted_status'])
                quotes_ids.append(t['quoted_status']['id'])
                added_tweets += 1
        print("Retweet status checked!")
        if added_tweets > 0:
            create_json_dict_file(new_quoted_list, 'Temp files/temp_quoted_status_retweets.json')
            print("{0} quoted_status tweets has been added from your tweet list of {1} total quoted_status\n"
                  .format(str(added_tweets), str(total_quoted_status)))
        else:
            print("No tweets has been removed. The JSON list is OK!\n")

        return new_tweets_list + new_quoted_list
