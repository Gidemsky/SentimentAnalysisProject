"""
A support class for future expanding to manage and control the tweets
"""

from support.Utils import get_json_tweet_list, create_json_dict_file, check_tweets_number


class JsonManager(object):

    def __init__(self, json_file_name):
        self.json_list = get_json_tweet_list(json_file_name)
        self.new_tweet_list = list()

    def create_json_with_quotes(self):
        """
        Creates new unlabeled data with suitable tweet's quote
        This function works with auxiliary file named temp_quoted_status_retweets (quoted_list)
        :return: new_quoted_list -> the final list with all the necessary quotes
        """
        new_quoted_list = list()
        quotes_ids = list()
        quoted_list = get_json_tweet_list('Temp files/temp_quoted_status_retweets.json')

        # checks if the file is new one according to the length of the list
        if check_tweets_number(quoted_list) > 0:
            # in case this file has been already saved -> load all the tweet's ids
            for id_from_t in quoted_list:
                quotes_ids.append(id_from_t['id'])

        added_tweets = 0
        total_quoted_status = 0

        # run all over the tweet's member list for checking the tweet's quotes
        for t in self.json_list:
            # in case the current tweet doesn't belong to quoted_status
            if 'quoted_status' not in t:
                # add the tweet to the final tweets list
                self.new_tweet_list.append(t)
            else:
                # in case it does we check if it has already been saved to the list before
                # first save the tweet
                self.new_tweet_list.append(t)
                total_quoted_status += 1

                # in case this tweets has been saved before do not proceed to the quote stage
                if t['quoted_status']['id'] in quotes_ids:
                    continue

                # second prepare the quote tweet to add to the final list too
                temp_tweet = t['quoted_status']
                # creates new dictionary tag in the tweet for future use
                temp_tweet['JsonManager'] = {"Origin": True}
                new_quoted_list.append(temp_tweet)
                # adds the tweet's id in order to avoid future reuse of the same tweet
                quotes_ids.append(t['quoted_status']['id'])
                added_tweets += 1

        # prints the results numbers
        print("Retweet status checked!")
        if added_tweets > 0:
            create_json_dict_file(new_quoted_list, 'Temp files/temp_quoted_status_retweets.json')
            print("{0} quoted_status tweets has been added from your tweet list of {1} total quoted_status\n"
                  .format(str(added_tweets), str(total_quoted_status)))
        else:
            print("No tweets has been removed. The JSON list is OK!\n")

        # adds the quotes to the final list and returns it
        self.new_tweet_list += new_quoted_list

        return self.new_tweet_list
