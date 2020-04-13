import json
from tweepy.streaming import StreamListener

# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            all_data = json.loads(data)
            self.append_json(all_data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def write_json(self, data):
        with open(self.fetched_tweets_filename, 'w', encoding="utf-8") as tf:
            json.dump(data, tf, ensure_ascii=False)

    def append_json(self, data):
        with open(self.fetched_tweets_filename, 'r', encoding="utf-8") as json_file:
            old_json = json.load(json_file)
            temp = old_json['tweets']
            temp.append(data)
            self.write_json(old_json)

    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)
