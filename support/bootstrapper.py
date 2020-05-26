from support.Utils import get_json_tweet_list

TEST_RATIO = 5
VALIDATION_CONST = 0.7  # TODO: decide the constant


class Bootstrapper(object):

    def __init__(self, model, data_test, data_set):
        self.my_model = model
        self.none_labeled_tweets = data_test
        self.model_data_set = data_set
        self.final_data = list()

    def execute(self):
        while self.none_labeled_tweets:  # TODO: add prints and parameter for debuging
            my_model_test_tweets = self.get_test_tweets(self.none_labeled_tweets)
            model_results = self.my_model.run(my_model_test_tweets, self.model_data_set)
            self.validate_model_solution(model_results)
            #  TODO: add prints and more stuff to trace the progress

    def get_test_tweets(self, tweets):
        test_tweets = tweets
        test_tweets = test_tweets[:int(len(self.model_data_set)*(TEST_RATIO/100))]
        return test_tweets

    def validate_model_solution(self, results):
        for res in results:
            if res.probabilty >= VALIDATION_CONST:
                self.final_data.append(res)
                self.model_data_set.append(res)
            else:
                self.none_labeled_tweets.append(res)