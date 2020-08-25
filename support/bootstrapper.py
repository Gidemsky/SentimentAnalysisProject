from Models.Model import *
from support.Utils import create_json_dict_file, get_json_tweet_list
from Models import model_utils as mUtils
import nltk

TEST_RATIO = 5
VALIDATION_CONST = 0.7  # TODO: decide the constant
#TRAIN_FILE = "C:\\SentimentAnalysisProject\Models\Data\\bootstrapped_train_set.json"
MANUAL_LABELING_FILE = "C:\\SentimentAnalysisProject\\Models\Data\\manual_labeling.json"
TRAIN_FILE = "C:\\SentimentAnalysisProject\Models\Data\\labeled_tweets.json"

class Bootstrapper(object):
    """
    bootstrapper constructor
    :param model: model type
    :param train_set: train set for training the model - data and labels
    :param data_test: data for labeling
    """
    def __init__(self, model, train_set, data_test):
        self.my_model = model
        self.none_labeled_tweets = data_test
        self.model_data_set = train_set
        self.final_data = list()
        self.manual_labeling = []

    def execute(self):
        """
        runs a while loop - runs the model and gets predictions and adds the predicted data
        (with high probability) to the data_set - for training the model in the future.
        the loop runs until the test_set is empty.
        """
        while self.none_labeled_tweets:
            self.my_model_test_tweets = self.get_test_tweets()
            model_results, confidence, sub_results, sub_confidence\
                = self.my_model.run(self.model_data_set, self.my_model_test_tweets)
            self.validate_model_solution(model_results, confidence, sub_results, sub_confidence)
        self.save_new_train_set()
        return

    def get_test_tweets(self):
        """
        each time the function return a slice of the test_set
        :return: test set for current iteration
        """
        test_tweets = self.none_labeled_tweets
        ratio = int(len(self.model_data_set)*(TEST_RATIO/100))
        if len(test_tweets) > ratio:
            test_tweets = test_tweets[: ratio]
            self.none_labeled_tweets = self.none_labeled_tweets[ratio: -1]
        else:
            test_tweets = test_tweets[: -1]
            self.none_labeled_tweets = None
        return test_tweets

    def validate_model_solution(self, results, confidence, sub_results, sub_confidence):
        """
        validates prediction for each example in data test
        :param results: polarity predictions
        :param confidence: probability of polarity prediction
        :param sub_results: subjectivity predictions
        :param sub_confidence: probability of subjectivity predictions
        """
        for id, conf, res, sub_conf, sub_res in zip(results[0], confidence, results[1], sub_confidence, sub_results[1]):
            if conf[res-1] >= VALIDATION_CONST/5 and sub_conf[sub_res] >= VALIDATION_CONST:
                self.append_to_train_set(id, res, sub_res)
            else:
                self.manual_labeling.append(self.find_by_id(id))

    def append_to_train_set(self, id, result, sub_res):
        """
        append a new example from test set to train set - if it's probability is high enough
        :param id: example id
        :param result: example polarity prediction
        :param sub_res: example subjectivity prediction
        """
        tweet = self.find_by_id(id)
        tweet["label"] = {"positivity": result, "relative subject": sub_res}
        self.final_data.append(tweet)
        self.model_data_set.append(tweet)

    def find_by_id(self, id):
        """
        finds the tweet with the given id in test set
        :param id: id of the wanted tweet
        :return: wanted tweet
        """
        tweet = next(t for t in self.my_model_test_tweets if t["id_str"] == id)
        return tweet

    def save_new_train_set(self):
        """
        serialize the new train set - combined with old train set and data from test set
        """
        create_json_dict_file(self.model_data_set, TRAIN_FILE)
        create_json_dict_file(self.manual_labeling, MANUAL_LABELING_FILE)


if __name__ == '__main__':
    model = Model()
    get_json_tweet_list(MANUAL_LABELING_FILE)
    train, test = mUtils.get_train_test_tweets()
    bootStrapper = Bootstrapper(model, train, test)
    bootStrapper.execute()
