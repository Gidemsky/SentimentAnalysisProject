from sklearn.metrics import accuracy_score
from Models.modelHelperBase import *

class modelHelperTest(modelHelperBase):
    def __init__(self):
        """
        constructor
        """
        super().__init__()

    def get_train_and_test_sets(self, tweets_ids, processed_features, labels):
        """
        separate data to train and test sets
        :param tweets_ids: tweets ids
        :param processed_features: vectorized features
        :param labels: labels
        :return: separated data
        """
        s = int(len(processed_features)*0.8)

        train_X = processed_features[:s]
        test_X = processed_features[s:]
        test_Y = labels[s:]
        test_ids = tweets_ids[s:]
        train_Y = labels[:s]
        fet_ids = tweets_ids[:s]

        return fet_ids, train_X, train_Y, test_ids, test_X, test_Y

    def get_test_accuracy(self, predicted, test_labels):
        """
        :return: accuracy - based on comparing predictions and known labels
        """
        return accuracy_score(test_labels, predicted)