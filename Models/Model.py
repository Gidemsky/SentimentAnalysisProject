from Models.modelHelperBase import *
from Models.model_utils import separate_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd

MODEL_NAME = "random forest"
SUBJECTIVITY_MODEL_NAME = "svm"
POLARITY_MODEL_FILE = "polarity_model.joblib"
SUBJECTIVITY_MODEL_FILE = "subjectivity_model.joblib"
TRESHOLD = 0.6


class Model:
    def __init__(self):
        """
        model constructor
        """
        self.model_helper = modelHelperBase()
        self.filtered_train_set = None
        self.filtered_test_set = None

    def from_words_to_vector(self, train_set, test_set):
        """
        :param train_set: train set sentences
        :param test_set: test set sentences
        :return: a numeric vector representation for the features sentences
        """
        train_ids, train_X, polarity_Y, subjectivity_Y = separate_data(train_set)
        test_ids, test_X, _, _ = separate_data(test_set)
        vectorizer = TfidfVectorizer(max_features=2500, min_df=0.05, max_df=0.85, stop_words=stopwords.words('english'))
        filtered_data_train, filtered_data_test = self.model_helper.filter_data(train_X, test_X, vectorizer)

        return (train_ids, filtered_data_train, polarity_Y, subjectivity_Y), (test_ids, filtered_data_test)

    def run_model(self, model_name, train_set, train_set_labels, test_set):
        """
        runs a specific model and gets prediction for test set
        :param model_name: name of current model
        :param train_set_labels: labels type for train set (polarity / subjectivity)
        :return: prediction and confidence of current model
        """
        self.model_helper.create_model(model_name)
        # self.model_helper.load_model(model_name)
        self.model_helper.train_model(model_name, self.filtered_train_set[0], train_set, train_set_labels)
        predictions = self.model_helper.test_model(model_name, self.filtered_test_set[0], test_set)
        accuracy = self.model_helper.get_accuracy()
        confidence = self.model_helper.get_confidence(model_name, test_set)
        print(model_name + ' Results:')
        print(" Cross Validation Accuracy: ", accuracy[1])

        return predictions, confidence

    def run(self, train_set, test_set):
        """
        runs one model for polarity predictions and the second for subjectivity predictions
        :param train_set: train set for training both models - each one with different labels type
        :param test_set: test set
        :return: predictions and confidence of polarity and subjectivity models
        """
        self.filtered_train_set, self.filtered_test_set = self.from_words_to_vector(train_set, test_set)

        # Train and test model for subjectivity results.
        s_predictions, s_confidence = self.run_model(SUBJECTIVITY_MODEL_NAME, self.filtered_train_set[1],
                                                     self.filtered_train_set[3], self.filtered_test_set[1])

        regressed_train_set = self.get_regressed_features()

        # Train and test model for polarity results.
        p_predictions, p_confidence = self.run_model(MODEL_NAME, regressed_train_set,
                                                     self.filtered_train_set[2], self.filtered_test_set[1])

        return p_predictions, p_confidence, s_predictions, s_confidence

    # def get_regressed_features(self):
    #     model = self.model_helper.get_params(SUBJECTIVITY_MODEL_NAME)
    #     train_set = self.model_helper.resize_data(model, self.filtered_train_set[1])
    #     test_set = self.model_helper.resize_data(model, self.filtered_test_set[1])
    #     return train_set, test_set

    def get_regressed_features(self):
        bad_indices = self.model_helper.get_bad_indices(SUBJECTIVITY_MODEL_NAME)[0]
        train_set = self.reduce_weight(bad_indices)
        return train_set.values

    def reduce_weight(self, bad_indices):
        train_df = pd.DataFrame(self.filtered_train_set[1])
        train_df[bad_indices] = train_df[bad_indices] * TRESHOLD
        # test_df = pd.DataFrame(self.filtered_test_set[1])
        # test_df[bad_indices] = test_df[bad_indices] * TRESHOLD
        return train_df
