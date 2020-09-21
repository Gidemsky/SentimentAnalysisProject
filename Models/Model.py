import numpy

from Models.modelHelperBase import *
from Models.model_utils import separate_data, clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd

from support.Utils import separate_debug_print_big, separate_debug_print_small

MODEL_NAME = "random forest"
SUBJECTIVITY_MODEL_NAME = "svm"
TRESHOLD = 0.6
POLARITY_MODEL_FILE = "polarity_model.joblib"
SUBJECTIVITY_MODEL_FILE = "subjectivity_model.joblib"


def calc_avg(param):
    return str(((param[0] + param[1] + param[2] + param[3] + param[4])/5)*100)


def check_values_acc(predictions, filtered_test_set, polarity):
    right = 0
    almost_right = 0
    if polarity is True:
        for real_pol, predict_pol in zip(filtered_test_set[2], predictions[1]):
            if real_pol == predict_pol:
                right += 1
            if real_pol == predict_pol + 1 or real_pol == predict_pol - 1 or real_pol == predict_pol:
                almost_right += 1
        print("Polarity: The real accuracy in this iteration -> " + str(right/len(predictions[1])))
        print("Polarity: The almost real accuracy in this iteration -> " + str(almost_right / len(predictions[1])))
    else:
        for real_subject, predict_subject in zip(filtered_test_set[3], predictions[1]):
            if real_subject == predict_subject:
                right += 1
            if predict_subject !=0 and predict_subject!=1:
                print(predict_subject)
        print("Subjectivity: The real Subjectivity in this iteration -> " + str(right / len(predictions[1])))


def remove_zeros(train_ids, filtered_data_train, polarity_Y, subjectivity_Y, zero_index_list):
    objects_to_convert = list()
    objects_to_convert.extend([train_ids, filtered_data_train, polarity_Y, subjectivity_Y])

    for obj, i in zip(objects_to_convert, range(len(objects_to_convert))):
        if type(obj) is list:
            objects_to_convert[i] = [i for j, i in enumerate(obj) if j not in zero_index_list]
            # for index in zero_index_list:
            #     obj.remove(index)
        elif type(obj) is numpy.ndarray:
            for index in zero_index_list:
                objects_to_convert[i] = numpy.delete(obj, index)
        else:
            print("wrong objects to remove the zeros")

    return train_ids, filtered_data_train, polarity_Y, subjectivity_Y


class Model:
    def __init__(self, stop_words, language='heb'):
        """
        model constructor
        """
        self.model_helper = modelHelperBase()
        self.filtered_train_set = None
        self.filtered_test_set = None
        self.language = language
        if language == 'en':
            self.vectorizer = TfidfVectorizer(max_features=2500, min_df=0.005, max_df=0.9,
                                          stop_words='english', ngram_range=(1, 2))
        else:
            self.vectorizer = TfidfVectorizer(max_features=2500, min_df=0.005, max_df=0.9,
                                          stop_words=stop_words, ngram_range=(1, 2))

    def from_train_to_vector(self, train_set):
        """
        :param train_set: train set sentences
        :param test_set: test set sentences
        :return: a numeric vector representation for the features sentences
        """
        #train_ids, train_X, polarity_Y, subjectivity_Y = separate_data(train_set, self.language)
        train_ids, train_X, polarity_Y, subjectivity_Y = clean_data(train_set)
        filtered_data_train, zero_index_list = \
        self.model_helper.filter_data\
            (train_X, self.vectorizer, train_ids, polarity_Y,
             subjectivity_Y, is_train=True, language=self.language, is_filtered=True)
        return remove_zeros(train_ids, filtered_data_train, polarity_Y, subjectivity_Y, zero_index_list)

    def from_test_to_vector(self, test_set):
        #test_ids, test_X, test_polarity, test_subjectivity = separate_data(test_set, self.language)
        test_ids, test_X, test_polarity, test_subjectivity = clean_data(test_set)
        filtered_data_test, zero_index_list = self.model_helper.filter_data\
            (test_X, self.vectorizer, test_ids, test_polarity,
             test_subjectivity, language=self.language, is_filtered=True)

        return remove_zeros(test_ids, filtered_data_test, test_polarity, test_subjectivity, zero_index_list)

    def run_model(self, model_name, train_set, train_set_labels, test_set, is_loaded = True , polarity=False):
        """
        runs a specific model and gets prediction for test set
        :param model_name: name of current model
        :param train_set_labels: labels type for train set (polarity / subjectivity)
        :return: prediction and confidence of current model
        """
        if is_loaded:
            self.model_helper.create_model(model_name)
        else:
            self.model_helper.load_model(model_name)
        self.model_helper.train_model(model_name, self.filtered_train_set[0], train_set, train_set_labels.astype(int))
        predictions = self.model_helper.test_model(model_name, self.filtered_test_set[0], test_set)
        accuracy = self.model_helper.get_accuracy()
        confidence = self.model_helper.get_confidence(model_name, test_set)
        separate_debug_print_big(title="start iteration")
        print(model_name + ' Results:')
        print(" Cross Validation Accuracy: ", accuracy[1], " Average ->", calc_avg(accuracy[1]), "%")
        if self.filtered_test_set[3] is not None:
            check_values_acc(predictions, self.filtered_test_set, polarity)
            separate_debug_print_big(title="end of iteration")

        return predictions, confidence

    def run(self, train_set, test_set, is_loaded = True):
        """
        runs one model for polarity predictions and the second for subjectivity predictions
        :param train_set: train set for training both models - each one with different labels type
        :param test_set: test set
        :return: predictions and confidence of polarity and subjectivity models
        """
        if train_set is not None:
            self.filtered_train_set = self.from_train_to_vector(train_set)
        self.filtered_test_set = self.from_test_to_vector(test_set)

        # Train and test model for subjectivity results.
        s_predictions, s_confidence = self.run_model(SUBJECTIVITY_MODEL_NAME,
                                                     self.filtered_train_set[1],
                                                     self.filtered_train_set[3],
                                                     self.filtered_test_set[1]
                                                     )

        regressed_train_set = self.get_regressed_features()

        # Train and test model for polarity results.
        p_predictions, p_confidence = self.run_model(MODEL_NAME,
                                                     regressed_train_set,
                                                     self.filtered_train_set[2],
                                                     self.filtered_test_set[1],
                                                     polarity=True)

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

    def save_models(self):
        self.model_helper.save_model(SUBJECTIVITY_MODEL_NAME)
        self.model_helper.save_model(MODEL_NAME)
