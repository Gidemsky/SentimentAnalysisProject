from Models.modelHelperBase import *
import Models.model_utils as mUtils
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pandas as pd

from support.Utils import separate_debug_print_big, separate_debug_print_small

MODEL_NAME = "random forest"
SUBJECTIVITY_MODEL_NAME = "svm"
TRESHOLD = 0.7
POLARITY_MODEL_FILE = "polarity_model.joblib"
SUBJECTIVITY_MODEL_FILE = "subjectivity_model.joblib"


class Model:
    def __init__(self, stop_words, language='heb'):
        """
        model constructor
        """
        self.model_helper = modelHelperBase()
        self.filtered_train_set = None
        self.filtered_test_set = None
        self.language = language
        if language == 'english':
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=2500, min_df=0.005, max_df=0.9,
                                          stop_words='english', ngram_range=(1, 2))
        else:
            self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_features=2500, min_df=0.005, max_df=0.9,
                                         ngram_range=(1, 2), strip_accents='unicode')

    def from_train_to_vector(self, train_set, stemmed = False):
        """
        :param train_set: train set sentences
        :param test_set: test set sentences
        :return: a numeric vector representation for the features sentences
        """
        if stemmed:
            train_ids, train_X, polarity_Y, subjectivity_Y = mUtils.clean_data(train_set)
        else:
            train_ids, train_X, polarity_Y, subjectivity_Y = mUtils.separate_data(train_set, self.language)
        print("adding train dictionary features")
        dic_feature = pd.Series.to_numpy(mUtils.add_dictionary_feature(train_ids, train_X, polarity_Y, self.language))
        filtered_data_train, zero_index_list = \
        self.model_helper.filter_data\
            (train_X, self.vectorizer, train_ids, polarity_Y,
             subjectivity_Y, is_train=True, language=self.language, is_filtered=stemmed)
        filtered_data_train = np.append(filtered_data_train, dic_feature[:, None], axis=1)

        return mUtils.remove_zeros(train_ids, filtered_data_train, polarity_Y, subjectivity_Y, zero_index_list)

    def from_test_to_vector(self, test_set, stemmed=False):
        if stemmed:
            test_ids, test_X, test_polarity, test_subjectivity = mUtils.clean_data(test_set)
        else:
            test_ids, test_X, test_polarity, test_subjectivity = mUtils.separate_data(test_set, self.language)
        print("adding test dictionary features")
        dic_feature = pd.Series.to_numpy(mUtils.add_dictionary_feature(test_ids, test_X,
                                         np.empty(test_X.__len__()), lan=self.language))
        filtered_data_test, zero_index_list = self.model_helper.filter_data\
            (test_X, self.vectorizer, test_ids, test_polarity,
             test_subjectivity, language=self.language, is_filtered=stemmed, is_train=False)
        filtered_data_test = np.append(filtered_data_test, dic_feature[:, None], axis=1)

        return mUtils.remove_zeros(test_ids, filtered_data_test, test_polarity, test_subjectivity, zero_index_list)

    def run_model(self, model_name, train_set, train_set_labels, test_set, is_loaded, polarity=False):
        """
        runs a specific model and gets prediction for test set
        :param model_name: name of current model
        :param train_set_labels: labels type for train set (polarity / subjectivity)
        :return: prediction and confidence of current model
        """
        if not is_loaded:
            self.model_helper.create_model(model_name)
        else:
            self.model_helper.load_model(model_name)
        self.model_helper.train_model(model_name, self.filtered_train_set[0], train_set, train_set_labels)
        predictions = self.model_helper.test_model(model_name, self.filtered_test_set[0], test_set)
        accuracy = self.model_helper.get_accuracy()
        confidence = self.model_helper.get_confidence(model_name, test_set)
        separate_debug_print_big(title="start iteration")
        print(model_name + ' Results:')
        print(" Cross Validation Accuracy: ", accuracy[1], " Average ->", mUtils.calc_avg(accuracy[1]), "%")
        if self.filtered_test_set[3] is not None:
            mUtils.check_values_acc(predictions, self.filtered_test_set, polarity)
        separate_debug_print_big(title="end of iteration")

        return predictions, confidence

    def run(self, train_set, test_set, is_loaded, stemmed = False):
        """
        runs one model for polarity predictions and the second for subjectivity predictions
        :param train_set: train set for training both models - each one with different labels type
        :param test_set: test set
        :return: predictions and confidence of polarity and subjectivity models
        """
        if train_set is not None:
            self.filtered_train_set = self.from_train_to_vector(train_set, stemmed)
        self.filtered_test_set = self.from_test_to_vector(test_set, stemmed)
        print("vector created!")
        # Train and test model for subjectivity results.
        print("running the subjectivity model")
        s_predictions, s_confidence = self.run_model(SUBJECTIVITY_MODEL_NAME,
                                                     self.filtered_train_set[1],
                                                     self.filtered_train_set[3],
                                                     self.filtered_test_set[1],
                                                     is_loaded
                                                     )

        regressed_train_set = self.get_regressed_features()

        # Train and test model for polarity results.
        print("running the polarity model")
        p_predictions, p_confidence = self.run_model(MODEL_NAME,
                                                     regressed_train_set,
                                                     self.filtered_train_set[2],
                                                     self.filtered_test_set[1],
                                                     is_loaded,
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
        test_df = pd.DataFrame(self.filtered_test_set[1])
        test_df[bad_indices] = test_df[bad_indices] * TRESHOLD
        return train_df

    def save_models(self):
        self.model_helper.save_model(SUBJECTIVITY_MODEL_NAME)
        self.model_helper.save_model(MODEL_NAME)
