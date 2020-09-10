import re

from pandas import np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from joblib import dump, load

TRESHHOLD = 0.5
RANDOM_FOREST_FILE = "C:\\SentimentAnalysisProject\Models\Data\\polarity_model.joblib"
SVM_FILE = "C:\\SentimentAnalysisProject\Models\Data\\subjectivity_model.joblib"


class modelHelperBase:
    def __init__(self):
        """
        constructor
        """
        self.conf_matrices = []
        self.class_reports = []
        self.accuracy_scores = []
        self.pred_and_lab = []
        self.models = {}

    def filter_data(self, features, vectorizer, is_train = False):
        """
        filter redundant tokens and returns each feature as a vector v witch represents
        the words the feature contains
        :param features: train set
        :param test_X: test set
        :param vectorizer: tfidfvectorizer
        :return: filtered and vectorized train and test sets
        """
        processed_features = []
        for sentence in features:
            # Remove all words with @ characters
            processed_feature = re.sub(r'[@|_][a-zA-Z]+', ' ', str(sentence))

            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', processed_feature)

            # remove all single characters
            processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

            # Remove single characters from the start
            processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

            # Substituting multiple spaces with single space
            processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)

            # Converting to Lowercase
            processed_feature = processed_feature.lower()

            processed_features.append(processed_feature)
        if is_train:
            results = vectorizer.fit_transform(processed_features).toarray()
        else:
            results = vectorizer.transform(processed_features).toarray()

        # this is part for checking the zeros array list
        all_zero_array_list = list()
        for i_array, index in zip(results, range(len(results))):
            all_zero = True
            for nunber in i_array:
                if nunber != 0:
                    all_zero = False
                    break
            if all_zero:
                all_zero_array_list.append(index)

        return results, all_zero_array_list

    def create_model(self, modelName):
        """
        creates a model by its name and stores in "models" dictionary
        :param modelName: name of wanted model
        :return: the created model
        """
        if modelName in self.models:
            return
        else:
            if modelName == 'naive bayes':
                model = GaussianNB()
            elif modelName == 'svm':
                model = SVC(kernel="linear", class_weight="balanced", probability=True)
            elif modelName == 'random forest':
                model = RandomForestClassifier(n_estimators=1000, random_state=0, class_weight="balanced")
            else:
                raise Exception('unknown model')
        self.models[modelName] = model

    def train_model(self, model_name, fet_ids, train_set, labels):
        """
        trains model on train set, and executing cross validation for validating results
        :param model_name: name of model
        :param fet_ids: features ids
        :param train_set: train set
        :param labels: train set labels
        """
        model = self.models[model_name]
        if model is None:
            raise Exception('unknown model')
        self.scores = (fet_ids, cross_val_score(model, train_set, labels, cv=5))
        model.fit(train_set, labels)

    def test_model(self, model_name, test_ids, test_set):
        """
        test model and get predictions
        :param model_name: model name
        :param test_ids: test set ids
        :param test_set: test set
        :return: predictions for test set
        """
        model = self.models[model_name]
        if model is None:
            raise Exception('unknown model')
        return test_ids, model.predict(test_set)

    def get_accuracy(self):
        """
        :return: accuracy based on cross validation
        """
        return self.scores

    def get_confidence(self, model_name, data):
        """
        :param model_name: name of model
        :param data: test set - to get confidence for it
        :return: confidence for predictions
        """
        return self.models[model_name].predict_proba(data)

    def get_params(self, model_name):
        model = SelectFromModel(self.models[model_name], prefit=True)
        return model

    def resize_data(self, model, data):
        return model.transform(data)

    def get_bad_indices(self, model_name):
        importance = np.abs(self.get_features_importances(model_name))
        treshold = np.mean(importance) * TRESHHOLD
        bad_indices = np.where(importance < treshold)
        return bad_indices

    def get_features_importances(self, model_name):
        if model_name == 'random forest':
            return self.models[model_name].feature_importances_
        elif model_name == 'svm':
            return self.models[model_name].coef_.flatten()
        else:
            raise Exception('unknown model')

    def save_model(self, model_name):
        model = self.models[model_name]
        if model is None:
            raise Exception('unknown model')
        file_name = self.find_file_name_by_model_name(model_name)
        dump(model, file_name)

    def load_model(self, model_name):
        file_name = self.find_file_name_by_model_name(model_name)
        try:
            model = load(file_name)
            self.models[model_name] = model
        except:
            raise Exception('unknown model')
        return model

    def find_file_name_by_model_name(self, model_name):
        file_name = ""
        if model_name == "svm":
            file_name = SVM_FILE
        elif model_name == "random forest":
            file_name = RANDOM_FOREST_FILE
        return file_name