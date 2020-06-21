import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import model_utils as mUtils


class modelHelperBase:
    def __init__(self):
        self.conf_matrices = []
        self.class_reports = []
        self.accuracy_scores = []
        self.pred_and_lab = []
        self.models = {}

    """
    filter redundant tokens and returns each feature as a vector v witch represents 
    the words the feature contains.
    """
    def filter_data(self, features, vectorizer):
        processed_features = []
        for sentence in range(0, len(features)):
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

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
        return self.fit_transform(vectorizer, processed_features)

    def fit_transform(self, vectorizer, processed_features):
        # if isVocab:
        #     vectorizer.fit_transform(processed_features).toarray()
        #     vectorizer_features = vectorizer.get_feature_names()
        #     vocabulary = mUtils.get_vocabulary()
        #     vocabulary += vectorizer_features
        #     vocabulary = list(set(dict.fromkeys(vocabulary)))
        #     vectorizer.set_params(vocabulary=vocabulary)
        return vectorizer.fit_transform(processed_features).toarray()

    def create_model(self, modelName):
        if modelName == 'naive bayes':
            model = GaussianNB()
        elif modelName == 'svm':
            model = SVC(kernel="linear", class_weight="balanced")
        elif modelName == 'random forest':
            # does't stop with 1000, but should be 1000
            model = RandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced")
        else:
            raise Exception('unknown model')
        self.models[modelName] = model

    def train_model(self, model_name, fet_ids, train_set, labels):
        model = self.models[model_name]
        if model is None:
            raise Exception('unknown model')
        self.scores = (fet_ids, cross_val_score(model, train_set, labels, cv=5))
        model.fit(train_set, labels)

    def test_model(self, model_name, test_ids, test_set):
        model = self.models[model_name]
        if model is None:
            raise Exception('unknown model')
        return test_ids, model.predict(test_set)

    def get_accuracy(self):
        return self.scores
