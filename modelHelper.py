import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class modelHelper:
    def __init__(self, model, k, labels):
        self.modelName = model
        self.model = self.create_model()
        self.k = k
        self.labels = labels
        self.processed_features = None
        self.conf_matrices = []
        self.class_reports = []
        self.accuracy_scores = []

    def create_features(self, features, vectorizer):
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
        processed_features = vectorizer.fit_transform(processed_features).toarray()
        self.processed_features = processed_features

    def create_model(self):
        if self.modelName == 'naive bayes':
            return GaussianNB()
        elif self.modelName == 'svm':
            return SVC(gamma='auto')
        elif self.modelName == 'random forest':
            return RandomForestClassifier(n_estimators=200, random_state=0)
        else:
            raise Exception('unknown model')

    def train_and_test_model(self):
        # creating kfold cross validation sets
        kf = KFold(n_splits=self.k, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(self.processed_features):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.processed_features[train_index], self.processed_features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            self.conf_matrices.append(confusion_matrix(y_test, predictions))
            # precision:tp/(tp+fp) , recall: tp/(tp+fn), f1: 2*((precision*recall)/(precision+recall)),
            # support: # of samples of the true response
            # macro - regular average of each score, weighted: average according to # of samples
            self.class_reports.append(classification_report(y_test, predictions, output_dict=True))
            #  the fraction of correctly classified samples
            self.accuracy_scores.append(accuracy_score(y_test, predictions))

    def get_accuracy(self):
        sum = np.zeros(shape=[3, 3])
        for c in self.conf_matrices:
            sum += c
        conf_mat_av = sum / len(self.conf_matrices)

        clas_reps_sum = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                         'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                         'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
        clas_reps_av = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
        for c in self.class_reports:
            # negativ, neutral, positive
            for k in clas_reps_sum.keys():
                # precision, recall, f1-score, support
                for k1 in clas_reps_sum[k].keys():
                    clas_reps_sum[k][k1] += c[k][k1]
        for k in clas_reps_sum.keys():
            # precision, recall, f1_score, support
            for k1 in clas_reps_sum[k]:
                clas_reps_av[k][k1] = clas_reps_sum[k][k1] / 6

        acc_av = 0
        for a in self.accuracy_scores:
            acc_av += a
        acc_av /= 6
        return conf_mat_av, clas_reps_av, acc_av

