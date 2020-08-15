import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import pandas as pd
import model_utils as mUtils
#sklearn.confusionMetrics - classificationReport

class modelHelper:
    def __init__(self, model, k, labels, ids):
        self.modelName = model
        self.model = self.create_model()
        self.k = k
        self.labels = labels
        self.processed_features = None
        self.conf_matrices = []
        self.class_reports = []
        self.accuracy_scores = []
        self.pred_and_lab = []
        self.tweet_ids = ids

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
        vectorizer.fit_transform(processed_features).toarray()
        vectorizer_features = vectorizer.get_feature_names()
        vocabulary = mUtils.get_vocabulary()
        vocabulary += vectorizer_features
        vocabulary = list(set(dict.fromkeys(vocabulary)))
        vectorizer.set_params(vocabulary=vocabulary)
        processed_features = vectorizer.fit_transform(processed_features).toarray()
        s = int(len(processed_features)*0.8)

        self.processed_features = processed_features[:s]
        self.test = processed_features[s:]
        self.test_labels = self.labels[s:]
        self.test_ids = self.tweet_ids[s:]
        self.labels = self.labels[:s]
        self.fet_ids = self.tweet_ids[:s]


    def create_model(self):
        if self.modelName == 'naive bayes':
            return GaussianNB()
        elif self.modelName == 'svm':
            return SVC(kernel="linear", class_weight="balanced")
        elif self.modelName == 'random forest':
            # does't stop with 1000, but should be 1000
            return RandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced")
        else:
            raise Exception('unknown model')

    def train_and_test_model(self):
        self.scores = (self.fet_ids, cross_val_score(self.model, self.processed_features, self.labels, cv=5))
        self.model.fit(self.processed_features, self.labels)
        y_predicted = self.model.predict(self.test)
        self.accuracy_score = accuracy_score(self.test_labels, y_predicted)
        self.preds = (self.test_ids, y_predicted, self.test_labels)

    def get_accuracy(self):
        return self.scores

    def get_real_accuracy(self):
        return self.accuracy_score

    def get_predictions(self):
        return self.preds

    '''
    def train_and_test_model(self):
        # creating kfold cross validation sets
        kf = KFold(n_splits=self.k, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(self.processed_features):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.processed_features[train_index], self.processed_features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            train_ids, test_ids = self.tweet_ids[train_index], self.tweet_ids[test_index]
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            self.conf_matrices.append(confusion_matrix(y_test, predictions))
            # precision:tp/(tp+fp) , recall: tp/(tp+fn), f1: 2*((precision*recall)/(precision+recall)),
            # support: # of samples of the true response
            # macro - regular average of each score, weighted: average according to # of samples
            self.class_reports.append(classification_report(y_test, predictions, output_dict=True))
            #  the fraction of correctly classified samples
            self.accuracy_scores.append(accuracy_score(y_test, predictions))
            self.pred_and_lab.append((test_ids, y_test, predictions))



    def get_accuracy(self):
        s = len(self.conf_matrices[0])
        sum = np.zeros(shape=[s, s])
        for c in self.conf_matrices:
            sum += c
        conf_mat_av = sum / len(self.conf_matrices)
        # neg, neut, pos
  
        clas_reps_sum = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                         'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                         'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
        clas_reps_av = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
   
        # -1, 1
        clas_reps_sum = {'-1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                         '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
        clas_reps_av = {'-1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                        '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
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



    def pred_vs_ytest_comp(self):
        res = pd.DataFrame(columns=['tweet', 'label', 'prediction', 'correct_prediction'])
        tweets = []
        labels = []
        preds = []
        for tup in self.pred_and_lab:
            tweets += list(tup[0])
            labels += list(tup[1])
            preds += list(tup[2])
        res.tweet = tweets
        res.label = labels
        res.prediction = preds
        res.correct_prediction = res.label == res.prediction
        return res
    '''
    # need to create a function that creates a table that each row is a tweet, and each column is a model,
    # 1 being the model was correct for that tweet, 0 it was incorrect
