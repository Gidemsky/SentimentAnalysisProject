# svm
import sklearn
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

plt.show()
KFOLD_S = 5


data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
print(airline_tweets.head())

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
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

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

conf_matrices = []
class_reports = []
accuracy_scores = []
# creating kfold cross validation sets
clf = SVC(gamma='auto')
kf = KFold(n_splits=KFOLD_S, random_state=None, shuffle=False)
for train_index, test_index in kf.split(processed_features):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = processed_features[train_index], processed_features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    conf_matrices.append(confusion_matrix(y_test, predictions))
    # precision:tp/(tp+fp) , recall: tp/(tp+fn), f1: 2*((precision*recall)/(precision+recall)),
    # support: # of samples of the true response
    # macro - regular average of each score, weighted: average according to # of samples
    class_reports.append(classification_report(y_test, predictions, output_dict=True))
    #  the fraction of correctly classified samples
    accuracy_scores.append(accuracy_score(y_test, predictions))

sum = np.zeros(shape=[3, 3])
for c in conf_matrices:
    sum += c
conf__mat_av = sum / len(conf_matrices)

clas_reps_sum = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                 'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                 'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
clas_reps_av = {'negative': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                'neutral': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                'positive': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
for c in class_reports:
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
for a in accuracy_scores:
    acc_av += a
acc_av /= 6

print('svm results:')
print(conf__mat_av)
print(clas_reps_av)
print(acc_av)

a = 1