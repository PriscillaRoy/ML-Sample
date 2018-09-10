# AdaBoost Decision Tree Classification
from __future__ import print_function

print(__doc__)
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import requests
import os



# Getting ionosphere data

files_in_folder = os.listdir()
if not 'ionosphere.txt' in files_in_folder:
  url = ('https://archive.ics.uci.edu/ml/machine-learning-databases' +
         '/ionosphere/ionosphere.data')
  response = requests.get(url)
  raw_data = response.text
  with open('ionosphere.txt', 'w') as f:
    f.write(raw_data)
else:
  with open('ionosphere.txt', 'r') as f:
    raw_data = f.read()

rows = raw_data.strip('\n').split('\n')
rows = [row.split(',') for row in rows]
df = pd.DataFrame(rows)
df.rename(columns={df.columns.values[-1]: 'target'}, inplace=True)
df.to_csv('ionosphere_processed.csv')

data = pd.read_csv('ionosphere_processed.csv', index_col=0)
data_mat = data.as_matrix()
n_rows, n_cols = data_mat.shape
c = data_mat[:, :n_cols-1]
d = data_mat[:, n_cols-1]

print("------------------------------------------------------------------------------")
print("DataSet :1 >> IONOSPHERE DATA")
print("------------------------------------------------------------------------------")

# Adaboost Decision Tree Ionosphere
clf1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: DecisionTreeClassifier:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: DecisionTreeClassifier:', accuracy1)

print("******************************************************************************")

clf1 = AdaBoostClassifier(base_estimator=BernoulliNB(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: Naive Bayes:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: Naive Bayes:', accuracy1)

print("******************************************************************************")

clf1 = AdaBoostClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: Radius Neighbors:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: Extra Tree Classifier:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: DecisionTreeClassifier:', accuracy1)
print('Accuracy :: BaggingClassifier :: DecisionTreeClassifier:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: Naive Bayes:', accuracy1)
print('Accuracy :: BaggingClassifier :: Naive Bayes:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: Radius Neighbors:', accuracy1)
print('Accuracy :: BaggingClassifier :: Extra Tree Classifier:', accuracy1)

print("******************************************************************************")


#Getting Housing Data
files_in_folder = os.listdir()
if not 'housing.txt' in files_in_folder:
  url = ('https://archive.ics.uci.edu/ml/machine-learning-databases' +
         '/housing/housing.data')
  response = requests.get(url)
  raw_data = response.text
  with open('housing.txt', 'w') as f:
    f.write(raw_data)
else:
  with open('housing.txt', 'r') as f:
    raw_data = f.read()

rows = raw_data.strip('\n').split('\n')
rows = [row.split(',') for row in rows]
df = pd.DataFrame(rows)
df.rename(columns={df.columns.values[-1]: 'target'}, inplace=True)
df.to_csv('housing_processed.csv')
data_mat = data.as_matrix()
n_rows, n_cols = data_mat.shape
c = data_mat[:, :n_cols-1]
d = data_mat[:, n_cols-1]

print("------------------------------------------------------------------------------")
print("DataSet :2 >> BOSTON HOUSING DATA")
print("------------------------------------------------------------------------------")
# Adaboost Decision Tree Housing Data
clf1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: DecisionTreeClassifier:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: DecisionTreeClassifier:', accuracy1)

print("******************************************************************************")

clf1 = AdaBoostClassifier(base_estimator=BernoulliNB(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: Naive Bayes:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: Naive Bayes:', accuracy1)

print("******************************************************************************")

clf1 = AdaBoostClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: AdaBoostClassifier :: Radius Neighbors:', accuracy1)
print('Accuracy :: AdaBoostClassifier :: Extra Tree Classifier:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: DecisionTreeClassifier:', accuracy1)
print('Accuracy :: BaggingClassifier :: DecisionTreeClassifier:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: Naive Bayes:', accuracy1)
print('Accuracy :: BaggingClassifier :: Naive Bayes:', accuracy1)

print("******************************************************************************")

clf1 = BaggingClassifier(base_estimator=ExtraTreeClassifier(), n_estimators=200)
clf1.fit(c, d)
# Perform K-fold cross validation
for i in range(2, 10, 2):
    kfold = model_selection.KFold(n_splits=i)
    scores1 = model_selection.cross_val_score(clf1, c, d, cv=kfold)
    #print('Cross-validated scores for KFold=:', kfold, scores1)
    predictions1 = model_selection.cross_val_predict(clf1, c, d, cv=kfold)

    accuracy1 = metrics.accuracy_score(d, predictions1)
    #print('Accuracy :: BaggingClassifier :: Radius Neighbors:', accuracy1)
print('Accuracy :: BaggingClassifier :: Extra Tree Classifier:', accuracy1)
print("******************************************************************************")
