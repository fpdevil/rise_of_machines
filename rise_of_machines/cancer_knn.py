"""Cancer data classification

Classifying the Wisconsin cancer data from UCI repository
into benign and malignant classes with k Nearest Neighbors
"""

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# File              : cancer_knn.py
# find whether cancer is malignant or benign using kNN
import time
import warnings

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics, model_selection, neighbors, preprocessing

warnings.filterwarnings('ignore')

# get the initial time
t_init = time.time()

# url for the Wisconsin Breast Cancer data from UCI
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

# set the names of the columns as pulled from the file accompanying the dataset
# which is breast-cancer-wisconsin.names
names = [
    "SampleCodeNumber", "ClumpThickness", "UniformityCellSize",
    "UniformityCellShape", "MarginalAdhesion", "SingleEpithelialCellSize",
    "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "Class"
]
print('[INFO] gathering the {} data'.format(url.split('/')[-2]))

df = pd.read_csv(url, names=names)
print('[INFO] shape of the cancer data {}'.format(df.shape))
print('[INFO] information about the cancer database \n{}'.format(df.info()))
print('[INFO] report of the data at a fine grained level \n{}'.format(
    df.describe()))

# As per  the documentation note of  the cancer dataset, there  are some
# missing attribute values. There are 16  instances in Groups 1 to 6 that
# contain  a single  missing  (i.e., unavailable)  attribute value,  now
# denoted by "?".
missing_counts = df.apply(lambda x: x == '?', axis=1).sum()
null_counts = df.apply(lambda x: x.isnull().values.ravel().sum())
isnull_predicate = df.isnull().values.any()

print('[INFO] report of the missing attribute information \n{}'.format(
    missing_counts))
print('[INFO] BareNuclei attribute information details \n{}'.format(
    df.groupby('BareNuclei').BareNuclei.count()))
print('[INFO] does the dataset has any null values ? {}'.format(
    isnull_predicate))
print(
    '[INFO] null attribute value information if any \n{}'.format(null_counts))

# As per the above result, BareNuclei has 16 values equal to "?" for which
# we may either discard the rows with missing values or replace them with
# the  most   common  or   frequent  values  in   the  dataset   given  by
# df[df.BareNuclei != ‘?’]

# most frequent value of BareNuclei from the table
frequent_value = df['BareNuclei'].value_counts().index[0]
print('[INFO] replacing the ? with most frequent value of {}'.format(
    frequent_value))
df['BareNuclei'] = df['BareNuclei'].replace('?', np.NaN)
df['BareNuclei'] = df['BareNuclei'].fillna(frequent_value)
df['BareNuclei'] = df['BareNuclei'].apply(lambda x: int(x))

# Heatmap of the correlation matrix calculated from pandas with index of
# the nlargest = 10
# nlargest represents the n largest values sorted in decreasing order.
plt.figure(1)
fields = df.corr().nlargest(10, 'Class')['Class'].index
corr = df[fields].corr()
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.4)
plt.title('Heatmap of Cancer Data Correlation Matrix')
plt.show()
# distribute the dataset between training data and target/labels as under
X = df.drop(['SampleCodeNumber', 'Class'], axis=1)
y = df['Class']

# here we are representing class label 2 as 'benign' and 4 as 'malignant'
df.Class.replace([2, 4], ['benign', 'malignant'], inplace=True)
print('[INFO] target class labels for cancer {}'.format(np.unique(y)))
print('[INFO] count of benign and malignant classes \n{}'.format(
    df.Class.value_counts()))

plt.figure(2)
sns.countplot(df['Class'],
              label='Count',
              palette=sns.color_palette("deep", 10))
plt.show()

# as per the  accompanying documentation, the class labels 2  and 4 correspond
# to cancer states, Benign and Malignant as under
# class label = 2 -> Benign
# class label = 4 -> Malignant
# we can  encode the labels  with scikit learn LabelEncoder  though it's
# not needed in this case as it's  usually applied in the cases where the
# target labels are all strings
le = preprocessing.LabelEncoder()
labels = le.fit_transform(df['Class'])
print('[INFO] scikit encoded labels {}'.format(np.unique(labels)))

# get a box plot of all the parameters
plt.figure(3)
df.drop('Class',
        axis=1).plot(kind='box',
                     subplots=True,
                     layout=(4, 3),
                     sharex=False,
                     sharey=False,
                     figsize=(9, 9),
                     title='Box Plot of individual cancer input variables')
plt.show()

# Feature Scaling - Standardization
# As  a part  of optimization  of the  algorithm, we  can apply  feature
# scaling,  by  standardizing  features   using  StandardScaler  class  from
# sklearn's preprocessing module. Scaling  will ensure that the features
# will have a 0 mean and standard  deviation of 1. This helps in all the
# features contributing equally.
scaler = preprocessing.StandardScaler()
print('[INFO] re-scaling the features with options {}'.format(
    scaler.get_params()))
X_std_array = scaler.fit_transform(X.values)
X_std = pd.DataFrame(X_std_array, index=X.index, columns=X.columns)

# now print mean and standard deviation
print("[DEBUG] Dataset Mean:", round(X_std.mean()))
print("[DEBUG] Dataset Standard deviation:", X_std.std())

# For the purpose of checking how well the trained model will perform on
# sample  unseen test  data, we  will  split the  dataset into  separate
# training and testing data sets
# we will split the datasets in 70% - 30% ratio, also the below function
# provides a shuffled data.
(X_train, X_test, y_train,
 y_test) = model_selection.train_test_split(X_std,
                                            y,
                                            test_size=0.3,
                                            random_state=1,
                                            stratify=y)

print('[INFO] now evaluating the kNN classifier with minkowski metric...')
model = neighbors.KNeighborsClassifier(n_neighbors=3,
                                       p=2,
                                       metric='minkowski',
                                       weights='uniform')
# fit the model using training and target labels
model.fit(X_train, y_train)

# test data prediction
y_predicted = model.predict(X_test)
target_names = le.classes_

print(
    "[DEBUG] kNN Training Confusion Matrix\n\n",
    pd.crosstab(y_train,
                model.predict(X_train),
                rownames=["Actual"],
                colnames=["Predicted"]))
print('[DEBUG] Training set Accuracy score: {:0.3%}'.format(
    metrics.accuracy_score(y_train, model.predict(X_train))))
print(
    '[DEBUG] kNN Testing Confusion Matrix\n\n',
    pd.crosstab(y_test,
                model.predict(X_test),
                rownames=['Actual'],
                colnames=['Predicted']))
print('[DEBUG] Testing set Accuracy score: {:0.3%}'.format(
    metrics.accuracy_score(y_test, model.predict(X_test))))

clr = metrics.classification_report(y_test,
                                    y_predicted,
                                    target_names=target_names)
print('[INFO] classification metrics for the model \n{}'.format(clr))

# calculate the running time of the model
run_time = time.time() - t_init
print('[INFO] total time taken for the model: %.3f s' % run_time)
