#!/usr/bin/env python
# coding: utf-8

# In[47]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Read CSV Heart Failure dataset
heartFailure = pd.read_csv('heart_failure_clinical_records_dataset.csv')

heartFailure['age'].fillna(heartFailure['age'].mean(), inplace=True)
heartFailure['creatinine_phosphokinase'].fillna(heartFailure['creatinine_phosphokinase'].mean(), inplace=True)
heartFailure['ejection_fraction'].fillna(heartFailure['ejection_fraction'].mean(), inplace=True)
heartFailure['platelets'].fillna(heartFailure['platelets'].mean(), inplace=True)
heartFailure['serum_creatinine'].fillna(heartFailure['serum_creatinine'].mean(), inplace=True)
heartFailure['serum_sodium'].fillna(heartFailure['serum_sodium'].mean(), inplace=True)
heartFailure['time'].fillna(heartFailure['time'].mean(), inplace=True)

features = heartFailure.drop(columns = 'DEATH_EVENT', axis = 1)
labels = heartFailure['DEATH_EVENT']


feature_train
feature_test
label_train
label_test


training_results = []
testing_results = []

probability_test = []

predict_train = []

for i in range(0, 50):
    feature_train, feature_test, label_train, label_test = train_test_split(features, labels)

    gnb = GaussianNB()

    gnb.fit(feature_train, label_train)

    # Accuracy score for training set
    training_results += [gnb.score(feature_train, label_train)]

    # Accuracy score for testing set
    testing_results += [gnb.score(feature_test, label_test)]
    
    # Probably for testing set
    probability_test += [gnb.predict_proba(feature_test)]
    
    predict_train += [(label_train != gnb.predict(feature_train)).sum()]

training_accuracy = np.average(np.array(training_results))
testing_accuracy = np.average(np.array(testing_results))

probability_test_average = np.average(np.array(probability_test))

simple_test_predict = np.average(np.array(predict_test))

predict_train_average = np.average(np.array(predict_train))

# REPORT
# Dataset: Heart Failure Clinical Records Dataset
# Link: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
# Average Accuracy (Correct): 78.61%
# Average Probabilty Score (Correct): 50%
# Average Accuracy (Incorrect): 21.39%
# Average Probabilty Score (Inccorect): 50%
# Naive Bayes vs Assignment 2: When comparing the preformance between Naive Bayse and
# my best reported preformance of kNN and Decision Trees; I found that kNN provided
# the best results in terms of accuracy. Naive Bayes might have done worse because
# kNN does better with smaller datasets.
# Comparing Probability Score: The probability score indicates that the dataset
# doesn't have a strong relation with the features and the labels.

probability_test_average


# In[ ]:





# In[ ]:




