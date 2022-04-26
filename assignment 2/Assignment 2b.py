#!/usr/bin/env python
# coding: utf-8

# In[20]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random

heartFailure = pd.read_csv('heart_failure_clinical_records_dataset.csv')

attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=2, min_samples_leaf=3, min_impurity_decrease=0.5)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

 
print('\nCHANGE\n')


####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=2, min_samples_leaf=3, min_impurity_decrease=2.0)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=2, min_samples_leaf=10, min_impurity_decrease=0.5)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=2, min_samples_leaf=10, min_impurity_decrease=2.0)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=3, min_impurity_decrease=0.5)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=3, min_impurity_decrease=2.0)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=10, min_impurity_decrease=0.5)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')

####################################
attemps = []

for i in range(50):
    dataSplit = random.uniform(0.2, 0.4)
    heartTrainData, heartTestData = train_test_split(heartFailure, test_size = dataSplit)

    testTarget = heartTestData.pop('DEATH_EVENT').values
    trainTarget = heartTrainData.pop('DEATH_EVENT').values

    classifier = tree.DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=10, min_impurity_decrease=2.0)
    classifier.fit(heartTrainData.values, trainTarget)

    results = classifier.predict(heartTestData.values)
    correct = (results == testTarget).sum()

    attemps += [correct / results.shape[0] * 100]

print(pd.Series(attemps).mean(), pd.DataFrame(attemps).head(5))

print('\nCHANGE\n')
    


# In[ ]:




