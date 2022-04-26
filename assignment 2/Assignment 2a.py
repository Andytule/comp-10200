#!/usr/bin/env python
# coding: utf-8

# In[68]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import csv
import pandas as pd
import numpy as np
import math 
import random

heartFailure = pd.read_csv('heart_failure_clinical_records_dataset.csv')

def euclideanDistance(x, y):
    return math.sqrt(((x['age'] - y['age']) ** 2) + ((x['creatinine_phosphokinase'] - y['creatinine_phosphokinase']) ** 2))

def manhattan(x, y):
    return abs(x['age'] - y['age']) + abs(x['creatinine_phosphokinase'] - y['creatinine_phosphokinase'])
    
def sortData(x, training, correct, k, d):
    if (d == 0):
        distances = training.apply(lambda y: euclideanDistance(x, y), axis=1).sort_values(ascending=True).head(k).index
    elif (d == 1):
        distances  = training.apply(lambda y: manhattan(x, y), axis=1).sort_values(ascending=True).head(k).index
    death = 0
    alive = 0
    guess = 1 
    for i in distances:
        if (heartTrainingData.loc[i]['DEATH_EVENT'] == 0):
            death += 1
        else:
            alive += 1
    if (death > alive):
        guess = 0
    
    if (int(x['DEATH_EVENT']) == guess):
        correct += 1
    
    return correct

print("\nDifferent K\n")

# K = 13
accuracyA = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctA = 0
    for index, item in heartTestingData.iterrows():
        correctA = sortData(item, heartTrainingData, correctA, 13, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyA.append(correctA / heartTestingData.shape[0] * 100)
    else:
        accuracyA.append(0.0)
print(pd.Series(accuracyA).mean(), accuracyA)
    
K = 27
accuracyB = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctB = 0
    for index, item in heartTestingData.iterrows():
        correctB = sortData(item, heartTrainingData, correctB, 27, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyB.append(correctB / heartTestingData.shape[0] * 100)
    else:
        accuracyB.append(0.0)
print(pd.Series(accuracyB).mean(), accuracyB)


# K = 7
accuracyC = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctC = 0
    for index, item in heartTestingData.iterrows():
        correctC = sortData(item, heartTrainingData, correctC, 7, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyC.append(correctC / heartTestingData.shape[0] * 100)
    else:
        accuracyC.append(0.0)
print(pd.Series(accuracyC).mean(), accuracyC)

print("\nDifferent K and Different Distance\n")
# DIFFERENT DISTANCE

# K = 13
accuracyA = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctA = 0
    for index, item in heartTestingData.iterrows():
        correctA = sortData(item, heartTrainingData, correctA, 13, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyA.append(correctA / heartTestingData.shape[0] * 100)
    else:
        accuracyA.append(0.0)
print(pd.Series(accuracyA).mean(), accuracyA)
    
K = 27
accuracyB = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctB = 0
    for index, item in heartTestingData.iterrows():
        correctB = sortData(item, heartTrainingData, correctB, 27, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyB.append(correctB / heartTestingData.shape[0] * 100)
    else:
        accuracyB.append(0.0)
print(pd.Series(accuracyB).mean(), accuracyB)


# K = 7
accuracyC = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctC = 0
    for index, item in heartTestingData.iterrows():
        correctC = sortData(item, heartTrainingData, correctC, 7, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyC.append(correctC / heartTestingData.shape[0] * 100)
    else:
        accuracyC.append(0.0)
print(pd.Series(accuracyC).mean(), accuracyC)


print("\nDifferent K and Euclidean Distance and Normalized Data\n")


# NORMALIZE ###############################

heartFailure = (heartFailure - heartFailure.min()) / (heartFailure.max() - heartFailure.min())

# K = 13
accuracyA = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctA = 0
    for index, item in heartTestingData.iterrows():
        correctA = sortData(item, heartTrainingData, correctA, 13, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyA.append(correctA / heartTestingData.shape[0] * 100)
    else:
        accuracyA.append(0.0)
print(pd.Series(accuracyA).mean(), accuracyA)
    
K = 27
accuracyB = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctB = 0
    for index, item in heartTestingData.iterrows():
        correctB = sortData(item, heartTrainingData, correctB, 27, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyB.append(correctB / heartTestingData.shape[0] * 100)
    else:
        accuracyB.append(0.0)
print(pd.Series(accuracyB).mean(), accuracyB)


# K = 7
accuracyC = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctC = 0
    for index, item in heartTestingData.iterrows():
        correctC = sortData(item, heartTrainingData, correctC, 7, 0)
    if (heartTestingData.shape[0] > 0):
        accuracyC.append(correctC / heartTestingData.shape[0] * 100)
    else:
        accuracyC.append(0.0)
print(pd.Series(accuracyC).mean(), accuracyC)


print("\nDifferent K and Manhattan Distance and Normalized Data\n")
# DIFFERENT DISTANCE

# K = 13
accuracyA = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctA = 0
    for index, item in heartTestingData.iterrows():
        correctA = sortData(item, heartTrainingData, correctA, 13, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyA.append(correctA / heartTestingData.shape[0] * 100)
    else:
        accuracyA.append(0.0)
print(pd.Series(accuracyA).mean(), accuracyA)
    
K = 27
accuracyB = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctB = 0
    for index, item in heartTestingData.iterrows():
        correctB = sortData(item, heartTrainingData, correctB, 27, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyB.append(correctB / heartTestingData.shape[0] * 100)
    else:
        accuracyB.append(0.0)
print(pd.Series(accuracyB).mean(), accuracyB)


# K = 7
accuracyC = []
for i in range(0, 5):
    split = random.uniform(0, len(heartFailure))
    heartFailure = heartFailure.sample(frac=1).reset_index(drop=True)
    heartTrainingData = heartFailure.loc[0:split, ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    heartTestingData = heartFailure.loc[split: , ('age', 'creatinine_phosphokinase', 'DEATH_EVENT')]
    correctC = 0
    for index, item in heartTestingData.iterrows():
        correctC = sortData(item, heartTrainingData, correctC, 7, 1)
    if (heartTestingData.shape[0] > 0):
        accuracyC.append(correctC / heartTestingData.shape[0] * 100)
    else:
        accuracyC.append(0.0)
print(pd.Series(accuracyC).mean(), accuracyC)




# In[ ]:





# In[ ]:




