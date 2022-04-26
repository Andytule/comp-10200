#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import csv
import numpy as np
from matplotlib import pyplot as plt

# Removes scientific notation
np.set_printoptions(suppress=True)

with open('heart_failure_clinical_records_dataset.csv') as file:
    reader = csv.reader(file)
    
    ## Feature Name Array
    feature_names = np.array(next(reader))
    
    ## Label and Data Array
    labels = []
    data = []
    
    ## Get data with reader
    for row in reader:
        labels.append(float(row[-1]))
        data.append([float(x) for x in row[:-1]])

    ## Create Numpy Arrays for Label and Data
    labels = np.array(labels)
    data = np.array(data)
       
    ## Randomly Shuffle the arrays
    i = np.arange( labels.shape[0] )
    np.random.shuffle(i)
    labels = labels[i]
    data = data[i]
    
    ## Split the data to Training and Test
    test_amount = int(len(data) * 0.75)
    
    train_labels = labels[:test_amount]
    train_data = data[:test_amount]
    
    test_labels = labels[test_amount:]
    test_data = data[test_amount:]
    
    
    print("Summary of the Heart failure clinical records Data Set")
    print("Andy Le, COMP 10200, 2022")
    
    
    ## Training Set output
    print("\n\nTRAINING SET")
    
    ## Print Training Names
    print("\nNames: " + str(feature_names))
    
    ## Print Training Minimal 
    print("\nMinima: " + str(train_data.min(axis=0)))
    
    ## Print Training Maxima 
    print("\nMaxima: " + str(train_data.max(axis=0)))
    
    ## Print Training Mean 
    print("\nMeans: " + str(train_data.mean(axis=0)))
    
    ## Print Training Mean 
    print("\nMedians: " + str(np.median(train_data, axis=0)))
    
    
    ## Testing Set output
    print("\n\nTESTING SET")
    
    ## Print Training Names
    print("\nNames: " + str(feature_names))
    
    ## Print Training Minimal 
    print("\nMinima: " + str(test_data.min(axis=0)))
    
    ## Print Training Maxima 
    print("\nMaxima: " + str(test_data.max(axis=0)))
    
    ## Print Training Mean 
    print("\nMeans: " + str(test_data.mean(axis=0)))
    
    ## Print Training Mean 
    print("\nMedians: " + str(np.median(test_data, axis=0)))
    
    train_data_alive = train_data[train_labels == 0]
    train_data_dead = train_data[train_labels == 1]
    
    # Age and Serum Sodium
    plt.figure(1)
    plt.title("Age VS Serum Sodium")
    plt.xlabel("Age")
    plt.ylabel("Serum Sodium Levels")
    plt.scatter(train_data_alive[:,0], train_data_alive[:,8], c="g", marker=".")
    plt.scatter(train_data_dead[:,0], train_data_dead[:,8], c="r", marker=".")
    
    # CPK and Ejection Fraction
    plt.figure(2)
    plt.title("CPK VS Ejection Fraction")
    plt.xlabel("Creatinine Phosphokinase Levels")
    plt.ylabel("Ejection Fraction Percentage")
    plt.scatter(train_data_alive[:,2], train_data_alive[:,4], c="g", marker=".")
    plt.scatter(train_data_dead[:,2], train_data_dead[:,4], c="r", marker=".")
    
    # Platelets and Serum Creatinine
    plt.figure(3)
    plt.title("Platelets VS Serum Creatinine")
    plt.xlabel("Platelets Amount")
    plt.ylabel("Serum Creatinine Level")
    plt.scatter(train_data_alive[:,6], train_data_alive[:,7], c="g", marker=".")
    plt.scatter(train_data_dead[:,6], train_data_dead[:,7], c="r", marker=".")
    
    # Bar Graph
    plt.figure(4)
    plt.title("Dead and Alive")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.bar(np.array(["Dead", "Alive"]), np.array([len(train_data_dead), len(train_data_alive)]), color=['r', 'g'])
    
    # Show the plot window
    plt.show()


# In[ ]:





# In[ ]:




