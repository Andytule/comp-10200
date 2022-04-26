#!/usr/bin/env python
# coding: utf-8

# In[13]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import csv

d_1 = []
d_2 = []
d_3 = []
d_4 = []

targets_1 = []
targets_2 = []
targets_3 = []
targets_4 = []

with open('805099_1.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        d_1.append(row[0:-2])
        targets_1.append(row[-1])

with open('805099_2.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        d_2.append(row[0:-2])
        targets_2.append(row[-1])
        
with open('805099_3.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        d_3.append(row[0:-2])
        targets_3.append(row[-1])
        
with open('805099_4.csv') as f:
    reader=csv.reader(f)
    for row in reader:
        d_4.append(row[0:-2])
        targets_4.append(row[-1])

train_data_1, test_data_1, train_target_1, test_target_1 = train_test_split(d_1, targets_1, test_size=0.25)
train_data_2, test_data_2, train_target_2, test_target_2 = train_test_split(d_2, targets_2, test_size=0.25)
train_data_3, test_data_3, train_target_3, test_target_3 = train_test_split(d_3, targets_3, test_size=0.25)
train_data_4, test_data_4, train_target_4, test_target_4 = train_test_split(d_4, targets_4, test_size=0.25)

one = ['805099_1.csv', train_data_1, test_data_1, train_target_1, test_target_1]
two = ['805099_2.csv', train_data_2, test_data_2, train_target_2, test_target_2]
three = ['805099_3.csv', train_data_3, test_data_3, train_target_3, test_target_3]
four = ['805099_4.csv', train_data_4, test_data_4, train_target_4, test_target_4]

myData = [one, two, three, four]

########################### 8005099_1 ###########################

for data in myData:  
    w_1 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    t_1 = float(0)
    l = float(0.001)
    max_A_1 = 0
    bad_1 = 0

    while bad_1 < 100 and max_A_1 < 100:
        v_1 = 0
        a_1 = 0
        for i, row in enumerate(data[1]):  
            o_1 = 0
            for j, info in enumerate(row):
                o_1 += float(info) * float(w_1[j])
            if o_1 > t_1:
                o_1 = 1
            else:
                o_1 = 0
            if o_1 < float(data[3][i]):
                for k, value in enumerate(row):
                    w_1[k] += float(value) * l
                t_1 -= l
            elif o_1 > float(data[3][i]):
                for k, value in enumerate(row):
                    w_1[k] -= float(value) * l
                t_1 += l
            elif o_1 == float(data[3][i]):
                v_1 += 1
        a_1 = v_1 / len(data[1]) * 100
        if a_1 > max_A_1:
            max_A_1 = a_1
            bad_1 = 0
        else:
            bad_1 += 1
    v_1 = 0       
    for i, row in enumerate(data[2]):  
            o_1 = 0
            for j, info in enumerate(row):
                o_1 += float(info) * float(w_1[j])
            if o_1 > t_1:
                o_1 = 1
            else:
                o_1 = 0
            if o_1 == float(data[4][i]):
                v_1 += 1
    a_1 = v_1 / len(data[2]) * 100
    print(data[0], ":", round(a_1,2), "W:", w_1, "T:", round(t_1,2),"\n")



# In[ ]:




