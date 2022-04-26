#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

d = pd.read_csv('sgemm_product.csv')
t = d['Run']
d = d.drop(labels='Run', axis=1)

x_train, x_test, y_train, y_test = train_test_split(d, t, test_size=0.3)
rgr = LinearRegression()
rgr.fit(x_train, y_train)
predict = rgr.predict(x_test)

print("Train size: ",x_train.shape[0])
print("Test size: ",x_test.shape[0])
print("Features size: ",x_test.shape[1])

print("Linear")
print("Coefficients:",rgr.coef_)
print("Intercept:",rgr.intercept_)
print("Correlation (r):",np.corrcoef(predict,y_test)[0,1])
print("RSS (e):",((predict-y_test)**2).sum())


print("DecisionTree")
n = DecisionTreeRegressor(criterion='friedman_mse',)
n.fit(x_train, y_train)
predict = n.predict(x_test)
print("RSS (e):",((predict-y_test)**2).sum())


# In[ ]:





# In[ ]:




