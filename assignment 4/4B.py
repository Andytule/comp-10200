#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

# https://archive.ics.uci.edu/ml/datasets/Physical+Unclonable+Functions

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

