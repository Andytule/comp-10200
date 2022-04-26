#!/usr/bin/env python
# coding: utf-8

# In[84]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import  accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

label_encoder = LabelEncoder()

cnb_model = ComplementNB()
mnb_model = MultinomialNB()

c_vectorizer1 = CountVectorizer()
c_vectorizer2 = CountVectorizer(ngram_range=(2,2))
c_vectorizer3 = CountVectorizer(stop_words='english')
c_vectorizer4 = CountVectorizer(ngram_range=(2,2), stop_words='english')

# Read Spam Message Data
spam_messages = pd.read_csv('SPAM text message 20170820 - Data.csv')
spam_messages.dropna(inplace = True)

spam_messages['Category'] = label_encoder.fit_transform(spam_messages['Category'])

# Split Data
features = spam_messages['Message']
labels = spam_messages['Category']
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state = 30)

print("MULTINOMIAL\n")

# Vectorizer Default & Multinomial
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer1), ('classifier', mnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("MNB - Vectorizer 1", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer ngram(2,2) & Multinomial
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer2), ('classifier', mnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("MNB - Vectorizer 2", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer stop_words(english) & Multinomial
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer3), ('classifier', mnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("MNB - Vectorizer 3", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer ngram(2,2) + stop_words(english) & Multinomial
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer4), ('classifier', mnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("MNB - Vectorizer 4", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

print("\nCOMPLEMENT\n")

# Vectorizer Default & Complement
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer1), ('classifier', cnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("CNB - Vectorizer 1", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer ngram(2,2) & Complement
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer2), ('classifier', cnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("CNB - Vectorizer 2", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer stop_words(english) & Complement
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer3), ('classifier', cnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("CNB - Vectorizer 3", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

# Vectorizer ngram(2,2) + stop_words(english) & Complement
v_mnb_model = Pipeline(steps = [('vectorizer', c_vectorizer4), ('classifier', cnb_model)])
v_mnb_model.fit(feature_train, label_train)
v_mnb_model.score(feature_test, label_test)
predictions = v_mnb_model.predict(feature_test)
print("CNB - Vectorizer 4", 
      "Accuracy:", round(accuracy_score(label_test, predictions), 2),
      "Precision:", round(precision_score(label_test, predictions, average='macro'), 2),
      "Recall:", round(recall_score(label_test, predictions, average='macro'), 2)
     )

spam_messages.loc[spam_messages.Category == 0].size


# In[ ]:





# In[ ]:




