#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# loading data into pandas data frame
heart_data = pd.read_csv("heart_disease_data.csv")
#heart_data.head(10)


# In[5]:


# shape of dataset
heart_data.shape


# In[6]:


# describing data
heart_data.describe()


# In[7]:


# dataset information
heart_data.info()


# In[8]:


# checking for missing values 
heart_data.isnull().sum()


# In[9]:


# checking the distribution of target variable
heart_data['target'].value_counts()


# In[10]:


X = heart_data.drop(columns = 'target', axis = 1)
X.head()

# now X contains table without target column which will help for training the dataset


# In[11]:


Y = heart_data['target']
Y.head()

# Y contains one column which includes output for validating the result after model prediction 


# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, stratify = Y, random_state = 3 )


# In[13]:


# checking shape of splitted data
print(X.shape, X_train.shape, X_test.shape)


# In[14]:


model = LogisticRegression()


# In[15]:


# training the LogisticRegression model with training data
model.fit(X_train.values, Y_train)


# In[16]:


# accuracy of traning data
# accuracy function measures accuracy between two values,or columns

X_train_prediction = model.predict(X_train.values)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print("The accuracy of training data : ", training_data_accuracy)


# In[17]:


# prediction graph for training data

plt.figure(figsize=(4,4))
plt.scatter(Y_train, X_train_prediction , color = 'green')
plt.title(" Heart Disease Prediction model fitting ")
plt.show()


# In[18]:


# accuracy of test data

X_test_prediction = model.predict(X_test.values)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("The accuracy of test data : ", test_data_accuracy)


# In[19]:


# input feature values
input_data = (42,1,0,136,315,0,1,125,1,1.8,1,0,1)

# change the input data into numpy array 
input_data_as_numpy_array = np.array(input_data)

# reshape the array to predict data for only one instance
reshaped_array = input_data_as_numpy_array.reshape(1,-1)


# In[20]:


# predicting the result and printing it

prediction = model.predict(reshaped_array)

print(prediction)

if(prediction[0] == 0):
    print("Patient has a healthy heart 💛💛💛💛")

else:
    print("Patient has a unhealthy heart 💔💔💔💔")


# In[21]:


from sklearn import tree


# In[24]:


models = tree.DecisionTreeClassifier()


# In[25]:


models.fit(X_train.values, Y_train)


# In[26]:


# accuracy of traning data
# accuracy function measures accuracy between two values,or columns

X_train_prediction = models.predict(X_train.values)
training_accuracy = accuracy_score(X_train_prediction, Y_train)

print("The accuracy of training data : ", training_accuracy)


# In[28]:


# accuracy of test data

X_test_prediction = models.predict(X_test.values)
test_accuracy = accuracy_score(X_test_prediction, Y_test)

print("The accuracy of test data : ", test_accuracy)


# In[29]:


# predicting the result and printing it

predict = models.predict(reshaped_array)

print(predict)

if(predict[0] == 0):
    print("Patient has a healthy heart 💛💛💛💛")

else:
    print("Patient has a unhealthy heart 💔💔💔💔")







