#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU


# In[2]:


# Read Data
train_data = pd.read_csv("../data/train_data.csv")
train_set = train_data.iloc[:,2:].values


# In[3]:


# Normalize data - makes more sense than standardization
sc = MinMaxScaler()
train_set_sc = sc.fit_transform(train_set)
num_samples = train_set_sc.shape[0]
num_features = train_set_sc.shape[1]


# In[4]:


# Formulate the time based data
timesteps = 5 # Hyperparameter
X_train = []
y_train = []
for i in range(timesteps, num_samples-1):
    X_train.append(train_set_sc[i-timesteps:i])
    y_train.append(train_set_sc[i+1, 6])
X_train, y_train  = np.array(X_train), np.array(y_train)
print (X_train.shape)


# In[5]:


regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, num_features)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units=100, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=1000, batch_size=32)


# In[11]:


# Load the Test data
test_data = pd.read_csv("../data/test_data.csv")
test_set = test_data.iloc[:,8].values

full_data = pd.concat((train_data, test_data), axis=0)
inputs = full_data[len(full_data) - len(test_data) - timesteps:].iloc[:, 2:].values


# In[12]:


# Create test set and call the predict function
inputs = sc.transform(inputs)
num_test_samples = inputs.shape[0]
X_test = []
for i in range(timesteps, num_test_samples):
    X_test.append(inputs[i-timesteps:i])
X_test = np.array(X_test)

y_pred = regressor.predict(X_test)


# In[13]:


# Un-scale the outputs
xmin = sc.data_min_[6]
xmax = sc.data_max_[6]
for i in range(y_pred.shape[0]):
    y_pred[i][0] = y_pred[i][0] * (xmax-xmin) + xmin


# In[14]:


print (test_set)
print (y_pred)


# In[15]:


# Plot actual vs predicted
plt.plot(test_set, color='red', label='Actual Exchange Rate')
plt.plot(y_pred, color='blue', label='Predicted Exchange Rate')
plt.title("USD/INR Exchange Rate prediction")
plt.xlabel("Time")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()


# In[ ]:




