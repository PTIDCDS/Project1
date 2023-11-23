#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[3]:


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# ### Check the shape of the data

# In[4]:


x_train.shape


# In[5]:


y_train.shape


# In[6]:


x_test.shape


# In[7]:


y_test.shape


# ## Preprocessing

# ### Visualization of the data

# In[16]:


plt.imshow(x_train[178],cmap=plt.get_cmap('gray'))


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation ="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation ="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10,activation ="softmax")
])


# In[ ]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(100),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10,activation ="softmax")
])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




