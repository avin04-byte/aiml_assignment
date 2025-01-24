#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("data_clean.csv")
print(data)


# In[2]:


data.info()


# In[3]:


#Data frame attributes
print(type(data))
print(data.size)
print(data.shape)


# In[4]:


#Drop dupplicate
data1 = data.drop(['Unnamed: 0',"Temp C"], axis = 1)
data1


# In[6]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[11]:


#checking for Duplicate rows
data1[data1.duplicated(keep = False)]


# In[9]:


data1[data1.duplicated()]


# In[12]:


#Drop duplcated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[ ]:




