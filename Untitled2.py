#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
cars = pd.read_csv('Toyoto_Corrola.csv')
cars.head()


# In[4]:


cars['Cylinders'].value_counts()


# In[5]:


cars.info()


# In[6]:


cars.describe()


# In[7]:


cars['Gears'].value_counts()


# In[ ]:





# In[11]:


cars = cars.drop(columns=['Model','Id','Cylinders'])
cars


# In[12]:


cars.corr()


# In[13]:


cars.rename(columns = {'Age_08_04': 'Age'}, inplace = True)
cars


# In[14]:


cars.corr()


# In[ ]:




