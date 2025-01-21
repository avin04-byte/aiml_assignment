#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
df = pd.read_csv("Universities.csv")
df


# In[5]:


np.mean(df["SAT"])


# In[9]:


np.median(df["Top10"])


# In[10]:


np.std(df["SFRatio"])


# In[11]:


#Find the variance
np.var(df["SFRatio"])


# In[12]:


df.describe()


# In[ ]:




