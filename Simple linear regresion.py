#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.describe()


# # Obsevations

# 1> No Null values are found in data

# In[5]:


plt.scatter(data1['daily'],data1['sunday'])


# In[6]:


data1['daily'].corr(data1['sunday'])


# # Observations

# 1> Daily and Sunday has high positive correlation strength

# In[7]:


import statsmodels.formula.api as smf
model = smf.ols('sunday~daily',data = data1).fit()
model.summary()


# In[9]:


x = data1['daily'].values
y = data1['sunday'].values
plt.scatter(x, y, color = 'm', marker = 'o', s = 30)
b0 = 13.84
b1 = 1.33
y_hat = b0 + b1*x
plt.plot(x, y_hat, color = 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.show


# In[ ]:




