#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv("universities.csv")
df


# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df = pd.read_csv("universities.csv")
df


# In[ ]:


#USE GROUP() TO FIND AGGREGATED VALUES 


# In[7]:


sal = pd.read_csv("salaries.csv")
sal


# In[13]:


sal[["salary","phd","service"]].groupby(sal["rank"]).median()


# In[ ]:


#RESHAPING THE 


# In[16]:


import pandas as pd

data = {
    'User ID': [1, 1, 2, 2, 3, 3, 4, 3, 7],
    'Movie Name': [
        'Inception', 'Titanic', 'Inception', 'Avatar', 
        'Titanic', 'Avatar', 'Lion King', 'Inter Stellar', 'Bahubali'
    ],
    'Rating': [9, 8, 7, 18, 9, 8, 7, 8, 9]
}

df = pd.DataFrame(data)

pivot_table = df.pivot(index='User ID', columns='Movie Name', values='Rating')

print(pivot_table)


# In[ ]:




