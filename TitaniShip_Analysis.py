#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
data = pd.read_csv('Titanic.csv')
data


# In[2]:


data.info()


# In[3]:


data.describe()


# In[4]:


data.isna().sum()


# In[5]:


get_ipython().system('pip install mlxtend')


# In[6]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[7]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[8]:


titanic.info()


# #### Observations
# - All columns are objects data type and categorical in nature
# - There are no null values
# - As the columns are categorical,we can adopt one-hot-encoding

# In[9]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[14]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[16]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[11]:


df = pd.get_dummies(titanic,dtype=int)
df.head()


# In[12]:


df.info()


# In[13]:


frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[18]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules.sort_values(by='lift',ascending = False)


# In[19]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show


# In[20]:


import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()


# In[21]:


plt.scatter(rules['confidence'], rules['lift'])
plt.show()


# In[22]:


rules[rules['consequents']==({'Survived_Yes'})]


# In[ ]:




