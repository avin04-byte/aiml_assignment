#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,StratifiedKFold


# In[6]:


dataframe = pd.read_csv("diabetes01.csv")
dataframe


# In[9]:


from numpy import array
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
#array1 = dataframe.values
X=dataframe.iloc[:,0:8]
Y=dataframe.iloc[:,8]
kfold=StratifiedKFold(n_splits=10,random_state=2023,shuffle=True)
model=RandomForestClassifier(n_estimators=200,random_state=20,max_depth=None)
results=cross_val_score(model,X,Y,cv=kfold)
print(results)
print(results.mean())


# In[ ]:




