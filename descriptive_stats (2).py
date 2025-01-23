#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
df = pd.read_csv("Universities.csv")
df


# In[11]:


np.mean(df["SAT"])


# In[12]:


np.median(df["Top10"])


# In[13]:


np.std(df["SFRatio"])


# In[14]:


#Find the variance
np.var(df["SFRatio"])


# In[15]:


df.describe()


# In[ ]:





# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns1
plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[21]:


sns.histplot(df["Accept"], kde = True)


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
s1 = [20,15,10,25,30,35,28,40,45,60]
scores1 = pd.Series(s1)
scores1


# In[13]:


plt.figure(figsize=(6,2))
plt.title("Boxplot for batsman scores")
plt.xlabel("scores")
plt.boxplot(scores1, vert = False)


# In[16]:


s2 = [20,15,10,25,30,35,28,40,45,60]
scores2 = pd.Series(s2)
scores2
plt.figure(figsize=(10,4))
plt.title("Boxplot for batsman scores")
plt.xlabel("scores")
plt.boxplot(scores2, vert = False)


# In[ ]:




