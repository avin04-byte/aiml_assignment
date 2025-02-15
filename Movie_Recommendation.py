#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
movies_df = pd.read_csv('Movie.csv')
movies_df


# In[3]:


movies_df.info()


# In[5]:


movies_df.describe()


# In[6]:


movies_df.isnull().sum()


# In[7]:


import matplotlib.pyplot as plt
counts = movies_df['movie'].value_counts()
plt.bar(counts.index, counts.values)


# In[8]:


counts = movies_df['rating'].value_counts()
plt.bar(counts.index, counts.values)


# In[9]:


rating = movies_df['movie'].value_counts()
rating


# In[10]:


import seaborn as sns


# ### EDA

# In[13]:


counts = movies_df['movie'].value_counts().reset_index()
print(counts)
columns = ['movie', 'counts']
plt.xlabel('Movies')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')
sns.barplot(data=counts,x='movie', y='count', hue = 'movie',palette='Set2')


# In[15]:


print(movies_df.userId.unique())
len(movies_df.userId.unique())


# In[16]:


movies_df.sort_values('userId')


# In[18]:


counts = movies_df['rating'].value_counts().reset_index()
print(counts)
columns = ['rating', 'counts']
plt.xlabel('Rating')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')
sns.barplot(data=counts,x='rating', y='count', hue = 'rating',palette='Set2')


# In[20]:


user_movies_df=movies_df.pivot_table(index='userId',columns='movie',values='rating')
user_movies_df


# In[24]:


user_movies_df.fillna(0,inplace=True)
user_movies_df


# In[27]:


from sklearn.metrics import pairwise_distances
user_sim=1-pairwise_distances(user_movies_df.values,metric='cosine')
user_sim


# In[28]:


user_sim.shape


# In[29]:


np.fill_diagonal(user_sim,0)
user_sim


# In[31]:


user_sim_df=pd.DataFrame(user_sim)
user_sim_df


# In[32]:


movies_df.userId.unique()


# In[36]:


user_sim_df.index=movies_df.userId.unique()
user_sim_df.columns=movies_df.userId.unique()
user_sim_df.idxmax(axis=1)[0:50]


# In[37]:


movies_df[(movies_df['userId']==6)|(movies_df['userId']==168)]


# In[39]:





# In[40]:


user_6=movies_df[movies_df['userId']==6]
user_168=movies_df[movies_df['userId']==168]
moview_watched_by_user168=list(set(user_168.movie))
moview_watched_by_user6=list(set(user_6.movie))
print(moview_watched_by_user168)
print(moview_watched_by_user6)


# In[41]:


moview_watched_by_user168=list(set(user_168.movie))
moview_watched_by_user6=list(set(user_6.movie))
for movie_name in moview_watched_by_user6:
    if movie_name not in moview_watched_by_user168:
        print("Recommendation: ", movie_name)


# In[ ]:




