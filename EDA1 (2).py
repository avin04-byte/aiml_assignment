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


# In[5]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[6]:


#checking for Duplicate rows
data1[data1.duplicated(keep = False)]


# In[7]:


data1[data1.duplicated()]


# In[8]:


#Drop duplcated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[9]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[10]:


#Display data1 info()
data1.info()


# In[11]:


#Display data1 missing value count in each column using isnull().sum()
data1.isnull().sum()


# In[12]:


#Visualize data1 missing values graph
cols = data1.columns
colors = ['black', 'pink']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[13]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[14]:


#Replace the Ozone missing values with median value
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[15]:


data1['Solar'] = data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ", median_ozone)
print("Mean of Solar: ", mean_ozone)


# In[17]:


data1['Solar'] = data1['Solar'].fillna(mean_solar)
data1.isnull().sum()


# In[18]:


print(data1['Weather'].value_counts())
mode_weather = data1['Weather'].mode()[0]
print(mode_weather)


# In[19]:


data1['Weather'] = data1['Weather'].fillna(mode_weather)
data1.isnull().sum()


# In[22]:


mode_month = data1['Month'].mode()[0]
data1['Month'] = data1['Month'].fillna(mode_month)
data1.isnull().sum()


# In[23]:


#Reset the index column
data1.reset_index(drop=True)


# In[38]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1,3]})
sns.boxplot(data=data1["Ozone"], ax=axes[0], color='blue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1['Ozone'], kde=True, ax=axes[1], color='yellow', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[ ]:


## Obervations
> The ozone column has extrem values beyond 81 as seen from box plot
> The same is confirmed from the below right=skewed histogram


# In[40]:


sns.violinplot(data=data1["Ozone"], color='black')
plt.title("Violin Plot")
plt.show()


# In[ ]:




