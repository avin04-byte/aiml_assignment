#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[ ]:


data = pd.read_csv("iris.csv")


# In[ ]:


data


# In[2]:


import seaborn as sns
counts = data["variety"].value_counts()
sns.barplot(data = counts)


# In[3]:


data.info()


# In[34]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))


# In[4]:


data[data.duplicated(keep=False)]


# In[ ]:


data[data.duplicated()]


#  ### Observations
#  - There are 150 rows and 5 columns
#  - There are no null values 
#  - There are 1 duplicated value 
#  - The x-columns are  sepal.length,sepal.width,petal.length,petal.width
#  - The y-column is variety
#  - All x values are continous
#  - y column is catgorical
#  - There are three flower categories

# In[5]:


data.drop_duplicates(keep='first', inplace = True)


# In[ ]:


data[data.duplicated()]



# In[6]:


data = data.reset_index(drop=True)


# In[ ]:


data



# In[7]:


labelencoder = LabelEncoder()
data.iloc[:,-1] = labelencoder.fit_transform(data.iloc[:,-1])
data.head()


# In[8]:


data.tail()


# In[36]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))


# In[37]:


data.info()


# In[38]:


data.head(4)


# In[39]:


X = data.iloc[:,0:4]
Y = data['variety']



# In[40]:


X


# In[41]:


Y


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
x_train




# In[43]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth= None)

model.fit(x_train, y_train)


# In[44]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[45]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[46]:


preds = model.predict(x_test)
preds


# In[47]:


print(classification_report(y_test,preds))


# In[49]:


pred_train = model.predict(x_train)
print(classification_report(y_train,pred_train))


# In[ ]:




