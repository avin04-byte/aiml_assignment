#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in Multilinear Regression
# * Linearity: The relationship between the predictors and the response is linear.
# * Independence: Observations are independent of each other.
# * Homoscedasticity: The residuals(differences between observed and predicted values)exhibit constant variance at all levels of the predictor.
# * Normal Distribution of Errors: The rediduals of the model are normally distributed.
# * No multicollinearity: the independent variables should not be too high correltion with

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars, columns=['HP','VOL','SP','WT','MPG'])
cars.head()


# #### Description of columns
# * MPG : Milege of the car(Mile per Gallon)
# * HP : Horse Power of the car
# * VOL : Volume of the car(size)
# * SP : Top speed of the car(Miles per hour)
# * WT : Weight of the car(pounds)

# # EDA

# In[5]:


cars.info()


# In[6]:


cars.describe()


# In[7]:


cars.isna().sum()


# #### Observations
# * There are no missing values
# * There are 81 observations
# * The data types of the columns are relevant and valid

# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.015, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[ ]:




