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


# In[3]:


cars = pd.DataFrame(cars, columns=['HP','VOL','SP','WT','MPG'])
cars.head()


# #### Description of columns
# * MPG : Milege of the car(Mile per Gallon)
# * HP : Horse Power of the car
# * VOL : Volume of the car(size)
# * SP : Top speed of the car(Miles per hour)
# * WT : Weight of the car(pounds)

# # EDA

# In[4]:


cars.info()


# In[5]:


cars.describe()


# In[6]:


cars.isna().sum()


# #### Observations
# * There are no missing values
# * There are 81 observations
# * The data types of the columns are relevant and valid

# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='s')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[12]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# #### OBSERVATIONS
# * There are some extreme values(outliers)observed in towards the right tail of SP and HP distribiution.
# * In VOL and WT columns, a few outliers are observed in both tails of their distribution.
# * The extreme values of cars data may have come from the specially designed nature of cars.
# * As this multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model.

# # Checking the duplicate rows

# In[14]:


cars[cars.duplicated()]


# In[15]:


cars.corr()


# # OBSERVATIONS
# * Between VOL and WT hasthe highest correlation variable(0.999203)
# - Between SP and HP has second higest corelation variable(0.973848)
# - Among x column(x1,x2,x3,x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT

# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(cars)
plt.show()


# In[22]:


model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()
model1.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




