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

# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='s')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


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

# In[11]:


cars[cars.duplicated()]


# In[12]:


cars.corr()


# # OBSERVATIONS
# * Between VOL and WT hasthe highest correlation variable(0.999203)
# - Between SP and HP has second higest corelation variable(0.973848)
# - Among x column(x1,x2,x3,x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(cars)
plt.show()


# In[14]:


model1 = smf.ols('MPG~WT+VOL+SP+HP', data=cars).fit()
model1.summary()


# #### Obervations from model summary
# * The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is close to zero, including that all or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further exlored

# In[15]:


df1 = pd.DataFrame()
df1['actual_y1'] = cars['MPG']
df1.head()


# In[17]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1['pred_y1'] = pred_y1
df1.head()


# In[18]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1['actual_y1'], df1['pred_y1'])
print('MSE :', mse)
print('RMSE :', np.sqrt(mse))


# # Checking for multicollinearity amomg X-columns using VIF method

# In[20]:


cars.head()


# In[21]:


rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[ ]:




