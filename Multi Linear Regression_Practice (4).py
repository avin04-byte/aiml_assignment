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


# In[ ]:





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


# In[16]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1['pred_y1'] = pred_y1
df1.head()


# In[17]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1['actual_y1'], df1['pred_y1'])
print('MSE :', mse)
print('RMSE :', np.sqrt(mse))


# # Checking for multicollinearity amomg X-columns using VIF method

# In[18]:


cars.head()


# In[19]:


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


# # Observations
# - The ideal range of VIF values shall be between 0 to 10.However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity prone
# - Hence it is decide to drop one of the columns(either VOL or WT) to overcome the multicollinearity
# - It id decided to drop WT and retain VOL column in further models

# In[20]:


cars1 = cars.drop('WT', axis=1)
cars1.head()


# In[21]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()


# # Performance metrics for model2

# In[22]:


df2 = pd.DataFrame()
df2['actual_y2'] = cars['MPG']
df2.head()


# In[23]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2['pred_y2'] = pred_y2
df2.head()


# In[24]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2['actual_y2'], df2['pred_y2'])
print('MSE: ', mse)
print('RMSE: ', np.sqrt(mse))


# # Observations
# - The adjust R-squqred value improved slightly to 0.76.
# - All the p-values for model parameters are less than 5% hence they are significant.
# - Therefore the HP, VOL, SP columns are finalized as the significant predictor for the MPG.
# - There is no improvment in MSE value.

# In[25]:


cars1.shape


# In[26]:


k = 3
n = 81
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# ### from statsmodels.graphics.regressionplots import influence_plot
# 

# In[27]:


influence_plot(model1,alpha=.05)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# # Observations
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencer
# - As thier H Leverage values are higher and size is higher

# In[28]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[29]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[30]:


model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()
model3.summary()


# In[34]:


df3 = pd.DataFrame()
df3['actual_y3'] = cars2['MPG']
df3.head()


# In[35]:


pred_y3 = model3.predict(cars.iloc[:,0:3])
df3['pred_y3'] = pred_y3
df3.head()


# In[37]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3['actual_y3'], df3['pred_y3'])
print('MSE: ', mse)
print('RMSE: ', np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[38]:


model3.resid
model3.fittedvalues


# In[43]:


import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title('Normal Q-Q plot of residuals')
plt.show


# In[40]:


sns.displot(model3.resid, kde = True)


# In[44]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()



plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:





# In[ ]:




