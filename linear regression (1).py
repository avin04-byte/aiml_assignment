#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.describe()


# # Obsevations

# 1> No Null values are found in data

# In[5]:


plt.scatter(data1['daily'],data1['sunday'])


# In[6]:


data1['daily'].corr(data1['sunday'])


# # Observations

# 1> Daily and Sunday has high positive correlation strength
# 
# 2> The relation between x(daily) and y(sunday) is seen to be linear as seen from scatter plot

# In[7]:


import statsmodels.formula.api as smf
model = smf.ols('sunday~daily',data = data1).fit()
model.summary()


# * Y = B0 + B1
# 
# * Predicted equation Y = 13.8356 + 1.3397X
# 
# * The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# 
# * Therefore the intercept coefficient may not be that much significant in prediction
# 
# * However the p-value for "daily" (beta_1) is 0.00 < 0.05
# 
# * Therefor the beta_1 coefficient is highly significant and is contributint to prediction

# In[8]:


x = data1['daily'].values
y = data1['sunday'].values
plt.scatter(x, y, color = 'm', marker = 'o', s = 30)
b0 = 13.84
b1 = 1.33
y_hat = b0 + b1*x
plt.plot(x, y_hat, color = 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.show


# # Observations

# 1> There are no missing values
# 
# 2> The daily column values appears to be right-skewed
# 
# 3> The sunday column values also appear to be right-skewed
# 
# 4> There are two outliers in both daily column and also in sunday column as observed from the boxplot

# In[9]:


x = data1['daily']
y = data1['sunday']
plt.scatter(data1['daily'], data1['sunday'])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[12]:


model.params


# In[13]:


print(f'model t-values:\n{model.tvalues}\n-----------------------------\nmodel p-values: \n{model.pvalues}')


# In[14]:


(model.rsquared,model.rsquared_adj)


# In[15]:


newdata=pd.Series([200,300,1500])
data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[16]:


model.predict(data_pred)


# In[17]:


pred=model.predict(data1['daily'])
pred


# In[19]:


data1['Y_hat']=pred
data1


# In[20]:


data1['residuals']=data1['sunday']-data1['Y_hat']
data1


# In[21]:


mse = np.mean((data1['daily']-data1['Y_hat'])**2)
rmse = np.sqrt(mse)
print('MSE: ',mse)
print('RMSE: ',rmse)


# In[22]:


plt.scatter(data1['Y_hat'], data1['residuals'])


# # Observations

# * The residuals data points are randomly scattered
# * There appears to be no trend and the residuals are randomly placed around the zero error line
# * Hence the assupmtion of homoscedasticty is satisfied(constant varience in residuals)

# In[24]:


import statsmodels.api as sm
sm.qqplot(data1['residuals'], line='45', fit=True)
plt.show()


# In[25]:


sns.histplot(data1['residuals'], kde=True)


# # Observations

# * The data points are seen to closely follow the reference line of normality
# * Hence the residuals are approximately normally distributed as also can be seen from the kde plot

# In[ ]:




