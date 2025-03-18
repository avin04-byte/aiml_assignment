#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install xgboost


# In[4]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# In[5]:


df=pd.read_csv('diabetes03.csv')
df


# In[6]:


X=df.drop('class',axis=1)
y=df['class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:





# In[18]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print("------------------------------------------------------------------------------")
print(X_test_scaled )


# In[22]:


xgb=XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=42)
param_grid={
    'n_estimators':[100,150,200,300],
    'learning_rate':[0.01,0.1,0.15],
    'max_depth':[2,3,4,5],
    'subsample':[0.8,1.0],
    'colsample_bytree':[0.8,1.0]
}
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
grid_search=GridSearchCV(estimator=xgb,
                         param_grid=param_grid,
                         scoring='recall',
                         cv=skf,
                         verbose=1,
                         n_jobs=-1)


# In[23]:


grid_search.fit(X_train_scaled,y_train)
best_model=grid_search.best_estimator_
print('Best Parameters:',grid_search.best_params_)
print('Best Cross-Validated Recall:',grid_search.best_score_)
y_pred=best_model.predict(X_test_scaled)


# In[ ]:


print('\Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
print()


# In[ ]:


best_model.feature_importances_


# In[ ]:


features=pd.DataFrame(best_model.feature_importances_,index=df.iloc[:,:-1].columns,columns=['Importances'])
df1=features.sort_values(by='Importances')


# In[ ]:


import seaborn as sns
sns.barplot(data=df1,x=features.index,y='Importances',hue=features.index,palette="Set2")

