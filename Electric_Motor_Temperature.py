#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


motor=pd.read_csv("/Users/apple/Downloads/DS_Projects/Electric Motor Temperature/pmsm_temperature_data.csv")
motor.head()


# # EDA

# In[3]:


motor.shape


# In[4]:


motor.info()


# In[5]:


motor.describe().T


# In[6]:


#checking correlation with respect to torque
motor.corr()['torque']


# In[7]:


#correlation map
ax=plt.subplots(figsize=(30,20))
corr=motor.corr()

sns.heatmap(corr, annot=True, linewidths=1, fmt='.2f', 
            mask= np.zeros_like(corr,dtype=np.bool), 
            cmap=sns.diverging_palette(100,200,as_cmap=True), 
            square=True, ax=ax)

plt.show()


# In[8]:


#checking for missing data 
motor.isnull().sum()


# In[9]:


motor.hist(figsize = (35,25))
plt.show()


# In[8]:


sns.pairplot(motor)


# In[3]:


#checking for duplicates in the dataset
motor1=motor.drop_duplicates(keep='first')


# In[11]:


motor1.shape


# There is no duplicate values present in the dataset

# In[9]:


sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (25,8))
sns.countplot('profile_id',data=motor).set_title('count of profile_id')


# In[10]:


#barplot for profile_id 
sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (25,8))
grpd = motor.groupby(['profile_id'])
_df = grpd.size().sort_values().rename('samples').reset_index()
ordered_ids = _df.profile_id.values.tolist()
sns.barplot(y='samples', x='profile_id', data=_df, order=ordered_ids).set_title('Count of profile_id')


# In[11]:


sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize = (55,25))
plt.subplot(3,3,1)
sns.boxplot(x='profile_id', y='motor_speed', data=motor)


# In[4]:


x=motor.drop("motor_speed",axis=1)
y=motor.iloc[:,5:6]
x.head()


# In[5]:


y.head()


# In[6]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)


# In[7]:


#shape of train and test
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)


# # MODEL BUILDING

# In[8]:


#Multi Linear Regression
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(train_x,train_y)


# In[9]:


y_pred=lm.predict(test_x)
y_pred


# In[10]:


from sklearn.metrics import r2_score
score=r2_score(y_pred,test_y)
score


# In[11]:


#MSE
import sklearn
sklearn.metrics.mean_squared_error(test_y,y_pred)


# In[12]:


#RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error
sqrt(mean_squared_error(test_y,y_pred))


# In[13]:


#Ridge Regressioin
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(train_x,train_y)
pred = ridgeReg.predict(test_x)
r2_score(test_y,pred)


# In[14]:


#MSE for Ridge Regression
mean_squared_error(test_y,pred)


# In[15]:


#Lasso Regression 
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=0.3, normalize=True)

lassoReg.fit(train_x,train_y)

pred1 = lassoReg.predict(test_x)
lassoReg.score(test_x,test_y)


# In[16]:


#MSE for Ridge Regression
mean_squared_error(test_y,pred1)


# In[17]:


#cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score1=cross_val_score(lm, x, y, cv=10, scoring='r2').mean()
cross_val_score1


# In[18]:


cross_val_score2=cross_val_score(ridgeReg, x, y, cv=10, scoring='r2').mean()
cross_val_score2


# from sklearn.neighbors import KNeighborsRegressor
# scr_max = 0
# knn_test_score_list = []
# knn_train_score_list = []
# 
# for i in range(1,train_x.shape[0]+1):
#     knn = KNeighborsRegressor(n_neighbors = i)
#     knn.fit(train_x,train_y)
#     knn_test_scr = knn.score(test_x,test_y)
#     knn_test_score_list.append(knn_test_scr)
#     knn_train_scr = knn.score(train_x,train_y)
#     knn_train_score_list.append(knn_train_scr)
#     if knn_test_scr >= scr_max:
#         scr_max = knn_test_scr
#         index = i
# 
# print("Best K value = ",index)
# print("Best score = ",scr_max)

# In[19]:


#creating a model for SVM
from sklearn.svm import SVR
SVR=SVR(kernel='linear')
SVR.fit(train_x,train_y)


# In[20]:


SVR_y_pred=SVR.predict(test_x)
SVR_y_pred


# In[21]:


from sklearn.metrics import r2_score
SVR_score=r2_score(SVR_y_pred,test_y)
SVR_score


# #n_jobs :int or None, optional (default=None)
# The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
# 
# oob_score:bool, optional (default=False)
# whether to use out-of-bag samples to estimate the R^2 on unseen data.
# 
# 

# In[22]:


#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                         max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                         bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
                         warm_start=False, )


# In[23]:


RF.fit(train_x,train_y)


# In[11]:


RF_y_pred=RF.predict(test_x)


# In[12]:


from sklearn.metrics import r2_score
RF_score=r2_score(RF_y_pred,test_y)
RF_score


# In[12]:


#AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
ad=AdaBoostRegressor(base_estimator=RF,n_estimators=50,learning_rate=1,loss='linear')


# In[ ]:


ad.fit(train_x,train_y)
ada_y_pred=ad.predict(test_x)


# In[19]:


AdaBoost_RF_score=r2_score(ada_y_pred,test_y)
AdaBoost_RF_score


# In[12]:


#xgboost
import xgboost as xgb
xgb = xgb.XGBRegressor(base_estimator=RF,n_estimators=50,learning_rate=1,loss='linear')
xgb.fit(train_x,train_y)


# In[14]:


xgb.fit(train_x,train_y)
xbg_y_pred=xgb.predict(test_x)


# In[17]:


from sklearn.metrics import r2_score
XGBBoost_RF_score=r2_score(xbg_y_pred,test_y)
XGBBoost_RF_score


# In[18]:


# BaggingRegressor
from sklearn.ensemble import BaggingRegressor
BR=BaggingRegressor(base_estimator=RF,n_estimators=50,)
BR.fit(train_x,train_y)
BR_y_pred=BR.predict(test_x)
BR_score=r2_score(BR_y_pred,test_y)
BR_score


# In[20]:


#Nueral Networks MLPRegressor
from sklearn.neural_network import MLPRegressor
mlpR= MLPRegressor(hidden_layer_sizes=(100,))
mlpR.fit(train_x,train_y)
mlpr_y_pred=mlpR.predict(test_x)


# In[21]:


NN_score=r2_score(mlpr_y_pred,test_y)
NN_score


# In[ ]:




