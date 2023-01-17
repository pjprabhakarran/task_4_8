#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Packages

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[13]:


df1 = pd.read_csv("C:/Users/pjpra/OneDrive/Desktop/My_Learnings/Python/Task6/data-20221219T151715Z-001/data/train.csv")
df1.head()


# In[141]:


import io
df2 = pd.read_csv(r"C:/Users/pjpra/OneDrive/Desktop/My_Learnings/Python/Task6/data-20221219T151715Z-001/data/test.csv")
df2.head()
df2 = df2.drop('ID',axis = 1)


# In[15]:


df1.describe()


# In[16]:


#df1.drop(["ID"], axis = 1,inplace=True)
df1.columns[0]


# In[17]:


for each in range(len(df1.columns)-1):
    print(df1.iloc[0,each])
    df1.plot(x=each , y='MEDV', style='o')
    plt.title(f'{df1.columns[each]} vs MEDV')
    plt.xlabel(each)
    plt.ylabel('MEDV')
    plt.show()


# In[18]:


df1[["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]].corrwith(df1["MEDV"])


# In[26]:


X = df1.loc[:, ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]].values
y = df1.loc[:, 'MEDV'].values
# Syntax : dataset.loc[:, :-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[36]:


# Task 3

from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train,y_train)
print(f"coeff: \n{regress.coef_} \nIntercept: \n{regress.intercept_}")


# In[39]:


# Check for R2_Score
y_pred = regress.predict(X_test)
from sklearn import metrics
metrics.r2_score(y_pred,y_test)


# In[151]:


# Choosing the linear features
b_features = ['ZN','RM','B']
X = df1.loc[:,["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]].values
y = df1.loc[:,'MEDV'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4)

# Scale data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

# Fit model
regress.fit(X_train,y_train)
print(f"coeff: \n{regress.coef_} \nIntercept: \n{regress.intercept_}")

# R2 score of the model
y_pred = regress.predict(X_train)
from sklearn import metrics
metrics.r2_score(y_pred,y_train)


# In[154]:


# Predicting the test data
y_pred = regress.predict(df2.loc[:,["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]])
df3 = df2
df3['pred_MEDV'] = y_pred
df3.to_csv("Predic_Test_Data.csv")


# In[155]:


df3


# In[166]:


# Fitting Ridge Model

# Choosing the linear features
b_features = ['ZN','RM','B']
X = df1.loc[:,["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]].values
y = df1.loc[:,'MEDV'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# Scale data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import Ridge
l2_norm = Ridge()
l2_norm.fit(X_train,y_train)
print(f"coeff: \n{l2_norm.coef_} \nIntercept: \n{l2_norm.intercept_}")

# R2 score of the model
y_pred = l2_norm.predict(X_train)
from sklearn import metrics
metrics.r2_score(y_pred,y_train)


# In[165]:


# Fitting Lasso Model

# Choosing the linear features
b_features = ['ZN','RM','B']
X = df1.loc[:,["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]].values
y = df1.loc[:,'MEDV'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

# Scale data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_test = ss.transform(X_test)

from sklearn.linear_model import Lasso
l1_norm = Lasso()
l1_norm.fit(X_train,y_train)
print(f"coeff: \n{l1_norm.coef_} \nIntercept: \n{l1_norm.intercept_}")

# R2 score of the model
y_pred = l1_norm.predict(X_train)
from sklearn import metrics
metrics.r2_score(y_pred,y_train)


# In[ ]:




