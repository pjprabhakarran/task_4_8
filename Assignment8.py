#!/usr/bin/env python
# coding: utf-8

# # Assignment8

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # How Much is Your Car Worth?
# 
# Data about the retail price of 2005 General Motors cars can be found in `car_data.csv`.
# 
# The columns are:
# 
# 1. Price: suggested retail price of the used 2005 GM car in excellent condition.
# 2. Mileage: number of miles the car has been driven
# 3. Make: manufacturer of the car such as Saturn, Pontiac, and Chevrolet
# 4. Model: specific models for each car manufacturer such as Ion, Vibe, Cavalier
# 5. Trim (of car): specific type of car model such as SE Sedan 4D, Quad Coupe 2D          
# 6. Type: body type such as sedan, coupe, etc.      
# 7. Cylinder: number of cylinders in the engine        
# 8. Liter: a more specific measure of engine size     
# 9. Doors: number of doors           
# 10. Cruise: indicator variable representing whether the car has cruise control (1 = cruise)
# 11. Sound: indicator variable representing whether the car has upgraded speakers (1 = upgraded)
# 12. Leather: indicator variable representing whether the car has leather seats (1 = leather)
# 
# ## Tasks, Part 1
# 
# 1. Find the linear regression equation for mileage vs price.
# 2. Chart the original data and the equation on the chart.
# 3. Find the equation's $R^2$ score (use the `.score` method) to determine whether the
# equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)
# 
# ## Tasks, Part 2
# 
# 1. Use mileage, cylinders, liters, doors, cruise, sound, and leather to find the linear regression equation.
# 2. Find the equation's $R^2$ score (use the `.score` method) to determine whether the
# equation is a good fit for this data. (0.8 and greater is considered a strong correlation.)
# 3. Find the combination of the factors that is the best predictor for price.
# 
# ## Tasks, Hard Mode
# 
# 1. Research dummy variables in scikit-learn to see how to use the make, model, and body type.
# 2. Find the best combination of factors to predict price.

# In[51]:


df = pd.read_csv("C:/Users/pjpra/OneDrive/Desktop/My_Learnings/Python/Task6/Ass_5,6,7,8/car_data.csv")


# In[52]:


df.head()
df.shape


# In[53]:


df.describe()


# In[65]:


"""
df.isnull().sum()
df.drop_duplicates()
df.shape
df.Cylinder.unique()
df.Liter.unique()
df.Doors.unique()
df.Cruise.unique()
df.Sound.unique()
df.Leather.unique()
"""

dfd = pd.get_dummies(df['Doors'])
dfd


# In[55]:


# Task_1

from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[56]:


# EDA of Mileage vs Price
df.plot(x = 'Mileage', y = 'Price', style='o')
plt.title("Mileage vs Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.show()


# In[57]:


df[['Mileage']].corrwith(df['Price'])
df.corr()
# There is no linear relation between the Mileage and Price as there is no Correlation


# In[58]:


X = df.loc[:,['Mileage']].values
y = df.loc[:,'Price'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 6)


# In[59]:


# Linear Model Fitting
from sklearn.linear_model import LinearRegression
regress = LinearRegression()
regress.fit(X_train,y_train)
print("m:",regress.coef_, "\nc:",regress.intercept_)


# In[60]:


# R2 value
y_pred = regress.predict(X_test)
from sklearn import metrics
print("R2_score:",round(metrics.r2_score(y_test,y_pred),4))


# In[61]:


# Task_2
X = df.loc[:,['Mileage','Cylinder','Liter','Doors','Cruise','Sound','Leather']].values
y = df.loc[:,'Price'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 7)
regress.fit(X_train,y_train)
y_pred = regress.predict(X_test)
print(metrics.r2_score(y_pred,y_test))


# In[62]:


score_comp = 0
r2 = []
r2_s = 0
y = df.loc[:,['Price']].values
for i in range(len(df.columns)-1):
    feature = combinations(["Mileage","Cylinder","Liter","Doors","Cruise","Sound","Leather"],i+1)
    for pair in list(feature):
        X = df.loc[:,list(pair)].values
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        regress.fit(X_train,y_train)
        y_pred = regress.predict(X_test)
        r2.append({pair:metrics.r2_score(y_test,y_pred)})
        r2_s = metrics.r2_score(y_test,y_pred)
        if score_comp < r2_s:
            score = r2_s            


print(f"Best combination and R2_Score is {pair}:{r2_s}")


# In[48]:


# Task 3

# label encoding
from sklearn.preprocessing import LabelEncoder

# Creating the initial dataframe
bridge_types = ('Arch','Beam','Truss','Cantilever','Ties Arch','Suspension','Cable')
bridge_df = pd.DataFrame(bridge_types, columns = ['Bridge_Types'])
print(bridge_df)

encoder = LabelEncoder()

#Transformation
df['Bridge_Types_Cat'] = encoder.fit_transform(bridge_df['Bridge_Types'])
print(bridge_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




