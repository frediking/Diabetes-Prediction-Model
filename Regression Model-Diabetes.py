#!/usr/bin/env python
# coding: utf-8

# In[179]:


#Load all packages and libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[181]:


#Load dataset
diabetes = datasets.load_diabetes()


# In[183]:


diabetes


# In[185]:


# Show feature names
print(diabetes.feature_names)


# In[187]:


# Create Feature and Target variables
X = diabetes.data
y = diabetes.target
print("dimension of X:", X.shape)
print("dimension of y:", y.shape)


# In[189]:


# Split data into testing & training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[191]:


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[193]:


# Dimensions of training data and testing data
print("feature training data:", X_train.shape, "/ target training data:", y_train.shape)
print("feature testing data:", X_test.shape, "/ target testing data:" , y_test.shape)
print("scaled feature training data:", X_train_scaled.shape)
print("scaled feature test data:", X_test_scaled.shape)


# In[195]:


#BUILDING LINEAR REGRESSION MODEL
# 1. defining the model
model = linear_model.LinearRegression()


# In[197]:


# 2. build training model
model.fit(X_train_scaled, y_train)


# In[199]:


# Applying training model to testing dataset
Y_pred = model.predict(X_test_scaled)
print(Y_pred)


# In[201]:


# PREDICTION RESULTS
# PRINT MODEL PERFORMANCE
print("Parameters:", model.coef_)       #coefficients of feature varibales(diabetes.feature_names)
print("Intercept:", model.intercept_)
print("MSE: %.2f"
      % mean_squared_error(y_test, Y_pred))  
print("Coefficient of determination:%.2f"
      % r2_score(y_test, Y_pred))


# In[203]:


feature_names = diabetes.feature_names
print(feature_names[0])


# In[97]:


# Each model coeffecient is multiplied by its corresponding feature variable 
# The sum of all products of model coef and features should produce the value of model intercept. EG:
# Y^ = model.coef_[0]*feature_names[0] +  model.coef_[1]*feature_names[1] +  model.coef_[2]*feature_names[2]...
# model.coef_[10]*feature_names[10] = model.intercept_


# In[205]:


y_test


# In[207]:


Y_pred


# In[209]:


# VISUALIZATIONS - SCATTERPLOTS
import seaborn as sns
import matplotlib.pyplot as plt


# In[211]:


# first look at data points on the plot
sns.scatterplot(x=y_test,y=Y_pred, )


# In[213]:

# FINAL AND MAIN SCATTER PLOT WITH REGRESSION MODEL LINE
plt.figure(figsize = (12, 8))
plt.scatter(y_test, Y_pred, marker ='+', alpha =0.5, color='blue', label = 'data points')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r-',
         lw = 2,
         label = 'Prediction Line')
plt.title('Diabetes Prediction')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Prediction Values (Y_pred)')
plt.legend()
plt.grid(True)
plt.savefig("Diabetes Prediction(scaled)", dpi=300, bbox_inches = 'tight')
plt.show()


# In[ ]:




