#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Load all packages and libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


#Load dataset
diabetes = datasets.load_diabetes()


# In[6]:


diabetes


# In[7]:


# Show feature names
print(diabetes.feature_names)


# In[8]:


# Create Feature and Target variables
X = diabetes.data
y = diabetes.target
print("dimension of X:", X.shape)
print("dimension of y:", y.shape)


# In[9]:


# Split data into testing & training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[10]:


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[11]:


# Dimensions of training data and testing data
print("feature training data:", X_train.shape, "/ target training data:", y_train.shape)
print("feature testing data:", X_test.shape, "/ target testing data:" , y_test.shape)
print("scaled feature training data:", X_train_scaled.shape)
print("scaled feature test data:", X_test_scaled.shape)


# In[12]:


#BUILDING LINEAR REGRESSION MODEL
# 1. defining the model
model = LinearRegression()


# In[13]:


# 2. build training model
model.fit(X_train_scaled, y_train)


# In[14]:


import joblib

# Save the model to a file
joblib.dump(model, "diabetes_LGmodel.pkl")


# In[15]:


# Applying training model to testing dataset
Y_pred = model.predict(X_test_scaled)
print(Y_pred)


# In[16]:


# PREDICTION RESULTS
# PRINT MODEL PERFORMANCE
print("Parameters:", model.coef_)       #coefficients of feature varibales(diabetes.feature_names)
print("Intercept:", model.intercept_)
print("MSE: %.2f"
      % mean_squared_error(y_test, Y_pred)) 
print("Coefficient of determination:%.2f"
      % r2_score(y_test, Y_pred))


# # Each model coeffecient is multiplied by its corresponding feature variable 
# # The sum of all products of model coef and features should produce the value of model intercept. EG:
# # Y^ = model.coef_[0]*feature_names[0] +  model.coef_[1]*feature_names[1] +  model.coef_[2]*feature_names[2]...
# # model.coef_[10]*feature_names[10] = model.intercept_

# In[17]:


y_test


# In[18]:


Y_pred


# In[100]:


# VISUALIZATIONS - SCATTERPLOTS
import seaborn as sns
import matplotlib.pyplot as plt


# In[101]:


# # first look at data points on the plot
sns.scatterplot(x=y_test,y=Y_pred, )


# In[102]:


# FINAL & MAIN SCATTERPLOT WITH REGRESSION/PREDICTION LINE
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


# In[19]:


# CREATING POLYNOMIAL FEATURES
from sklearn.preprocessing import PolynomialFeatures


# In[20]:


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[21]:


# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)


# In[22]:


# Transform  X into X_poly with new features
X_poly = poly.fit_transform(X_scaled)


# In[23]:


# Split data into training and testing

X_poly_train, X_poly_test, y_train, y_test =  train_test_split(X_poly, y, test_size=0.2, random_state=42)


# In[24]:


# Ridge Regression
model = Ridge(alpha=10.0)
model.fit(X_poly_train, y_train)


# In[25]:


import joblib

# Save the model to a file
joblib.dump(model, 'ridge_regression_model.pkl')


# In[26]:


# Predict on the test set and calculate R^2 
y_pred_test = model.predict(X_poly_test)
r2 = r2_score(y_test, y_pred_test)
mse_poly = mean_squared_error(y_test, y_pred_test)
print("MSE on test set:", mse_poly)
print("R2 score on test set:", r2)


# In[27]:


# Check R2 on training set to spot overfitting
y_train_pred = model.predict(X_poly_train)
r2_train = r2_score(y_train, y_train_pred)

print("r2 score on training set:" ,r2_train)


# In[34]:


# EXAMPLE REAL WORLD USE CASE OF THE LINEAR REGRESSION MODEL

# Load the saved models
diabetes_LGmodel = joblib.load("diabetes_LGmodel.pkl")
ridge_regression_model = joblib.load("ridge_regression_model.pkl")

# Create a new patient data
# Patient data: [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
patient_data = [[0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235,
                 -0.03482076, -0.04340085, -0.00259226, 0.01990749, -0.01764613]]

# Scale the patient data
scaled_data = scaler.transform(patient_data)

# Predict using Linear Regression
prediction_lr = diabetes_LGmodel.predict(scaled_data)[0]

# Display results
print(f"Linear Regression predicts a diabetes progression score of {prediction_lr:.2f}")


# In[35]:


# EXAMPLE REAL WORLD USE CASE OF THE RIDGE REGRESSION MODEL

# Transform patient data to polynomial features
patient_data_poly = poly.transform(scaled_data)

# Now predict using Ridge Regression
prediction_ridge = ridge_regression_model.predict(patient_data_poly)[0]

print(f"Ridge Regression predicts a diabetes progression score of {prediction_ridge:.2f}")

