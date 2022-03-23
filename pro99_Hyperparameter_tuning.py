#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;font-size:30px;" > Demand Forecasting for a Store </h1>

# <h1>Hyper-parameter tuning

# <h3>Importing libraries

# In[1]:


# Importing useful libraries
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import time
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.chdir("D:\AAAAAAA RAJSHREE\DATA SCIENCE\DATASETs\Demand forecasting for a store")


# <h3>Loading files

# In[3]:


#Reading csv file using pandas
train = pd.read_csv("train.csv")
center = pd.read_csv("fulfilment_center_info.csv")
meal = pd.read_csv("meal_info.csv")
#Merging dataframes using "merge" function
df = train.merge(meal, on="meal_id")
df = df.merge(center, on="center_id")


# In[4]:


# As the data is a time series data
# Sorting the dataframe wrt week
df = df.sort_values(by="week")
# One hot Encoding to categorical columns - "category", "cuisine" and "center_type"
df = pd.get_dummies(df)
df.head()


# <h3>Removing outliers

# In[5]:


# Calculating quartile 1 and quartile 3 of log of the column num_orders
q1 = np.percentile(np.log(df["num_orders"]),25)
q3 = np.percentile(np.log(df["num_orders"]),75)
#Calculating Inter Quartile Region
iqr = q3-q1
# Removing the data points which are less than and greater than q1-1.5*iqr and q3+1.5*iqr respectively
df = df[np.log(df["num_orders"])>=q1-1.5*iqr]
df = df[np.log(df["num_orders"])<=q3+1.5*iqr]
# Also removing the data points whose checkout_price is greater than 800
df = df[df["checkout_price"]<800]


# In[6]:


# Adding an engineered feature "discount_percentage"
df["discount_percentage"] = (df["base_price"]-df["checkout_price"])*100/df["base_price"]
# Converting all column names to lower case
df = df.rename(columns=str.lower)
# Removing column "id" and "checkout_price" 
df = df.drop(["id","checkout_price","base_price","city_code","region_code"],axis=1)
df


# In[7]:


# Separating dependent and independent variables
# Assigning column "num_orders" to dependent_variable
dependent_variable = "num_orders"
# Assigning the list of columns to independent_variable
independent_variable = df.columns.tolist()
# Removing dependent_variable from the list of column in independent_variable
independent_variable.remove(dependent_variable)
# Assigning the values of independent and dependent variables to x and y respectively.
x = df[independent_variable].values
y = df[dependent_variable].values


# In[8]:


# Importing all the required libraries
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,SGDRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# In[9]:


# Splitting the data with TimeSeriesSplit
# Older data are assigned for training purpose and the recent data for testing
tss = TimeSeriesSplit(n_splits=5)
for i,j in tss.split(x):
    x_train, x_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# <h2>Hyper-parameter tuning on XGBoost

# In[10]:


# Building DMatrices
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


# In[11]:


# Default parameters of XGBoost
params = {'max_depth': 7,
 'min_child_weight': 6,
 'eta': 0.2,
 'subsample': 1.0,
 'colsample_bytree': 0.6,
 'eval_metric': 'rmse',
 'objective': 'reg:squarederror'}


# In[12]:


# Assigning the number of boosting round to a large value
num_boost_round = 999


# In[13]:


# Training the data with default parameters of XGBoost
# Evaluation is done on test data after each boosting round
# Also printing the minimum rmse value after every round till it gets the minimum rmse
# 10 boosting rounds are performed after getting the minimum rmse
s=time.time()
model = xgb.train(params,dtrain,num_boost_round=num_boost_round,                  evals=[(dtest, "Test")],early_stopping_rounds=10)
print("Best RMSE with default parameters: {:.2f} with {} rounds".
      format(model.best_score, model.best_iteration+1))
print("Time taken:",round(time.time()-s),"sec")


# <h3>1. Tuning max_depth and min_child_weight

# In[14]:


s=time.time()
# Forming parameters for gridsearch
gridsearch_params = [(max_depth, min_child_weight)
                     for max_depth in range(8,11)
                     for min_child_weight in range(6,9)]
# Defining initial best params and minimum RMSE
min_rmse = float("Inf")
best_params = None
# Iterating each value gridsearch params
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth,min_child_weight))
    # Updating parameters
    params['max_depth'],params['min_child_weight']=max_depth,min_child_weight
    # Running CV
    cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5,                        metrics={'rmse'},early_stopping_rounds=10)
    # Updating best RMSE and boosting rounds
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    # Updating min_rmse and best_params
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))
print("Time taken:",round(time.time()-s),"sec")


# In[15]:


# Updating the parameters with the best parameters found above
params["max_depth"] = best_params[0]
params["min_child_weight"] = best_params[1]
params


# <h3>2. Tuning subsample and colsample_bytree

# In[16]:


s=time.time()
gridsearch_params = [(subsample, colsample)
                     for subsample in [i/10. for i in range(7,11)]
                     for colsample in [i/10. for i in range(7,11)]]
# Assigning min_rmse to a large value
min_rmse = float("Inf")
# Assigning best_params to value None
best_params = None
# Iterating from the largest value and to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(subsample, colsample))
    # Updating our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Running CV
    cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5,                        metrics={'rmse'},early_stopping_rounds=10)
    # Updating best score
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = (subsample,colsample)
print("Best params: {}, {}, RMSE: {}".format(best_params[0], best_params[1], min_rmse))
print("Time taken:",round(time.time()-s),"sec")


# In[17]:


# Updating the parameters with the best parameters found above
params["subsample"] = best_params[0]
params["colsample_bytree"] = best_params[1]
params


# <h3>3. Tuning ETA (learning rate)

# In[18]:


s=time.time()
get_ipython().run_line_magic('time', '')
# Assigning min_rmse to a large value
min_rmse = float("Inf")
# Assigning best_params to value None
best_params = None
# Iterating each values
for eta in [.3, .2, .1, 0.01]:
    print("CV with eta={}".format(eta))
    # Updating parameters
    params['eta'] = eta
    # Running CV
    get_ipython().run_line_magic('time', "cv_results = xgb.cv(params,dtrain,num_boost_round=num_boost_round,seed=42,nfold=5,                              metrics=['rmse'],early_stopping_rounds=10)")
    # Updating best score with its boost_rounds
    mean_rmse = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tRMSE {} for {} rounds\n".format(mean_rmse, boost_rounds))
    # Checking whether mean_rmse is less than min_rmse
    if mean_rmse < min_rmse:
        min_rmse = mean_rmse
        best_params = eta
print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
print("Time taken:",round(time.time()-s),"sec")


# <h2>Best parameters

# In[19]:


# Updating the parameters with the best parameters found above
params["eta"] = best_params
params


# <h3>Training with the best parameters

# In[20]:


# Training the data with the best parameters found above
s=time.time()
model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=[(dtest, "Test")],                  early_stopping_rounds=10)
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))
print("Time taken:",round(time.time()-s),"sec")


# In[21]:


# Training the data with the best parameters found above 
# This time number of boosting round is set till we get the minimum rmse value
# So we do not need the early stopping round this time
s=time.time()
num_boost_round = model.best_iteration + 1
best_model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=[(dtest, "Test")])
print("Time taken:",round(time.time()-s),"sec")


# <h3>Feature importance

# In[22]:


# Plotting feature importance graph for the first 10 features.
xgb.plot_importance(best_model, height=0.7, grid=False, max_num_features=10)
plt.show()


# <h3>Prediction and error

# In[23]:


# Prediction is done on train and test data and errors are compared to check for overfitting.
# As the difference between the errors are not far apart the model doesn't overfit
# Model is performing good with an accuracy of 80% on unseen data

# Calculating train error after prediction
RMSE_train = round(np.sqrt(mean_squared_error(y_train,best_model.predict(dtrain))),3)
# Calculating accuracy for the prediction done on train data
R2_train = round(r2_score(y_train,best_model.predict(dtrain)),3)

# Calculating test error after prediction
RMSE_test = round(np.sqrt(mean_squared_error(y_test,best_model.predict(dtest))),3)
# Calculating accuracy for the prediction done on test data
R2_test = round(r2_score(y_test,best_model.predict(dtest)),3)

print("         Train         Test")
print("________________________________")
print(" RMSE:  {}      {}".format(RMSE_train, RMSE_test))
print("   R2:    {}        {}".format(R2_train, R2_test))

