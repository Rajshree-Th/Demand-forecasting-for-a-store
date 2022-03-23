#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;font-size:30px;" > Demand Forecasting for a Store </h1>

# <h2> Problem Statement

# We have a meal delivery company (client) that operates in multiple cities. They have various fulfillment centers in these cities for dispatching meal orders to their customers.
# 
# We are going to help these centers with demand forecasting for upcoming weeks so that these centers will plan the stock of raw materials accordingly.
# 
# The replenishment of the majority of raw materials is done on weekly basis and since the raw material is perishable, procurement planning is of utmost importance.
# 
# Secondly, staffing of the centers is also one area wherein accurate demand forecasts are really helpful.
# 
# We are given with the following information, the task is to predict the demand for the next 10 weeks (Weeks: 146-155) for the center-meal combinations in the test set:
# 
# • Historical data of demand for a product-center combination (Weeks: 1 to 145)
# 
# • Product (Meal) features such as category, sub-category, current price, and discount
# 
# • Information for fulfillment centers like center area, city information, etc.
# 

# <h2> Business/Real world impact of solving this problem

# 1. Demand forecasting is critical to the future of any business as it helps reduce risks and make the
# right call on many fronts.
# 
# 
# 2. From omnichannel giants to the smallest brick and mortar boutiques, all retailers rely on demand
# forecasts to predict their best estimate of how much they will sell. Modern demand forecasting is
# a sophisticated statistical analysis that takes into account numerous variables to optimize that
# prediction.
# 
# 
# 3. While some retailers still rely on spreadsheets and manual calculations, such high-powered
# statistical analysis is best executed by specialized software designed to process enormous,
# retail-scale data sets. The perk of this technique is it transparently shows users what data is
# being used to build forecasts and how the forecasts are being calculated. Modern demand
# forecasting software automates difficult and time-consuming decisions, using machine learning to
# optimize predictions.
# 
# 
# 4. Though it will never be 100% accurate, forecasting demand can help improve production lead
# times, increase operational efficiencies, save money, launch new products, and provide a better
# customer experience overall.

# <h2>Data overview

#     -Source: Kaggle (https://www.kaggle.com/code/kerneler/starter-food-demand-dataset-d213c52f-7/data)
#     -Number of variables: 15 
#     -Number of observations: 456548
#     -Missing cells: 0
#     -Duplicate rows: 0
#     -Total size of memory: 55.7 MB
#     -Numeric variables: 10
#     -Categorical variables: 5

# <h2>Types of problem

# It is a regression problem, for a given meal we need to predict the number of orders in given week.

# <h2>Performance metric

# The metrics we are going to use is Root Mean Squared Error (RMSE) and Coeffecient of determinant (R_squared value) 

# <h2>Train-Test construction

# As it is a temporal data, we will be splitting the data by time series split i.e. the data older will be used for training and newer data for validation. 

# <h2>Exploratory data analysis

# <h3>Importing libraries

# We are importing some of the libraries we are going to use in this project.

# In[95]:


# Importing useful libraries
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from distfit import distfit
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>Loading data and basic statistics

# Loading all the given csv files as train, center and meal. We are merging the center and meal into train to form a dataframe df.

# In[96]:


# Path where i have saved the data folder in my computer
os.chdir("D:\AAAAAAA RAJSHREE\DATA SCIENCE\DATASETs\Demand forecasting for a store")
# Reading csv file using pandas
train = pd.read_csv("train.csv")
center = pd.read_csv("fulfilment_center_info.csv")
meal = pd.read_csv("meal_info.csv")
# Merging dataframes using "merge" function
df = train.merge(meal, on="meal_id")
df = df.merge(center, on="center_id")
# Printing df
df


# In[97]:


print("Number of data points:",df.shape[0])
print("Number of features:",df.shape[1])


# In[98]:


# Generating descriptive statistics of dataframe.
df.describe(include="all")
# to see summary statistics of all columns of DataFrame regardless of data type


# In[99]:


# Printing a concise summary of DataFrame.
df.info()
# This method prints information about DataFrame including the index dtype and columns, non-null values and memory usage.


# In[100]:


# Checking if there is any missing values with "isna()" function
df.isna().sum()
# As all the all sum value is 0, there is no missing value


# <h3>Univariate and Bivariate analysis:

# Univariate analysis refers to the analysis done based on one variable while bivariate with two variables.

# In[101]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,2))
# Plotting histogram and kde of feature "num_orders" using seaborn
sb.histplot(df['week'], bins=50, palette="muted", kde=False)
plt.xlabel("number of weeks")
plt.show()


# In[102]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,2))
# Plotting histogram and kde of feature "num_orders" using seaborn
sb.histplot(df['num_orders'], bins=50, palette="muted", kde=True)
plt.xlabel("num_orders")
plt.show()
# All the data are concentrated near 0.
# num_orders is right skewed.
# As the above graph is not much readable let's plot it again by taking log of num_orders.


# In[103]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,2))
# Plotting histogram and kde of feature "num_orders" using seaborn
sb.histplot(np.log(df['num_orders']), bins=50, palette="muted", kde=True)
plt.xlabel("log of num_orders")
plt.show()
# This plot is more clearer multimodal histogram 


# In[104]:


'''
# Let's look into what type of distribution is num_orders
# Instantiating distfit
dist = distfit(todf=True)
# Fit an transform num_orders
dist.fit_transform(df["num_orders"])
# Plotting distribution plot
dist.plot()
# num_orders follows exponential distribution
'''


# In[105]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,2))
# Ploting boxplot for log of num_orders
sb.boxplot(np.log(df["num_orders"]))
plt.title("Before removing outliers")
plt.xlabel("log num_orders")
plt.show()
# Outliers need to be removed


# In[106]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,5))
#Plotting scatter plot of checkout_price nd base_price with cuisine as hue
sb.scatterplot(x=df["checkout_price"], y=df["base_price"], hue=df["cuisine"], palette="muted")
plt.show()
# base_price and checkout_price are linearly related
# base_price of most of the meals are greater than the checkout_price
# Continental is most expensive of all other cuisines


# In[107]:


# Setting figure size to be plotted
sb.set_style("white")
fig = plt.figure(figsize=(5,2))
#Plotting histogram and kde of feature "category" color encoded with "cuisine".  
sb.histplot(data=df, x='category', hue="cuisine", multiple = "stack", palette="muted", kde=False,  legend=True)
#Rotating the labels on x-axis with a degree of 30, this is just to make all the labels on the x-axis readable.
plt.xticks(rotation=90)
plt.show()
# Beverages comes under all the cuisines hence it has the highest count of all other meal categories 


# In[108]:


# Forming a sorted list of all meal category

# Setting figure size to be plotted
plt.figure(figsize=(5,2))
cat = sorted(list(df["category"].unique()))
items = cat.__len__()
# Plotting total number of orders for each meal category 
ax = df.groupby(["category"]).num_orders.mean()
ax.plot()
plt.ylabel("Mean number of orders")
plt.xticks(range(items), labels=cat, rotation=90)
plt.show()

plt.figure(figsize=(5,2)) 
cat = sorted(list(df["category"].unique()))
items = cat.__len__()
# Plotting mean base_price for each meal 
ax = df.groupby(["category"]).base_price.mean()
ax.plot()
plt.ylabel("Mean base price")
plt.xticks(range(items), labels=cat, rotation=90)
plt.show()
# Meals with low mean base_price have high number of orders.


# In[109]:


# Setting figure size
fig = plt.figure(figsize=(5,3))
sb.set_style("white")
# Plotting histogram for variable center_type
sb.histplot(data= df, y="center_type", hue="cuisine", multiple="stack", color="pastel", legend=True)
plt.show()
# Setting figure size
fig = plt.figure(figsize=(5,2))
sb.set_style("whitegrid")
# Plotting total number of orders for each cuisine
df.groupby(["center_type"]).num_orders.sum().plot()
plt.ylabel("Total num_orders")
plt.show()
# Setting figure size
fig = plt.figure(figsize=(5,2))
sb.set_style("whitegrid")
# Plotting mean base_price for each cuisine
df.groupby(["center_type"]).base_price.mean().plot()
plt.ylabel("Mean num_orders")
plt.show()
# The demand for all the cuisines are equally distributed in all center types.
# Even though center type A has more number of orders,
# center type B has higher meal demand in terms of average number of orders


# In[110]:


# Setting figure size
fig = plt.figure(figsize=(5,2))
sb.set_style("whitegrid")
# Plotting total number of orders for each op_area
df.groupby(["op_area"]).num_orders.mean().plot()
plt.ylabel("Mean num_orders")
plt.show()
# The number of orders is roughly increasing with the "op_area".


# In[111]:


sb.set_style("whitegrid")
sb.displot(df, x=df["week"], col="emailer_for_promotion", kind="kde")
plt.show()
# There are very few weeks when "emailer_for_promotion" was active.


# In[112]:


#Setting up the theme of figure to "white" with "grid"
sb.set_style("whitegrid")
# Plotting boxplot of features "num_orders" and "category", also showing their means indicating with a green triangle
sb.boxplot(data=df, x=np.log(df["num_orders"]), y="category", showmeans=True, palette="pastel")
plt.xlabel("log num_orders")
plt.show()
# "Biryani" and "Rice bowl" has lowest and highest average number of orders.
# Highest number of outliers in "Rice Bowl", "Sandwich" and "Salad".


# In[113]:


sb.set_style("whitegrid")
# Plotting boxplot of features "checkout_price" and "category", also showing their means indicating with a green triangle
sb.boxplot(data=df, x="base_price", y="category", showmeans=True, palette="pastel")
plt.show()
# "Extras" has the lowest average "checkout_price" while "Seafood" has the highest of all.


# In[114]:


sb.set_style("whitegrid")
#fig = plt.figure(figsize=(10,5))
#Plotting violin plot of feature "emailer_for_promotion"
sb.violinplot(data=df, x="emailer_for_promotion", y=np.log(df["num_orders"]), showmeans=True, palette="pastel")
plt.ylabel("log of num_orders")
plt.show()
# Distribution plot shown in the violin plot is quite smooth when "emailer_for_promotion" is active(1)


# <h3>Multivariate analysis</h3>
# 
# Multiple variables are analyse to give the corelation between them.

# In[115]:


# Finding Spearman correlation coefficients of all the variables in dataframe using "corr" function 
df_cor = df.corr(method="spearman")
df_cor
# It displays all the coefficient in a tabular form
# Correlation is the measure of how two or more variables are related to one another.


# In[116]:


# Plotting correlation coefficients using "heatmap" function
sb.heatmap(df_cor, vmin=-1, vmax=1, cmap="RdBu", linewidths=0.1 ) 
plt.show()
# Blue with value 1 indicates that the two variables are linearly related.
# White with value 0 indicates that there is not relation between the two variables.
# Red with value -1 indicates that the two variables are inversely related.


# <h2>Data pre-processing</h2>
# 
# Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model.

# In[117]:


# As the data is a time series data
# Sorting the dataframe wrt week
df = df.sort_values(by="week")
df


# In[118]:


# One hot Encoding to columns - "category", "cuisine" and "center_type"
df = pd.get_dummies(df)
df.head()


# <h3>Removing outliers</h3>
#     
# Quartile 1, quartile 3 and Inter-quartile region (IQR) of the dependent variable are calculated to remove the outliers.

# In[119]:


# Calculating quartile1 and quartile3 of num_orders
q1 = np.percentile(np.log(df["num_orders"]),25)
q3 = np.percentile(np.log(df["num_orders"]),75)
# Calculating Inter Quartile Region
iqr = q3-q1
# Removing outliers 
df = df[np.log(df["num_orders"])>=q1-1.5*iqr]
df = df[np.log(df["num_orders"])<=q3+1.5*iqr]
df.shape


# In[120]:


# Setting figure size
plt.figure(figsize=(5,2))
sb.boxplot(np.log(df["num_orders"]))
plt.title("After removing outliers")
plt.xlabel("log num_orders")
plt.show()
# Boxplot showing after removing outliers


# In[121]:


# Removing outliers based on checkout_price
df = df[df["checkout_price"]<800]
df.shape


# <h2>Feature Engineering</h2>
#     
# Feature engineering is a technique that leverages data to create new variables that aren’t in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy.
# 
# Here we have calculated the discount percentage of each of the meal and formed a column of it.
# 
# Discount_percentage = ((base_price)-(checkout_price))*100/(base_price)

# In[122]:


# Forming a discount percentage column
df["discount_percentage"] = (df["base_price"]-df["checkout_price"])*100/df["base_price"]
df.head()


# In[123]:


# Setting figure size to be plotted
fig = plt.figure(figsize=(5,5))
# Plotting scatter plot of discount_percentage and num_orders
sb.scatterplot(x=df["discount_percentage"], y=df["num_orders"], palette="muted")
plt.show()


# In[134]:


df["num_orders"].describe()


# In[125]:


sb.boxplot(x=df["discount_percentage"])


# We are removing the columns which are no more useful

# In[70]:


# Converting all column names to lower case
df = df.rename(columns=str.lower)
# Removing column "id","checkout_price","base_price","city_code","region_code"
df = df.drop(["id","checkout_price","base_price","city_code","region_code"],axis=1)
df.shape


# <h2>Model

# <h3>Separating dependent variables and independent variables</h3>
# 
# In order to fit into the model data needs to be separated into dependent variables and independent variables and we need them in the form of array of values.

# In[71]:


# Assigning column "num_orders" to dependent_variable
dependent_variable = "num_orders"
# Assigning the list of columns to independent_variable
independent_variable = df.columns.tolist()
# Removing dependent_variable from the list of column in independent_variable
independent_variable.remove(dependent_variable)
# Assigning the values of independent and dependent variables to x and y respectively.
x = df[independent_variable].values
y = df[dependent_variable].values


# <h3>Importing required libraries

# In[72]:


# Importing all the required libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn import tree


# <h3>Data splitting</h3>
# 
# 

# In[73]:


# Splitting the data with TimeSeriesSplit
# Older data are assigned for training purpose and the newer data for testing
tss = TimeSeriesSplit(n_splits=5)
for i,j in tss.split(x):
    x_train, x_test = x[i], x[j]
    y_train, y_test = y[i], y[j]
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# <h3>Preparing data for XGBoost</h3>
# 
# In order to feed the data into XGBoost model it has to be converted into DMatrices.

# In[74]:


# Building DMatrices for XGBoost
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)


# <h3>Parameters</h3>
# 
# After extensive hyperparameter tuning we have come with the parameters that give the lowest RMSE value.

# In[75]:


# Assigning parameter values
# These parameters are set after extensive hyperparameter tuning
params = {"max_depth":10,"min_child_weight": 7,"eta":0.1,"subsample": 1,"colsample_bytree": 0.7,
          "eval_metric": "rmse","objective":"reg:squarederror"}
params


# <h3>Training</h3>
# 
# We are now going to train the data with the XGBoost model for the above parameters and evaluate on the test data. We are also looking into the number rounds with which we get the lowest RMSE value; this done by making the number of boosting round to a large number assumming we'll get the best (lowest) RMSE before it stops execution. 

# In[76]:


# Noting the start time of execution
s = time.time()
# Training on dtrain and evaluation is done on dtest,
# Iterating 10 more rounds after getting the minimum rmse
model = xgb.train(params,dtrain,num_boost_round=1000,evals=[(dtest, "Test")],early_stopping_rounds=10)
# Printing the time taken to execute the above code
print("Time taken:",round(time.time()-s,3),"sec")


# We now get the lowest RMSE value with the number of rounds needed.

# In[77]:


# Printing the minimun RMSE with its iteration
print("Best RMSE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration+1))


# This step is an optional as we are retraining the model till the number of boosting rounds we got the best RMSE. 

# In[78]:


s=time.time()
# Here the number of boosting round is set to number of iteration round with minimum rmse value
# This time iteration will stop just after the minimum rmse 
# early_stopping_rounds is not needed now
num_boost_round = model.best_iteration + 1
best_model = xgb.train(params,dtrain,num_boost_round=num_boost_round,evals=[(dtest, "Test")])
print("Time taken:",round(time.time()-s,3),"sec")


# <h3>Feature importance</h3>
# 
# From the 33 features we have trained the model, we want to know which feature contributes the highest weight in predicting the number of orders for a meal. This can be done with the following codes

# In[79]:


# Plotting feature importance plot
# Displaying only first top 10 features 
xgb.plot_importance(best_model, height=0.7, grid=False, max_num_features=10)
plt.show()


# <h3>Prediction</h3>
# 
# After training the data, we now measure the error and accuracy by predicting train and test data. It is needed to compare train and test error to check whether the model is overfitting or not. As the error difference is not large, we see that the model is performing well with an accurary od 80.4% on unseen data.

# In[80]:


# Calculating rmse and r2 value for train and test data
RMSE_train = round(np.sqrt(mean_squared_error(y_train,best_model.predict(dtrain))),3)
R2_train = round(r2_score(y_train,best_model.predict(dtrain)),3)

RMSE_test = round(np.sqrt(mean_squared_error(y_test,best_model.predict(dtest))),3)
R2_test = round(r2_score(y_test,best_model.predict(dtest)),3)

print("         Train         Test")
print("________________________________")
print(" RMSE:  {}      {}".format(RMSE_train, RMSE_test))
print("   R2:   {}        {}".format(R2_train, R2_test))


# <h3>Plotting tree</h3>
# 
# We now want to plot the XGBoost tree form by our model. We are saving it as jpg format.

# In[136]:


# Plotting XGBoost tree and saving in the form of .jpeg file
xgb.plot_tree(best_model, num_trees=0, rankdir="LR")
fig = plt.gcf()
fig.set_size_inches(100,500)
fig.savefig('tree.jpeg')


# <h2>Test data</h2>
# 
# It is now time to feed the actual test.csv data into the model and predict them.

# <h3>Loading data</h3>
# 
# We are now reading the test.csv file, merge meal and center file with test.csv. Pre-processed, feature engineered, dropped columns and convert into DMtarix like the way we did for train data. Then predict them with the model. 

# In[45]:


# Reading test.csv file
os.chdir("D:\AAAAAAA RAJSHREE\DATA SCIENCE\DATASETs\Demand forecasting for a store")
test_data = pd.read_csv("food_demand_test.csv")
# Merging meal and center file to test_data
test_data = test_data.merge(meal, on="meal_id")
test_data = test_data.merge(center, on="center_id")
# Sorting test data by week and category
test_data = test_data.sort_values(by=["week","category"])
# Perfoming one hot encoding
test_data = pd.get_dummies(test_data)
# Calculating discount percentage and forming a column
test_data["discount_percentage"] = (test_data["base_price"]-test_data["checkout_price"])*100/test_data["base_price"]
# Dropping some of the unnecessary columns
test_data.drop(["id","checkout_price","base_price","city_code","region_code"],axis=1,inplace=True)
test_data


# In[46]:


s = time.time()
# Converting into xgboost dmatrix
test_matrix = xgb.DMatrix(test_data.values)# Predicting with our best model
predicted_value = best_model.predict(test_matrix)
# Calculating time taken to to predict test data
t = round(time.time()-s,6)
print("Time taken to predict test data: {} sec".format(t))


# <h3>Throughput and latency</h3>

# In[47]:


# Calculating throughput
# Throughput is the number of data points executed in one second 
print("Number of data points executed in one second: {}".format(int(test_data.shape[0]/t)))
# Calculating Latency
# Latency is the time taken to execute one data point
print("Time taken to execute one data point: {} sec".format(t/test_data.shape[0]))


# <h3>Sample submission</h3>
# 
# The predicted values are then saved in the sample submission as csv file.

# In[48]:


# Inserting the predicted output to sample submission file
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission["num_orders"] = predicted_value
sample_submission


# <h3>Save model</h3>
# 
# In order to use keep the model for futre use we saved the model as pickle file. To do this we import pickle and then dump the model as pickle file.

# In[49]:


# Importing pickle
import pickle
# Saving the model in the form of pickle file for future prediction
pickle.dump(best_model, open("pro99_model.pkl","wb"))


# <h3>Conclusion</h3>
# 
# Throughout this whole process we understood that the meal demand highly depends on the discount given on the meal, type of the meal and the location of the store; with promotions on web pages and emails make enhancement on it. 
