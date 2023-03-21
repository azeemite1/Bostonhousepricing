# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:39:59 2023

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Lets load the Boston House Pricing Dataset
from sklearn.datasets import load_boston
boston=load_boston()
boston.keys()
## Lets check the description of the dataset
print(boston.DESCR)
print(boston.data)
print(boston.target)
print(boston.feature_names)
## Preparing The Dataset
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)
dataset.head()
dataset['Price']=boston.target
dataset.head()
dataset.info()
## Summarizing The Stats of the data
dataset.describe()
## Check the missing Values
dataset.isnull().sum()
### EXploratory Data Analysis
## Correlation
dataset.corr()
import seaborn as sns
sns.pairplot(dataset)
## Analyzing The Correlated Features
dataset.corr()
plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")
plt.scatter(dataset['RM'],dataset['Price'])
plt.xlabel("RM")
plt.ylabel("Price")
import seaborn as sns
sns.regplot(x="RM",y="Price",data=dataset)
sns.regplot(x="LSTAT",y="Price",data=dataset)
sns.regplot(x="CHAS",y="Price",data=dataset)
sns.regplot(x="PTRATIO",y="Price",data=dataset)
## Independent and Dependent features

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
X.head()
y
##Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train
X_test
## Standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))
X_train
X_test
## Model Training
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)
## print the coefficients and the intercept
print(regression.coef_)
print(regression.intercept_)
## on which parameters the model has been trained
regression.get_params()
### Prediction With Test Data
reg_pred=regression.predict(X_test)
reg_pred
## Assumptions
## plot a scatter plot for the prediction
plt.scatter(y_test,reg_pred)
## Residuals
residuals=y_test-reg_pred
residuals
## Plot this residuals 

sns.displot(residuals,kind="kde")
## Scatter plot with respect to prediction and residuals
## uniform distribution
plt.scatter(reg_pred,residuals)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test,reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))
## R square and adjusted R square

#Formula

## R^2 = 1 - SSR/SST**


#R^2	=	coefficient of determination
#SSR	=	sum of squares of residuals
#SST	=	total sum of squares

from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)

## Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]**

#where:

#R2: The R2 of the model
#n: The number of observations
#k: The number of predictor variables
#display adjusted R-squared
1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
## New Data Prediction
boston.data[0].reshape(1,-1)
##transformation of new data
scaler.transform(boston.data[0].reshape(1,-1))
regression.predict(scaler.transform(boston.data[0].reshape(1,-1)))
## Pickling The Model file For Deployment
import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))
pickled_model=pickle.load(open('regmodel.pkl','rb'))
## Prediction
pickled_model.predict(scaler.transform(boston.data[0].reshape(1,-1)))
