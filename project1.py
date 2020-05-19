#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:41:20 2019

@author: mukund
"""

import pandas as pd
import numpy as np
import seaborn as sns

# seting dimension for plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

# read csv file and create a copy
cars_data = pd.read_csv("cars_sampled.csv")
cars = cars_data.copy()

# summarize the dataset
cars.info()
pd.set_option('max_columns',500)
cars.describe()

# remove unwanted columns and duplicate data
cols = ['dateCrawled','lastSeen','postalCode','dateCreated','name']
cars = cars.drop(columns = cols,axis=1)
cars.drop_duplicates(keep = 'first', inplace = True)
cars.info()

# data cleaning
#==============================================================================

#missing values
cars.isnull().sum()

# variable year of registration
sum(cars['yearOfRegistration'] > 2018)
sum(cars['yearOfRegistration'] < 1950)
# working range 1950-2018

# variable price
sum(cars['price'] < 100)
sum(cars['price'] > 150000)
# working range 100-150000


# variable powerPS
sum(cars['powerPS'] < 10)
sum(cars['powerPS'] > 500)
# working range 10-500



# working range of data
cars = cars[(cars.yearOfRegistration >= 1950) & (cars.yearOfRegistration <= 2018) &
            (cars.price >= 100) & (cars.price <= 150000) & (cars.powerPS >= 10) &
            (cars.powerPS <= 500)]


cars['monthOfRegistration'] /= 12

# create new variable age by combining year and month of registration
cars['Age'] = (2018-cars['yearOfRegistration'] + cars['monthOfRegistration'])
cars['Age'] = round(cars['Age'],2)


# drop year and month of registration
cars = cars.drop(columns = ['monthOfRegistration','yearOfRegistration'],axis = 1)


# visualising parameters
#==============================================================================

# age
sns.distplot(cars['Age'])
sns.boxplot(y = cars['Age'])

# price
sns.distplot(cars['price'])
sns.boxplot(y = cars['price'])

# powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y = cars['powerPS'])

# age vs price
sns.regplot(x='Age', y='price', scatter=True, fit_reg = False, data=cars)

# powerPS vs price
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg = False, data=cars)

# variable offerType 
cars['offerType'].value_counts()
# only one type so drop it

# variable abtest
cars['abtest'].value_counts()
sns.boxplot(y='price', x='abtest', data=cars)
# both have same effect hence insignificant
sns.countplot(x='abtest', data=cars)

# variable seller
cars['seller'].value_counts()
# all except one private, hence drop it

# variable fuelType
cars['fuelType'].value_counts()
sns.countplot(x='fuelType', data=cars)
sns.boxplot(y='price', x='fuelType', data=cars)

# variable gearbox
cars['gearbox'].value_counts()
sns.countplot(x='gearbox', data=cars)
sns.boxplot(y='price', x='gearbox', data=cars)

# variable vehicleType
cars['vehicleType'].value_counts()
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(y='price', x='vehicleType', data=cars)

# variable kilometer
cars['kilometer'].value_counts()
sns.countplot(x='kilometer', data=cars)
sns.boxplot(y='price', x='kilometer', data=cars)

# variable brand
cars['brand'].value_counts()
sns.countplot(x='brand', data=cars)
sns.boxplot(y='price', x='brand', data=cars)

# variable model
cars['model'].value_counts()
sns.countplot(x='model', data=cars)
sns.boxplot(y='price', x='model', data=cars)

# variable notRepairedDamage
cars['notRepairedDamage'].value_counts()
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(y='price', x='notRepairedDamage', data=cars)


# removing insignificant variables
col = ['offerType','seller', 'abtest']
cars = cars.drop(columns = col, axis = 1)
cars_copy = cars.copy()


# Correlation between numerical variables
cars_select1 =  cars.select_dtypes(exclude=[object])
correlation = cars_select1.corr()
round(correlation, 3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


# Omitting missing values
cars_omit = cars.dropna(axis=0)

#Converting categorical variables to dummy variables
cars_omit = pd.get_dummies(cars_omit, drop_first=True)


# importing libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# model building with omitted data

# separating input and output features
x1 = cars_omit.drop(['price'], axis= 'columns', inplace = False)
y1 = cars_omit['price']

# plotting the variable price
prices = pd.DataFrame({"1 Before": y1, "2 After": np.log(y1)})
prices.hist()

#transforming price as a logarithmic value to get rid of skewness
y1 = np.log(y1)
 
# Splitting into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Baseline model for omitted data
# This is to set a benchmark and compare our regression models
# We are building base model using mean of y_test 

base_pred = np.mean(y_test)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred,  len(y_test))

# finding the rmse
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)


# Built models should give rmse less than that obtained above

# Building Linear Regression Model

lgr = LinearRegression(fit_intercept=True)
model_ln1 = lgr.fit(X_train, y_train)
cars_pred_lr1 = lgr.predict(X_test)

# Computing RMSE and MSE

lin_mse1 = mean_squared_error(y_test, cars_pred_lr1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

# R squared value
r2_lin_test1 = model_ln1.score(X_test, y_test)
r2_lin_train1 = model_ln1.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train1)

# Regression diagnostics-Residual plot analysis
residuals1 = y_test-cars_pred_lr1
sns.regplot(x=cars_pred_lr1, y=residuals1, scatter=True, fit_reg=False)
residuals1.describe()

# Random forest with omitted data

# model parameters
rf = RandomForestRegressor(n_estimators=100, max_depth = 100, max_features='auto',
                           min_samples_split=10, min_samples_leaf=4,random_state=1)
model_rf1 = rf.fit(X_train, y_train)

# Predicting model on test data
cars_predictions_rf1 = rf.predict(X_test)

# computing mse and rmse

rf1_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf1_rmse1 = np.sqrt(rf1_mse1)
print(rf1_mse1, rf1_rmse1)

# R squared value
r2_rf_test1 = model_rf1.score(X_test, y_test)
r2_lin_train1 = model_rf1.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train1)




 



