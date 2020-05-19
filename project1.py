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
cars_data = pd.read_csv('cars_sampled.csv')
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
sns.distplot(cars['Price'])
sns.boxplot(y = cars['Price'])

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
