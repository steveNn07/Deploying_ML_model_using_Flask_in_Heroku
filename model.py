#!/usr/bin/env python --->o/n: prints after exporting from a notebook.Not required
# coding: utf-8 --->o/n: prints after exporting from a notebook.Not required

#Import libraries
import pandas as pd
import numpy as np
import pickle

#Read data
data = pd.read_csv('housing.csv')
data.head(5)


#Building model
###Split data
from sklearn.model_selection import train_test_split
X = data[['RM','LSTAT','PTRATIO']]
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=21)


#Import linear mode
from sklearn.linear_model import LinearRegression
###Create an instance of Linear regression model
lm = LinearRegression()
###train/fit lm on the training data
lm.fit(X_train, y_train)

#Predicting using test data
###use lm.predict
#predictions = lm.predict(X_test)

#serializing/saving model
model = pickle.dump(lm, open('model.pkl', 'wb'))

#loading model
#model = pickle.load(open('model.pkl', 'rb'))






