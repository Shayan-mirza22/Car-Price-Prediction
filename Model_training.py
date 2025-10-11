from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np 
import joblib
import json
import pickle

X = joblib.load('X.pkl')
Y = joblib.load('Y.pkl')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)    # test_size gives the % for splitting data into training data and test data. 
lr_clf = LinearRegression()       
lr_clf.fit(X_train, Y_train)
print(lr_clf.score(X_test, Y_test))