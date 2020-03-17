# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:57:57 2020

@author: Haniye
"""

#importing libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv('data_cleaned.csv')

#seperating independent and dependent variables
y = data['Survived']
X = data.drop(['Survived'], axis=1)

#importing train_test_split to create validation set
from sklearn.model_selection import train_test_split

#split the train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 101, stratify=y, test_size=0.25)

#importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier

# how to import decision tree regressor
from sklearn.tree import DecisionTreeRegressor

#creating the decision tree function
dt_model = DecisionTreeClassifier(random_state=10)

#fitting the model
dt_model.fit(X_train, y_train)

#checking the training score
dt_model.score(X_train, y_train)

#checking the validation score
dt_model.score(X_valid, y_valid)

#predictions on validation set
dt_model.predict(X_valid)

dt_model.predict_proba(X_valid)

y_pred = dt_model.predict_proba(X_valid)[:,1]

#assign 0 for probability less than 0.5 and 1 for probability greater than 0.5
y_new = []
for i in range(len(y_pred)):
    if y_pred[i]<=0.5:
        y_new.append(0)
    else:
        y_new.append(1)

from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_new)

#Changing the max_depth 
#in this step we want to limit the depth of tree to see which 
#depth has more accurate results 

train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    dt_model = DecisionTreeClassifier(max_depth=depth, random_state=10)
    dt_model.fit(X_train, y_train)
    train_accuracy.append(dt_model.score(X_train, y_train))
    validation_accuracy.append(dt_model.score(X_valid, y_valid))

frame = pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'valid_acc':validation_accuracy})
frame.head()

plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'], frame['train_acc'], marker='o')
plt.plot(frame['max_depth'], frame['valid_acc'], marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend()

#this one is limiting both depth and number of leaf nodes
dt_model = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=25, random_state=10)

#fitting the model
dt_model.fit(X_train, y_train)

#Training score
dt_model.score(X_train, y_train)

#Validation score
dt_model.score(X_valid, y_valid)

from sklearn import tree

decision_tree = tree.export_graphviz(dt_model,out_file='tree.dot',feature_names=X_train.columns,max_depth=2,filled=True)

image = plt.imread('tree.png')
plt.figure(figsize=(15,15))
plt.imshow(image)

