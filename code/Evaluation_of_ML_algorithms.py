#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:52:46 2020

@author: shawvey
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_roc_curve(fpr1, tpr1, fpr2, tpr2):
    plt.plot(fpr1, tpr1, color='tomato', label='ROC_Testset')
    plt.plot(fpr2, tpr2, color='steelblue', label='ROC_Trainset')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

dataset = pd.read_csv('imputated_data.csv')
X = dataset.iloc[:, 1:-2]
Y = dataset.iloc[:, -1]
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
acc_scorer = make_scorer(accuracy_score)

# RandomForest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
#parameters = { 'max_features':np.arange(1,10),'n_estimators':[100],'min_samples_leaf':[1,10,50,100,200,500],'max_depth': range(1,15,2)}
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
grid = GridSearchCV(rf, parameters, cv = 5, scoring=acc_scorer)
grid.fit(xtrain,ytrain)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
rf = grid.best_estimator_
rf.fit(xtrain,ytrain)
probs1 = rf.predict_proba(xtest)
probs1 = probs1[:, 1]
probs2 = rf.predict_proba(xtrain)
probs2 = probs2[:, 1]
auc1 = roc_auc_score(ytest, probs1)
print('AUC_Testset: %.2f' % auc1)
auc2 = roc_auc_score(ytrain, probs2)
print('AUC_Trainset: %.2f' % auc2)
fpr1, tpr1, thresholds1 = roc_curve(ytest, probs1)
fpr2, tpr2, thresholds2 = roc_curve(ytrain, probs2)
plot_roc_curve(fpr1, tpr1, fpr2, tpr2)


"""

# KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier 
model=KNeighborsClassifier(n_neighbors=5)
#print(model)
#print(predicted)
k_range = list(range(1,31))
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(model, param_grid, cv=5, scoring=acc_scorer)
grid.fit(xtrain,ytrain)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
model = grid.best_estimator_
model.fit(xtrain,ytrain)
probs1 = model.predict_proba(xtest)
probs1 = probs1[:, 1]
probs2 = model.predict_proba(xtrain)
probs2 = probs2[:, 1]
auc1 = roc_auc_score(ytest, probs1)
print('AUC_Testset: %.2f' % auc1)
auc2 = roc_auc_score(ytrain, probs2)
print('AUC_Trainset: %.2f' % auc2)
fpr1, tpr1, thresholds1 = roc_curve(ytest, probs1)
fpr2, tpr2, thresholds2 = roc_curve(ytrain, probs2)
plot_roc_curve(fpr1, tpr1, fpr2, tpr2)
"""

"""
# DecisionTree 
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
acc_scorer = make_scorer(accuracy_score)
param_grid = {'max_depth': np.arange(3, 15),'criterion':['gini','entropy']}
grid = GridSearchCV(clf, param_grid, scoring=acc_scorer, cv=5)
grid.fit(xtrain,ytrain)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
clf = grid.best_estimator_
clf.fit(xtrain,ytrain)
probs1 = clf.predict_proba(xtest)
probs1 = probs1[:, 1]
probs2 = clf.predict_proba(xtrain)
probs2 = probs2[:, 1]
auc1 = roc_auc_score(ytest, probs1)
print('AUC_Testset: %.2f' % auc1)
auc2 = roc_auc_score(ytrain, probs2)
print('AUC_Trainset: %.2f' % auc2)
fpr1, tpr1, thresholds1 = roc_curve(ytest, probs1)
fpr2, tpr2, thresholds2 = roc_curve(ytrain, probs2)
plot_roc_curve(fpr1, tpr1, fpr2, tpr2)

"""

"""
# naive_bayes 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
#print(model)
#print(predicted)
parameters = {}
grid = GridSearchCV(model, parameters, scoring=acc_scorer, cv=5)
grid.fit(xtrain, ytrain)
model = grid.best_estimator_
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
model.fit(xtrain, ytrain)
probs1 = model.predict_proba(xtest)
probs1 = probs1[:, 1]
probs2 = model.predict_proba(xtrain)
probs2 = probs2[:, 1]
auc1 = roc_auc_score(ytest, probs1)
print('AUC_Testset: %.2f' % auc1)
auc2 = roc_auc_score(ytrain, probs2)
print('AUC_Trainset: %.2f' % auc2)
fpr1, tpr1, thresholds1 = roc_curve(ytest, probs1)
fpr2, tpr2, thresholds2 = roc_curve(ytrain, probs2)
plot_roc_curve(fpr1, tpr1, fpr2, tpr2)
"""

"""
# svm 
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear',probability=True)
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
grid = GridSearchCV(svclassifier, param_grid,cv=5, scoring=acc_scorer)
grid.fit(xtrain,ytrain)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
model = grid.best_estimator_
model.fit(xtrain,ytrain)
probs1 = model.predict_proba(xtest)
probs1 = probs1[:, 1]
probs2 = model.predict_proba(xtrain)
probs2 = probs2[:, 1]
auc1 = roc_auc_score(ytest, probs1)
print('AUC_Testset: %.2f' % auc1)
auc2 = roc_auc_score(ytrain, probs2)
print('AUC_Trainset: %.2f' % auc2)
fpr1, tpr1, thresholds1 = roc_curve(ytest, probs1)
fpr2, tpr2, thresholds2 = roc_curve(ytrain, probs2)
plot_roc_curve(fpr1, tpr1, fpr2, tpr2)
"""
