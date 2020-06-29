#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:45:02 2020

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

dataset = pd.read_csv('imputated_newdata.csv')
X = dataset.iloc[:, 1:-2]
Y = dataset.iloc[:, -1]
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)
acc_scorer = make_scorer(accuracy_score)

"""
# RandomForest 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=5, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, n_estimators=9,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
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

"""
# KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier 
model=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=28, p=2,
                     weights='uniform')
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
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=6,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
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
model = GaussianNB(priors=None, var_smoothing=1e-09)
#print(model)
#print(predicted)
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

model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
    verbose=False)
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