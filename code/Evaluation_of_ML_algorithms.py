#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:52:46 2020

@author: shawvey
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

dataset = pd.read_csv('imputated_data.csv')
X = dataset.iloc[:, 0:-1]
Y = dataset.iloc[:, -1]
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3)

# RandomForest 随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(xtrain,ytrain)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0, n_estimators=100, n_jobs=1,
            oob_score=True, random_state=None, verbose=0,warm_start=False)
predicted=rf.predict(xtest)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))

"""

# KNeighborsClassifier K近邻算法
from sklearn.neighbors import KNeighborsClassifier 
model=KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain,ytrain)
#print(model)
predicted=model.predict(xtest)
#print(predicted)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""

"""
# DecisionTree 决策树算法
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(xtrain,ytrain)
predicted = clf.predict(xtest)
print(predicted)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""

"""
# naive_bayes 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(xtrain,ytrain)
#print(model)
predicted=model.predict(xtest)
#print(predicted)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""

"""
# simple svm 简单支持向量机
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(xtrain,ytrain)
predicted = svclassifier.predict(xtest)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""

"""
# 多项式核 svm
from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(xtrain,ytrain)
predicted = svclassifier.predict(xtest)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""

"""
# 高斯核 svm
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(xtrain,ytrain)
predicted = svclassifier.predict(xtest)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""
"""
# Sigmoid核 svm
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(xtrain,ytrain)
predicted = svclassifier.predict(xtest)
print('The Accuracy is %f'%accuracy_score(ytest,predicted))
print(confusion_matrix(ytest,predicted))
print(classification_report(ytest,predicted))
"""