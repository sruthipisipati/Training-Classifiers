#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:28:11 2018

@author: Sruthi Pisipati
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#-----------------------------------------------------------------#
#Importing the data from train & test csv files
#-----------------------------------------------------------------#
df_x_train = pd.read_csv("x_train.csv")
df_x_test = pd.read_csv("x_test.csv")
df_y_train = pd.read_csv("y_train.csv")
df_y_test = pd.read_csv("y_test.csv")
del df_x_train['Unnamed: 0'] # Remove the junk column
del df_x_test['Unnamed: 0'] # Remove the junk column
del df_y_train['Unnamed: 0'] # Remove the junk column
del df_y_test['Unnamed: 0'] # Remove the junk column
x_train = df_x_train.values 
y_train = df_y_train.values 
x_test = df_x_test.values   
y_test = df_y_test.values

##-----------------------------------------------------------------#
##Building KNN classifiers - Eucledian distance- Uniform weights
##-----------------------------------------------------------------#
resultsKNN_euu = pd.DataFrame(columns=['KNN','Score For Training','Score for Testing'])
for knnCount in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = knnCount,weights ='uniform',p = 2,metric = 'minkowski')
    knn.fit(x_train,y_train)
    scoreTrain = knn.score(x_train,y_train)
    scoreTest = knn.score(x_test,y_test)
    resultsKNN_euu.loc[knnCount]=[knnCount,scoreTrain,scoreTest]
print('================================================================')    
print('KNN Classifier - With Eucledian distance , Uniform Weights')    
print('================================================================')
print(resultsKNN_euu.head(11))
resultsKNN_euu.pop('KNN')
ax = resultsKNN_euu.plot(title = 'KNN - Eucledian-Uniform Weights')
fig = ax.get_figure()
fig.savefig('KNN - Eucledian_Uniform_Weights.png')
##-----------------------------------------------------------------#
##Building KNN classifiers - Eucledian distance-Weighted Distance
##-----------------------------------------------------------------#
resultsKNN_eud = pd.DataFrame(columns=['KNN','Score For Training','Score for Testing'])
for knnCount in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = knnCount,weights = 'distance',p = 2,metric = 'minkowski')
    knn.fit(x_train,y_train)
    scoreTrain = knn.score(x_train,y_train)
    scoreTest = knn.score(x_test,y_test)
    resultsKNN_eud.loc[knnCount]=[knnCount,scoreTrain,scoreTest]
print('================================================================')    
print('KNN Classifier - With Eucledian distance , Weighted Distance')    
print('================================================================')
print(resultsKNN_eud.head(11))
resultsKNN_eud.pop('KNN')
ax = resultsKNN_eud.plot(title = 'KNN - Eucledian-Weighted Distance')
fig = ax.get_figure()
fig.savefig('KNN - Eucledian_Weighted_Distance.png')
##-----------------------------------------------------------------#
##Building KNN classifiers - Manhattan Distance - Uniform weights
##-----------------------------------------------------------------#
resultsKNN_mnu = pd.DataFrame(columns=['KNN','Score For Training','Score for Testing'])
for knnCount in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = knnCount,weights ='uniform',p = 1,metric = 'minkowski')
    knn.fit(x_train,y_train)
    scoreTrain = knn.score(x_train,y_train)
    scoreTest = knn.score(x_test,y_test)
    resultsKNN_mnu.loc[knnCount]=[knnCount,scoreTrain,scoreTest]
print('================================================================')    
print('KNN Classifier - With Manhattan distance , Uniform Weights')    
print('================================================================')
print(resultsKNN_mnu.head(11))
resultsKNN_mnu .pop('KNN')
ax = resultsKNN_mnu.plot(title = 'KNN - Manhattan-Uniform Weights')
fig = ax.get_figure()
fig.savefig('KNN - Manhattan_Uniform_Weights.png')
##-----------------------------------------------------------------#
##Building KNN classifiers - Manhattan Distance - Weighted distance
##-----------------------------------------------------------------#
resultsKNN_mnd = pd.DataFrame(columns=['KNN','Score For Training','Score for Testing'])
for knnCount in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = knnCount,weights = 'distance',p = 1,metric = 'minkowski')
    knn.fit(x_train,y_train)
    scoreTrain = knn.score(x_train,y_train)
    scoreTest = knn.score(x_test,y_test)
    resultsKNN_mnd.loc[knnCount]=[knnCount,scoreTrain,scoreTest]
print('================================================================')    
print('KNN Classifier - With Manhattan distance , Weighted Distance')    
print('================================================================')
print(resultsKNN_mnd.head(11))
resultsKNN_mnd .pop('KNN')
ax = resultsKNN_mnd.plot(title = 'KNN - Manhattan-Weighted Distance')
fig = ax.get_figure()
fig.savefig('KNN - Manhattan_Weighted_Distance.png')



