#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:27:21 2018

@author: Sruthi Pisipati
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#-------------------------------------------------------------#
#Importing the data from dataPrep.csv file into dataframe df
#-------------------------------------------------------------#
df = pd.read_csv("wineData.csv")
print('===== Details about Dataset ========')
print(df.describe()) #Deatils about Dataset
print(df.shape)
print('====================================')
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

#-----------------------------------------------------------------#
#Using Random Forest to rank descriptive features
#-----------------------------------------------------------------#
feat_labels= df.columns[1:14]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(x_train,y_train.ravel())
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print('====================================')
print('Ranking of Features by RandomForest')
print('====================================')
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f"% (f+1,30,feat_labels[f],importances[indices[f]]))
##-----------------------------------------------------------------#
##Building Random Forest Classifier
##-----------------------------------------------------------------#    
results_RF = pd.DataFrame(columns=['Count Of Trees','Score For Training','Score for Testing'])
indexR = 1
for sizeofForest in range(1,500,10):
    forest = RandomForestClassifier(criterion='gini',n_estimators = sizeofForest,max_depth=3)
    forest.fit(x_train,y_train.ravel())
    scoreTrain = forest.score(x_train,y_train)
    scoreTest = forest.score(x_test,y_test)
    results_RF.loc[indexR]=[sizeofForest,scoreTrain,scoreTest]
print('================================================================')    
print('Random Forest with different count of Trees')    
print('================================================================')
print(results_RF.head(11))
results_RF.pop('Count Of Trees')
ax = results_RF.plot(title = 'Random Forest with different Trees')
fig = ax.get_figure()
fig.savefig('Random Forest with different Trees.png')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    