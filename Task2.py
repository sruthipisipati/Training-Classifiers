#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:30:02 2018

@author: Sruthi Pisipati
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
#-----------------------------------------------------------------#
#Importing the data from wineNormalized.csv file into dataframe df
#-----------------------------------------------------------------#
df = pd.read_csv("wineNormalized.csv")
del df['Unnamed: 0'] # Remove the junk column
print(df.head(5))
print(df.dtypes)
print(df.iloc[:,0:13])
#-----------------------------------------------------------------#
#Preparing Test & Train split with test_size = 1/3 & 
#train_size = 2/3 , with stratified sampling
#-----------------------------------------------------------------#
x,y = df.iloc[:,0:13].values,df['Class'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0,stratify = y)
#-----------------------------------------------------------------#
#Saving Train & Test files to csv 
#-----------------------------------------------------------------#
df_x_train = pd.DataFrame(x_train)
df_x_test = pd.DataFrame(x_test)
df_y_train = pd.DataFrame(y_train)
df_y_test  = pd.DataFrame(y_test)

df_x_train.to_csv("x_train.csv")
df_y_train.to_csv("y_train.csv")
df_x_test.to_csv("x_test.csv")
df_y_test.to_csv("y_test.csv")
#-----------------------------------------------------------------#
#Building Decision Tree Classifier - With Criterion Entropy
#-----------------------------------------------------------------#
resultsEntropy = pd.DataFrame(columns=['LevelLimit','Score For Training','Score for Testing'])
for treedepth in range(1,12):
    dct = DecisionTreeClassifier(criterion='entropy',max_depth=treedepth,random_state=0)
    dct = dct.fit(x_train,y_train)
    dct.predict(x_test)
    scoreTrain = dct.score(x_train,y_train)
    scoreTest = dct.score(x_test,y_test)
    resultsEntropy.loc[treedepth]=[treedepth,scoreTrain,scoreTest]
print('================================================================')    
print('Decision Tree Classifier - With Criterion Entropy')    
print('================================================================')
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax = resultsEntropy.plot(title = 'DCT-Entropy')
fig = ax.get_figure()
fig.savefig('DCT-Entropy.png')
#-----------------------------------------------------------------#
#Building Decision Tree Classifier - With Criterion Gini index
#-----------------------------------------------------------------#
resultsGini = pd.DataFrame(columns=['LevelLimit','Score For Training','Score for Testing'])
for treedepth in range(1,12):
    dct = DecisionTreeClassifier(criterion='gini',max_depth=treedepth,random_state=0)
    dct = dct.fit(x_train,y_train)
    dct.predict(x_test)
    scoreTrain = dct.score(x_train,y_train)
    scoreTest = dct.score(x_test,y_test)
    resultsGini.loc[treedepth]=[treedepth,scoreTrain,scoreTest]
print('================================================================')    
print('Decision Tree Classifier - With Criterion Gini')      
print('================================================================')
print(resultsGini.head(11))
resultsGini.pop('LevelLimit')
ax = resultsGini.plot(title = 'DCT-Gini')
fig = ax.get_figure()
fig.savefig('DCT-Gini.png')
#-----------------------------------------------------------------#
#Displaying Best Decision Tree
#-----------------------------------------------------------------#
export_graphviz(dct,out_file='tree.dot')
# Command used for conversion of dot file to png image from command prompt
# dot -Tpng tree.dot -o tree.png
