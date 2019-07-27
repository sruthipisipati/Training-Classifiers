#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:40:11 2018

@author: Sruthi Pisipati
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#-----------------------------------------------------------------#
#Importing the data from wineNormalized.csv file into dataframe df
#-----------------------------------------------------------------#
df = pd.read_csv("wineNormalized.csv")
del df['Unnamed: 0'] # Remove the junk column
print(df.head(5))
print(df.dtypes)
print(df.describe())
# Since there are zeros, plain ratios might not be the choice for derived attributes
#
# Taking few Aggregates as derived attributes
df['Malic acid+Ash'] = df['Malic acid']+df['Ash']
df['Alcalinity of ash+Magnesium'] = df['Alcalinity of ash']+df['Magnesium']
df['Magnesium+Total phenols'] = df['Magnesium']+df['Total phenols']
df['Nonflavanoid phenols+Flavanoids'] = df['Nonflavanoid phenols']+df['Flavanoids']

#Moving Class column to the end
tempCol = df.pop('Class')
df['Class'] = tempCol
print(df.head(5))
print(df.shape)

# Check for nulls if any
print(df.isnull().any())
#-----------------------------------------------------------------#
#Preparing Test & Train split with test_size = 1/3 & 
#train_size = 2/3 , with stratified sampling
#-----------------------------------------------------------------#
x,y = df.iloc[:,0:17].values,df['Class'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0,stratify = y)

#-----------------------------------------------------------------#
#Building Decision Tree Classifier - With Criterion Entropy
# along with derived attributes    
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
print('Decision Tree Classifier - With Criterion Entropy- Derived Features')    
print('================================================================')
print(resultsEntropy.head(11))
resultsEntropy.pop('LevelLimit')
ax = resultsEntropy.plot(title = 'DCT-Entropy-DerviedFeatures')
fig = ax.get_figure()
fig.savefig('DCT-Entropy-DerviedFeatures.png')

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
print('KNN Classifier - With Eucledian distance , Uniform Weights- Derived Features')    
print('================================================================')
print(resultsKNN_euu.head(11))
resultsKNN_euu.pop('KNN')
ax = resultsKNN_euu.plot(title = 'KNN - Eucledian-Uniform Weights-Dervied Features')
fig = ax.get_figure()
fig.savefig('KNN - Eucledian_Uniform_Weights_DerivedFeatures.png')

###-----------------------------------------------------------------#
### Assessing feature importance with Random Forests
###-----------------------------------------------------------------#
feat_labels= df.columns[0:17]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(x_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print('====================================')
print('Ranking of Features by RandomForest')
print('====================================')
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f"% (f+1,30,feat_labels[f],importances[indices[f]]))
    
###-----------------------------------------------------------------#
### Now Building classifiers taking the top 4 features only
###-----------------------------------------------------------------#    
x,y = df.iloc[:,0:4].values,df['Class'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0,stratify = y)

#-----------------------------------------------------------------#
#Building Decision Tree Classifier - With Criterion Entropy
# along with top 5 derived attributes    
#-----------------------------------------------------------------#
resultsEntropy_DerT5 = pd.DataFrame(columns=['LevelLimit','Score For Training','Score for Testing'])
for treedepth in range(1,12):
    dct = DecisionTreeClassifier(criterion='entropy',max_depth=treedepth,random_state=0)
    dct = dct.fit(x_train,y_train)
    dct.predict(x_test)
    scoreTrain = dct.score(x_train,y_train)
    scoreTest = dct.score(x_test,y_test)
    resultsEntropy_DerT5.loc[treedepth]=[treedepth,scoreTrain,scoreTest]
print('================================================================')    
print('Decision Tree Classifier - With Criterion Entropy - Derived Features- Top 5')    
print('================================================================')
print(resultsEntropy_DerT5.head(11))
resultsEntropy_DerT5.pop('LevelLimit')
ax = resultsEntropy_DerT5.plot(title = 'DCT-Entropy-DerviedFeatures-Top 5')
fig = ax.get_figure()
fig.savefig('DCT-Entropy-DerviedFeatures-Top5.png')

##-----------------------------------------------------------------#
##Building KNN classifiers - Eucledian distance- Uniform weights
## along with top 4 derived attributes
##-----------------------------------------------------------------#
resultsKNN_euu_DerT5 = pd.DataFrame(columns=['KNN','Score For Training','Score for Testing'])
for knnCount in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = knnCount,weights ='uniform',p = 2,metric = 'minkowski')
    knn.fit(x_train,y_train)
    scoreTrain = knn.score(x_train,y_train)
    scoreTest = knn.score(x_test,y_test)
    resultsKNN_euu_DerT5.loc[knnCount]=[knnCount,scoreTrain,scoreTest]
print('================================================================')    
print('KNN Classifier - With Eucledian distance , Uniform Weights- Derived Features- Top 5')    
print('================================================================')
print(resultsKNN_euu_DerT5.head(11))
resultsKNN_euu_DerT5.pop('KNN')
ax = resultsKNN_euu_DerT5.plot(title = 'KNN - Eucledian-Uniform Weights-Dervied Features- Top 5')
fig = ax.get_figure()
fig.savefig('KNN - Eucledian_Uniform_Weights_DerivedFeatures.png')
##-----------------------------------------------------------------#
##Testing with same Testing & Training data sets
##-----------------------------------------------------------------#

x,y = df.iloc[:,0:17].values,df['Class'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0,stratify = y)
x_train_fair = x_train[:,:5]
x_test_fair = x_test[:,:5]
y_train_fair = y_train
y_test_fair  = y_test


print(x_train_fair.shape)
print(x_test_fair.shape)
print(y_train_fair.shape)
print(y_test_fair.shape)


#-----------------------------------------------------------------#
#Building Decision Tree Classifier - With Criterion Entropy
# along with same datasets   
#-----------------------------------------------------------------#
resultsEntropy_fair = pd.DataFrame(columns=['LevelLimit','Score For Training','Score for Testing'])
for treedepth in range(1,12):
    dct = DecisionTreeClassifier(criterion='entropy',max_depth=treedepth,random_state=0)
    dct = dct.fit(x_train_fair,y_train_fair)
    dct.predict(x_test_fair)
    scoreTrain = dct.score(x_train_fair,y_train_fair)
    scoreTest = dct.score(x_test_fair,y_test_fair)
    resultsEntropy_fair.loc[treedepth]=[treedepth,scoreTrain,scoreTest]
print('================================================================')    
print('Decision Tree Classifier - With Criterion Entropy-Fair')    
print('================================================================')
print(resultsEntropy_fair.head(11))
resultsEntropy_fair.pop('LevelLimit')
ax = resultsEntropy_fair.plot(title = 'DCT-Entropy-DerviedFeatures-Fair')
fig = ax.get_figure()
fig.savefig('DCT-Entropy-DerviedFeatures-Fair.png')

























