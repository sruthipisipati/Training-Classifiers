#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:55:45 2018

@author: Sruthi Pisipati
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

#-------------------------------------------------------------#
#Importing the data from dataPrep.csv file into dataframe df
#-------------------------------------------------------------#
df = pd.read_csv("wineData.csv")
print('===== Details about Dataset ========')
print(df.describe()) #Deatils about Dataset
print(df.shape)
print('====================================')
#-------------------------------------------------------------#
#Identifying Class attribute using dtype
#Column which contains "object" as data type is class
#-------------------------------------------------------------#
print('===== Data Frame Datatypes ========')
print(df.dtypes)
print('====================================')
#-------------------------------------------------------------#
#Class mapping
#-------------------------------------------------------------#
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['Class']))}
df['Class'] = df['Class'].map(class_mapping)
#-------------------------------------------------------------#
#Normalization in the range (0,2)
#-------------------------------------------------------------#
dfTemp = df.iloc[:,1:14]
print('===== Data Frame values before Normalization ========')
print(dfTemp.head(5))
print('====================================')
x = df.iloc[:,1:14].values
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 2))
x_scaled = min_max_scaler.fit_transform(x)
NormalizedDF = pd.DataFrame(x_scaled,columns = dfTemp.columns)
NormalizedDF['Class'] = df['Class']
print('===== Normalized Data Frame ========')
print(NormalizedDF.head(11))
print('====================================')
#-------------------------------------------------------------#
# Save to csv
#-------------------------------------------------------------#
NormalizedDF.to_csv("wineNormalized.csv")
print("Saved to csv file")
#-------------------------------------------------------------#
# Check if classes are balanced
#-------------------------------------------------------------#
NormalizedDF['Class'].value_counts().plot(kind='bar')